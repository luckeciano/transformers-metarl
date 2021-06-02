"""GaussianMLPPolicy."""
import torch
from torch import nn
import math
import numpy as np
from garage.torch import global_device, np_to_torch
from garage.utils.running_stats import RunningMeanStd
import torch.nn.functional as F

from garage.torch.modules import GaussianMLPModule, MLPModule, TransformerEncoderLayerNoLN, GaussianMLPIndependentStdModule, GaussianMLPTwoHeadedModule
from garage.torch.policies.stochastic_policy import StochasticPolicy

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GaussianTransformerEncoderPolicy(StochasticPolicy):
    """Transformer whose outputs are fed into a Normal distribution..

    A policy that contains a Transformer to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): Minimum value for std.
        max_std (float): Maximum value for std.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 encoding_hidden_sizes=(64,),
                 encoding_non_linearity=None,
                 mlp_hidden_sizes=(64, 64),
                 mlp_hidden_nonlinearity=torch.tanh,
                 mlp_hidden_w_init=nn.init.xavier_uniform_,
                 mlp_hidden_b_init=nn.init.zeros_,
                 mlp_output_nonlinearity=None,
                 mlp_output_w_init=nn.init.xavier_uniform_,
                 mlp_output_b_init=nn.init.zeros_,
                 learn_std=True,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 annealing_rate=1.0,
                 std_parameterization='exp',
                 layer_normalization=False,
                 d_model=128,
                 dropout=0.0,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=512,
                 activation='relu',
                 obs_horizon=75,
                 policy_head_input="latest_memory",
                 policy_head_type="Default",
                 tfixup=True,
                 remove_ln=True,
                 recurrent_policy=False,
                 normalize_wm=False,
                 name='GaussianTransformerEncoderPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._obs_horizon = obs_horizon
        self._d_model = d_model
        self._policy_head_input = policy_head_input
        self._recurrent_policy = recurrent_policy
        self._normalize_wm = normalize_wm

        self._obs_embedding = nn.Linear(
            in_features = self._obs_dim,
            out_features = d_model,
            bias=False
        )

        self._wm_positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=self._obs_horizon,
        )

        if remove_ln:
            encoder_layers = TransformerEncoderLayerNoLN(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )
        else:
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation
            )

        self._transformer_module = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

        for p in self._transformer_module.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        if tfixup:
            for p in self._obs_embedding.parameters():
                if p.dim() > 1:
                    torch.nn.init.normal_(p, 0, d_model ** (- 1. / 2.))

            temp_state_dic = {}
            for name, param in self._obs_embedding.named_parameters():
                if 'weight' in name:
                    temp_state_dic[name] = ((9* num_encoder_layers) ** (- 1. / 4.)) * param

            for name in self._obs_embedding.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self._obs_embedding.state_dict()[name]
            self._obs_embedding.load_state_dict(temp_state_dic)    

            temp_state_dic = {}
            for name, param in self._transformer_module.named_parameters():
                if any(s in name for s in ["linear1.weight", "linear2.weight", "self_attn.out_proj.weight"]):
                    temp_state_dic[name] = (0.67 * (num_encoder_layers) ** (- 1. / 4.)) * param
                elif "self_attn.in_proj_weight" in name:
                    temp_state_dic[name] = (0.67 * (num_encoder_layers) ** (- 1. / 4.)) * (param * (2**0.5))

            for name in self._transformer_module.state_dict():
                if name not in temp_state_dic:
                    temp_state_dic[name] = self._transformer_module.state_dict()[name]
            self._transformer_module.load_state_dict(temp_state_dic)

        if self._policy_head_input == "latest_memory":
            self._policy_head_input_dim = d_model
        elif self._policy_head_input == "mixed_memory": # working memory + episodic memory
            self._policy_head_input_dim = 2*d_model
        elif self._policy_head_input == "full_memory":
            self._policy_head_input_dim = d_model * self._obs_horizon

        if policy_head_type == "TwoHeaded":
            self._policy_head = GaussianMLPTwoHeadedModule(
                input_dim=self._policy_head_input_dim, 
                output_dim=self._action_dim,
                hidden_sizes=mlp_hidden_sizes,
                hidden_nonlinearity=mlp_hidden_nonlinearity,
                hidden_w_init=mlp_hidden_w_init,
                hidden_b_init=mlp_hidden_b_init,
                output_nonlinearity=mlp_output_nonlinearity,
                output_w_init=mlp_output_w_init,
                output_b_init=mlp_output_b_init,
                learn_std=learn_std,
                init_std=init_std,
                min_std=min_std,
                max_std=max_std,
                std_parameterization=std_parameterization,
                layer_normalization=layer_normalization
            )
        elif policy_head_type == "IndependentStd":
            self._policy_head = GaussianMLPIndependentStdModule(
                input_dim=self._policy_head_input_dim, 
                output_dim=self._action_dim,
                hidden_sizes=mlp_hidden_sizes,
                hidden_nonlinearity=mlp_hidden_nonlinearity,
                hidden_w_init=mlp_hidden_w_init,
                hidden_b_init=mlp_hidden_b_init,
                output_nonlinearity=mlp_output_nonlinearity,
                output_w_init=mlp_output_w_init,
                output_b_init=mlp_output_b_init,
                learn_std=learn_std,
                init_std=init_std,
                min_std=min_std,
                max_std=max_std,
                std_parameterization=std_parameterization,
                layer_normalization=layer_normalization
            )
        elif policy_head_type == "Default":
            self._policy_head = GaussianMLPModule(
                input_dim=self._policy_head_input_dim, 
                output_dim=self._action_dim,
                hidden_sizes=mlp_hidden_sizes,
                hidden_nonlinearity=mlp_hidden_nonlinearity,
                hidden_w_init=mlp_hidden_w_init,
                hidden_b_init=mlp_hidden_b_init,
                output_nonlinearity=mlp_output_nonlinearity,
                output_w_init=mlp_output_w_init,
                output_b_init=mlp_output_b_init,
                learn_std=learn_std,
                init_std=init_std,
                min_std=min_std,
                max_std=max_std,
                annealing_rate=annealing_rate,
                std_parameterization=std_parameterization,
                layer_normalization=layer_normalization)
        
        if self._recurrent_policy:
            self._memory_embedding = nn.Linear(
                in_features = d_model * self._obs_horizon,
                out_features = self._obs_dim,
                bias=False
            )

        self.src_mask = None

        if self._normalize_wm:
            self.wm_rms = RunningMeanStd(shape=(self._obs_dim))

        self._prev_observations = None
        self._last_hidden_state = None
        self._prev_actions = None
        self._episodic_memory_counter = None
        self._new_episode = None
        self._step = None

    def get_mask(self):
        if self.src_mask is not None:
            return self.src_mask
        sz = self._obs_horizon
        ones = torch.ones(sz, sz).to(global_device())
        mask = (torch.triu(ones) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.src_mask = mask
        return mask

    def forward(self, observations, hidden_states=None):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device. Shape (S_len, B, input_step)

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
            torch.Tensor: Hidden States

        """
        policy_head_input, transformer_output = self.compute_memories(observations, hidden_states)
        dist = self._policy_head(policy_head_input)
        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()), transformer_output)

    def compute_memories(self, observations, hidden_states=None):
        # Get original shapes and reshape tensors to have a single batch dimension
        obs_shape = list(observations.shape)
        batch_shape = obs_shape[:-2]
        observations = torch.reshape(observations, (-1, obs_shape[-2], obs_shape[-1]))

        # Computing working memory as a representation from tuple (obs, act, rew)
        working_memo = self._obs_embedding(observations) #(B, S_len, output_step)
        #working_memo = working_memo * math.sqrt(self._d_model)

        # get memory index
        curr_em_index = self._compute_memory_index(observations).unsqueeze(-1).repeat(1, working_memo.shape[-1]).unsqueeze(1)

        # Get current working memory as the most recent in the tensor
        curr_working_memo = torch.gather(working_memo, dim=1, index=curr_em_index)

        working_memo = working_memo.permute(1, 0, 2) #Transformer module inputs (S_len, B, output_step)
        wm_pos = self._wm_positional_encoding(working_memo)
        transformer_output = self._transformer_module(wm_pos, mask=self.get_mask()) #(T, B, target_output)
        transformer_output = transformer_output.permute(1, 0, 2) # going back to batch first

        # Compute policy head input
        if self._policy_head_input == "full_memory":
            return torch.reshape(transformer_output, batch_shape + [self._policy_head_input_dim]), transformer_output.detach().cpu().numpy()

        curr_em = torch.gather(transformer_output, dim=1, index=curr_em_index)
        final_shape_obs = batch_shape + [self._obs_embedding.out_features]
        curr_em = torch.reshape(curr_em, final_shape_obs) #get just the last hidden state as input for policy head
        curr_working_memo = torch.reshape(curr_working_memo, final_shape_obs)
        
        memories = None
        if self._policy_head_input == "latest_memory":
            memories = curr_em
        elif self._policy_head_input == "mixed_memory":
            memories = torch.cat((curr_working_memo, curr_em), axis=-1)

        if self._recurrent_policy:
            transformer_output = self._memory_embedding(torch.reshape(transformer_output, batch_shape + [self._d_model * self._obs_horizon]))
            
        return memories, transformer_output.detach().cpu().numpy()

    def reset(self, do_resets=None):
        """Reset the policy.

        Note:
            If `do_resets` is None, it will be by default `np.array([True])`
            which implies the policy will not be "vectorized", i.e. number of
            parallel environments for training data sampling = 1.

        Args:
            do_resets (numpy.ndarray): Bool that indicates terminal state(s).

        """
        if do_resets is None:
            do_resets = np.array([True])
        if self._prev_actions is None or len(do_resets) != len(
                self._prev_actions):
            self._prev_actions = np.zeros(
                (len(do_resets), self._action_dim))

        self._prev_actions[do_resets] = 0.
        self._episodic_memory_counter = -1
        self._last_hidden_state = None
        self._policy_head.anneal_std()

    def reset_observations(self, do_resets=None):
        if do_resets is None:
            do_resets = np.array([True])
        self._prev_observations = np.zeros((len(do_resets), self._obs_horizon, self._obs_dim))
        self._step = 0
        self._episodic_memory_counter += 1
        self._new_episode = True

    def get_action(self, observation, deterministic=False):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.

        Note:
            It returns an action and a dict, with keys
            - mean (numpy.ndarray): Mean of the distribution.
            - log_std (numpy.ndarray): Log standard deviation of the
                distribution.
            - prev_action (numpy.ndarray): Previous action, only present if
                self._state_include_action is True.

        """
        actions, agent_infos, aug_obs, prev_hiddens = self.get_actions([observation])

        action = actions[0]        
        if deterministic:
            action = agent_infos['mean'][0]

        return action, {k: v[0] for k, v in agent_infos.items()}, aug_obs, prev_hiddens

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Actions
            dict: Predicted action and agent information.

        Note:
            It returns an action and a dict, with keys
            - mean (numpy.ndarray): Means of the distribution.
            - log_std (numpy.ndarray): Log standard deviations of the
                distribution.
            - prev_action (numpy.ndarray): Previous action, only present if
                self._state_include_action is True.

        """

        if self._recurrent_policy and self._last_hidden_state is not None:
            obs_idx = self._step - 1 if self._step - 1 < self._obs_horizon else self._obs_horizon - 1
            self._prev_observations[:, obs_idx, :] = np.copy(self._last_hidden_state)
        observations = self._env_spec.observation_space.flatten_n(observations)
        observations = np.expand_dims(observations, axis=1)
        self._update_prev_observations(observations)
        encoder_input = np_to_torch(self._prev_observations)
        dist, info, hidden_states = self.forward(encoder_input)

        if self._recurrent_policy:
            self._last_hidden_state = hidden_states
        samples = dist.sample().cpu().numpy()
        self._prev_actions = samples

        return samples, {
            k: v.detach().cpu().numpy()
            for (k, v) in info.items()
        }, self._prev_observations, hidden_states

    def apply_rms(self, observations):
        if self._normalize_wm:
            observations = (observations - self.wm_rms.mean) / np.sqrt(self.wm_rms.var + 1e-8)
            self.wm_rms.update(observations)   
        return observations 

    def _update_prev_observations(self, observations):
        if self._step < self._obs_horizon: # fits in memory: just keep updating the right index
            self._prev_observations[:, self._working_memory_index(), :] = observations
        else: # more observations than working memory length
            self._prev_observations = np.concatenate((self._prev_observations[:, 1:, :], observations), axis=1)
        
        self._step += 1

    def _compute_memory_index(self, memory):
        zero_tensor = torch.zeros(memory.shape[-1:]).to(global_device())

        mask = torch.all(torch.eq(memory, zero_tensor), dim=-1)
        mask = mask.float().masked_fill(mask == 1, float(0.0)).masked_fill(mask == 0, float(1.0))

        one_tensor = torch.ones(mask.shape[-1:]).to(global_device())
        full_memory = torch.all(torch.eq(mask, one_tensor), dim=-1)
        x = F.relu(torch.argmin(mask, dim=-1) - 1)
        max_index = memory.shape[-2] - 1
        final_index = torch.where(full_memory, max_index, x)
        return final_index

    def _working_memory_index(self):
        return self._step if self._step < self._obs_horizon else self._obs_horizon - 1

    def compute_current_embeddings(self):
        # Get original shapes and reshape tensors to have a single batch dimension
        observations = np_to_torch(self._prev_observations)
        obs_shape = list(observations.shape)
        batch_shape = obs_shape[:-2]
        observations = torch.reshape(observations, (-1, obs_shape[-2], obs_shape[-1]))

        # Computing working memory as a representation from tuple (obs, act, rew)
        working_memo = self._obs_embedding(observations) #(B, S_len, output_step)
        #working_memo = working_memo * math.sqrt(self._d_model)

        # get memory index
        curr_em_index = self._compute_memory_index(observations).unsqueeze(-1).repeat(1, working_memo.shape[-1]).unsqueeze(1)

        # Get current working memory as the most recent in the tensor
        curr_working_memo = torch.gather(working_memo, dim=1, index=curr_em_index)

        working_memo = working_memo.permute(1, 0, 2) #Transformer module inputs (S_len, B, output_step)
        wm_pos = self._wm_positional_encoding(working_memo)
        transformer_output = self._transformer_module(wm_pos) #(T, B, target_output)
        transformer_output = transformer_output.permute(1, 0, 2) # going back to batch first

        curr_em = torch.gather(transformer_output, dim=1, index=curr_em_index)
        final_shape_obs = batch_shape + [self._obs_embedding.out_features]
        curr_em = torch.reshape(curr_em, final_shape_obs) #get just the last hidden state as input for policy head
        curr_working_memo = torch.reshape(curr_working_memo, final_shape_obs)
        
        return curr_working_memo, curr_em

    def compute_attention_weights(self):
        # Get original shapes and reshape tensors to have a single batch dimension
        observations = np_to_torch(self._prev_observations)
        obs_shape = list(observations.shape)
        batch_shape = obs_shape[:-2]
        observations = torch.reshape(observations, (-1, obs_shape[-2], obs_shape[-1]))

        # Computing working memory as a representation from tuple (obs, act, rew)
        working_memo = self._obs_embedding(observations) #(B, S_len, output_step)
        #working_memo = working_memo * math.sqrt(self._d_model)

        # get memory index
        curr_em_index = self._compute_memory_index(observations).unsqueeze(-1).repeat(1, working_memo.shape[-1]).unsqueeze(1)

        # Get current working memory as the most recent in the tensor
        curr_working_memo = torch.gather(working_memo, dim=1, index=curr_em_index)

        working_memo = working_memo.permute(1, 0, 2) #Transformer module inputs (S_len, B, output_step)
        wm_pos = self._wm_positional_encoding(working_memo)

        output = wm_pos
        attn_list = []
        for mod in self._transformer_module.layers:
            _, attn_weights = mod.self_attn(output, output, output, attn_mask=self.get_mask(), key_padding_mask=None)
            attn_list.append(attn_weights.squeeze())
            output = mod(output, src_mask = self.get_mask(), src_key_padding_mask=None)
        
        return attn_list, curr_em_index.squeeze()[0]
        


    @property
    def memory_dim(self):
        return self._policy_head_input_dim

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        device = device or global_device()
        for net in [self._wm_positional_encoding, self._obs_embedding, self._transformer_module, self._policy_head]:
            net.to(device)