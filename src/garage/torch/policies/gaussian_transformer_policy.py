"""GaussianMLPPolicy."""
import torch
from torch import nn
import math
import numpy as np
from garage.torch import global_device, np_to_torch
import torch.nn.functional as F

from garage.torch.modules import GaussianMLPModule, MLPModule
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


class GaussianTransformerPolicy(StochasticPolicy):
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
                 std_parameterization='exp',
                 layer_normalization=False,
                 d_model=128,
                 dropout=0.0,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=512,
                 activation='relu',
                 hidden_horizon=4,
                 obs_horizon=75,
                 name='GaussianTransformerPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_horizon = hidden_horizon
        self._obs_horizon = obs_horizon
        self._d_model = d_model

        self._obs_embedding = MLPModule(
            input_dim = self._obs_dim,
            output_dim = d_model,
            hidden_sizes = encoding_hidden_sizes,
            hidden_nonlinearity=mlp_hidden_nonlinearity,
            hidden_w_init=mlp_hidden_w_init,
            hidden_b_init=mlp_hidden_b_init,
            output_nonlinearity= encoding_non_linearity,
            output_w_init=mlp_output_w_init,
            output_b_init=mlp_output_b_init,
            layer_normalization=layer_normalization
        )

        self._wm_positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=self._obs_horizon,
        )

        self._em_positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=self._hidden_horizon,
        )

        self._transformer_module = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )

        self._policy_head = GaussianMLPModule(
            input_dim=2*d_model, # current working memory + current episodic memory
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
            layer_normalization=layer_normalization)

        self._prev_hiddens = None
        self._prev_observations = None
        self._prev_actions = None
        self._episodic_memory_counter = None
        self._new_episode = None
        self._step = None

    def forward(self, observations, hidden_states):
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

    def compute_memories(self, observations, hidden_states):
        # Get original shapes and reshape tensors to have a single batch dimension
        obs_shape = list(observations.shape)
        hid_st_shape = list(hidden_states.shape)
        batch_shape = hid_st_shape[:-2]
        observations = torch.reshape(observations, (-1, obs_shape[-2], obs_shape[-1]))
        hidden_states = torch.reshape(hidden_states, (-1, hid_st_shape[-2], hid_st_shape[-1])) # reducing batching for single dimension

        # Computing working memory as a representation from tuple (obs, act, rew)
        working_memo = self._obs_embedding(observations) #(B, S_len, output_step)

        # get memory index
        curr_em_index = self._compute_memory_index(hidden_states).unsqueeze(-1).repeat(1, hid_st_shape[-1]).unsqueeze(1)
        curr_wm_index = self._compute_memory_index(observations).unsqueeze(-1).repeat(1, working_memo.shape[-1]).unsqueeze(1)

        # Get current working memory as the most recent in the tensor
        curr_working_memo = torch.gather(working_memo, dim=1, index=curr_wm_index)

        working_memo = working_memo.permute(1, 0, 2) #Transformer module inputs (S_len, B, output_step)
        wm_pos = self._wm_positional_encoding(working_memo)
        hidden_states = hidden_states.permute(1, 0, 2) #Transformer module inputs (S_len, B, output_step)
        em_pos = self._em_positional_encoding(hidden_states)
        transformer_output = self._transformer_module(wm_pos, em_pos) #(T, B, target_output)
        transformer_output = transformer_output.permute(1, 0, 2) # going back to batch first

        # Compute policy head input
        curr_hidden = torch.gather(transformer_output, dim=1, index=curr_em_index)
        final_shape_hidden = batch_shape + hid_st_shape[-1:] #final shape = batch shape + feature dimension
        final_shape_obs = batch_shape + [self._obs_embedding._output_dim]
        curr_hidden = torch.reshape(curr_hidden, final_shape_hidden) #get just the last hidden state as input for policy head
        curr_working_memo = torch.reshape(curr_working_memo, final_shape_obs)
        
        memories = torch.cat((curr_working_memo, curr_hidden), axis=-1)
        return memories, transformer_output

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
            self._prev_hiddens = np.zeros((len(do_resets), self._hidden_horizon, self._d_model))

        self._prev_actions[do_resets] = 0.
        self._prev_hiddens[do_resets] = 0.
        self._episodic_memory_counter = -1

    def reset_observations(self, do_resets=None):
        if do_resets is None:
            do_resets = np.array([True])
        self._prev_observations = np.zeros((len(do_resets), self._obs_horizon, self._obs_dim))
        self._step = 0
        self._episodic_memory_counter += 1
        self._new_episode = True

    def get_action(self, observation):
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
        return actions[0], {k: v[0] for k, v in agent_infos.items()}, aug_obs, prev_hiddens

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
        observations = self._env_spec.observation_space.flatten_n(observations)
        observations = np.expand_dims(observations, axis=1)
        self._update_prev_observations(observations)
        encoder_input = np_to_torch(self._prev_observations)
        decodder_input = np_to_torch(self._prev_hiddens)
        dist, info, hidden_states = self.forward(encoder_input, decodder_input)
        samples = dist.sample().cpu().numpy()
        self._prev_actions = samples
        self._update_episodic_memory(hidden_states)

        return samples, {
            k: v.detach().cpu().numpy()
            for (k, v) in info.items()
        }, self._prev_observations, self._prev_hiddens

    def _update_prev_observations(self, observations):
        if self._step < self._obs_horizon: # fits in memory: just keep updating the right index
            self._prev_observations[:, self._working_memory_index(), :] = observations
        else: # more observations than working memory length
            self._prev_observations = np.concatenate((self._prev_observations[:, 1:, :], observations), axis=1)
        
        self._step += 1


    def _update_episodic_memory(self, hidden_states):
        if self._episodic_memory_counter < self._hidden_horizon: # fits in memory: just keep updating the right index
            self._prev_hiddens[:, self._episodic_memory_counter, :] = hidden_states[:, self._episodic_memory_index(), :].detach().cpu().numpy()
        else: # more episodes than episodic memory length
            if self._new_episode: # remove the last episode and insert a new one
                self._prev_hiddens = self._prev_hiddens = np.concatenate((self._prev_hiddens[:, 1:, :], hidden_states[:, self._episodic_memory_index():, :].detach().cpu().numpy()), axis=1)
                self._new_episode = False
            else: # just update the last
                self._prev_hiddens[:, -1, :] = hidden_states[:, self._episodic_memory_index(), :].detach().cpu().numpy()

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

    # def _create_mask(self, input_tensor):
    #     #Tensor of shape (B, Seq_len, E)
    #     zero_tensor = torch.zeros(input_tensor.shape[-1:]).to(global_device())
    #     mask = torch.all(torch.eq(input_tensor, zero_tensor), dim=-1)
    #     mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
    #     return mask

    def _episodic_memory_index(self):
        return self._episodic_memory_counter if self._episodic_memory_counter < self._hidden_horizon else self._hidden_horizon - 1

    def _working_memory_index(self):
        return self._step if self._step < self._obs_horizon else self._obs_horizon - 1

    def compute_current_embeddings(self):
        # return obs_embeddings, wm_embedding (encoder output), and em_embedding (decoder output)
        # TODO: refactor policy to reuse the code of compute_memories
        # call this method after get action
        observations = np_to_torch(self._prev_observations)
        hidden_states = np_to_torch(self._prev_hiddens)

        # Get original shapes and reshape tensors to have a single batch dimension
        obs_shape = list(observations.shape)
        hid_st_shape = list(hidden_states.shape)
        batch_shape = hid_st_shape[:-2]
        observations = torch.reshape(observations, (-1, obs_shape[-2], obs_shape[-1]))
        hidden_states = torch.reshape(hidden_states, (-1, hid_st_shape[-2], hid_st_shape[-1])) # reducing batching for single dimension

        # Computing working memory as a representation from tuple (obs, act, rew)
        working_memo = self._obs_embedding(observations) #(B, S_len, output_step)

        # get memory index
        curr_em_index = self._compute_memory_index(hidden_states).unsqueeze(-1).repeat(1, hid_st_shape[-1]).unsqueeze(1)
        curr_wm_index = self._compute_memory_index(observations).unsqueeze(-1).repeat(1, working_memo.shape[-1]).unsqueeze(1)

        # Get current working memory as the most recent in the tensor
        curr_working_memo = torch.gather(working_memo, dim=1, index=curr_wm_index)

        working_memo = working_memo.permute(1, 0, 2) #Transformer module inputs (S_len, B, output_step)
        wm_pos = self._wm_positional_encoding(working_memo)
        hidden_states = hidden_states.permute(1, 0, 2) #Transformer module inputs (S_len, B, output_step)
        em_pos = self._em_positional_encoding(hidden_states)
        transformer_output = self._transformer_module(wm_pos, em_pos) #(T, B, target_output)
        transformer_output = transformer_output.permute(1, 0, 2) # going back to batch first

        encoder_output = self._transformer_module.encoder(wm_pos).permute(1, 0, 2)
        current_wm_embedding = torch.gather(encoder_output, dim=1, index=curr_wm_index)

        # Compute policy head input
        curr_hidden = torch.gather(transformer_output, dim=1, index=curr_em_index)
        final_shape_hidden = batch_shape + hid_st_shape[-1:] #final shape = batch shape + feature dimension
        final_shape_obs = batch_shape + [self._obs_embedding._output_dim]
        curr_hidden = torch.reshape(curr_hidden, final_shape_hidden) #get just the last hidden state as input for policy head
        curr_working_memo = torch.reshape(curr_working_memo, final_shape_obs)
        current_wm_embedding = torch.reshape(current_wm_embedding, final_shape_obs)
        
        return curr_working_memo, current_wm_embedding, curr_hidden
        


    @property
    def memory_dim(self):
        return 2*self._d_model