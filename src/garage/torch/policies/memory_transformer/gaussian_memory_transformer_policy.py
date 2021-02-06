"""GaussianMLPPolicy."""
import torch
from torch import nn
import math
import numpy as np
from garage.torch import global_device, np_to_torch
import torch.nn.functional as F

from garage.torch.modules import GaussianMLPModule, MLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy
from garage.torch.policies.memory_transformer.memory_transformer import MemoryTransformer


class GaussianMemoryTransformerPolicy(StochasticPolicy):
    """Memory Transformer whose outputs are fed into a Normal distribution.

    A policy that contains a Transformer to make prediction based on a gaussian
    distribution.
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
                 dropatt=0.0,
                 nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=512,
                 obs_horizon=75,
                 policy_head_input="latest_memory",
                 pre_lnorm=True,
                 tgt_len=0,
                 ext_len=0,
                 mem_len=0,
                 attn_type=0, #default attention
                 init_params=True,
                 name='GaussianTransformerEncoderPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._obs_horizon = obs_horizon
        self._d_model = d_model
        self._policy_head_input = policy_head_input

        head_dim = d_model // nhead
        assert head_dim * nhead == d_model, "embed_dim must be divisible by num_heads"

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

        self._transformer_module = MemoryTransformer(
            n_layer=num_encoder_layers, n_head=nhead, d_model=d_model, d_head=head_dim,
            dim_ff=dim_feedforward, dropout=dropout, dropatt=dropatt, obs_embedding_fn=self._obs_embedding, pre_lnorm=pre_lnorm,
            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len, attn_type=attn_type)

        if init_params:
            for p in self._transformer_module.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

        if self._policy_head_input == "latest_memory":
            self._policy_head_input_dim = d_model * self._obs_horizon
        elif self._policy_head_input == "full_memory":
            self._policy_head_input_dim = (num_encoder_layers + 1) * d_model * mem_len

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
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization)

        self._prev_observations = None
        self._prev_actions = None
        self._episodic_memory_counter = None
        self._new_episode = None
        self._step = None

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
        policy_head_input, new_hiddens = self.compute_memories(observations, hidden_states)
        dist = self._policy_head(policy_head_input) 
        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()), new_hiddens)

    def compute_memories(self, observations, hidden_states):
        # Get original shapes and reshape tensors to have a single batch dimension
        obs_shape = list(observations.shape)
        batch_shape = obs_shape[:-2]
        observations = torch.reshape(observations, (-1, obs_shape[-2], obs_shape[-1]))
        

        # get memory index
        # curr_em_index = self._compute_memory_index(observations).unsqueeze(-1).repeat(1, observations.shape[-1]).unsqueeze(1)

        # Get current working memory as the most recent in the tensor
        # curr_working_memo = torch.gather(observations, dim=1, index=curr_em_index)

        working_memo = observations.permute(1, 0, 2) #Transformer module inputs (S_len, B, output_step)
        input_hiddens = [] if hidden_states is not None else None
        if hidden_states is not None:
            hid_shape = list(hidden_states.shape)
            hidden_states = torch.reshape(hidden_states, (-1, hid_shape[-3], hid_shape[-2], hid_shape[-1])).permute(1, 2, 0, 3)
            for i, x in enumerate(hidden_states):
                input_hiddens.append(x)
            # for i in range(len(hidden_states)):
            #     hid_shape = list(hidden_states[i].shape)
                #hidden_states[i] = torch.reshape(hidden_states[i], (-1, hid_shape[-3], hid_shape[-2], hid_shape[-1])).permute(1, 0, 2)
        #hidden_states = hidden_states.permute(1, 0, 2) if hidden_states is not None else None #Transformer module inputs (S_len, B, output_step)
        transformer_output, new_hiddens = self._transformer_module(working_memo, input_hiddens) #(T, B, target_output)
        transformer_output = transformer_output.permute(1, 0, 2) # going back to batch first
        out_hiddens = []
        for i in range(len(new_hiddens)):
            out_hiddens.append(new_hiddens[i].permute(1, 0, 2).detach().cpu().numpy())
        out_hiddens = np.array(out_hiddens).transpose(1, 0, 2, 3)

        if self._policy_head_input == "latest_memory":
            return torch.reshape(transformer_output, batch_shape + [self._policy_head_input_dim]), out_hiddens
        elif self._policy_head_input == "full_memory":
            return torch.reshape(torch.cat(new_hiddens), batch_shape + [self._policy_head_input_dim]), out_hiddens

        # # Compute policy head input
        # curr_em = torch.gather(transformer_output, dim=1, index=curr_em_index)
        # final_shape_obs = batch_shape + [self._obs_embedding._output_dim]
        # curr_em = torch.reshape(curr_em, final_shape_obs) #get just the last hidden state as input for policy head
        # curr_working_memo = torch.reshape(curr_working_memo, final_shape_obs)
        
        # memories = curr_em if self._policy_head_em_only else torch.cat((curr_working_memo, curr_em), axis=-1)
        # return memories, new_hiddens

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
            self._prev_hiddens = None

        self._prev_actions[do_resets] = 0.
        # self._prev_hiddens[do_resets] = 0.
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
        decoder_input = None
        if self._prev_hiddens is not None:
            decoder_input = np_to_torch(self._prev_hiddens)
            # prev_hiddens = self._prev_hiddens.transpose(1, 0, 2, 3)
            # decoder_input = []
            # for i in range(len(prev_hiddens)):
            #     decoder_input.append(np_to_torch(prev_hiddens[i]))
        dist, info, hidden_states = self.forward(encoder_input, decoder_input)
        samples = dist.sample().cpu().numpy()
        self._prev_actions = samples
        self._prev_hiddens = hidden_states

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
        raise NotImplementedError

    def compute_attention_weights(self):
        raise NotImplementedError
        
    @property
    def memory_dim(self):
        return self._policy_head_input_dim