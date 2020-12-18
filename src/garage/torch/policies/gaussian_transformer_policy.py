"""GaussianMLPPolicy."""
import torch
from torch import nn
import math

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
                 encoding_output=120,
                 encoding_hidden_sizes=(64,64),
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
                 d_model=120,
                 dropout=0.0,
                 max_len=5000,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=120,
                 activation='relu',
                 name='GaussianTransformerPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._obs_embedding = MLPModule(
            input_dim = self._obs_dim,
            output_dim = encoding_output,
            hidden_sizes = encoding_hidden_sizes,
            hidden_nonlinearity=mlp_hidden_nonlinearity,
            hidden_w_init=mlp_hidden_w_init,
            hidden_b_init=mlp_hidden_b_init,
            output_nonlinearity= encoding_non_linearity,
            output_w_init=mlp_output_w_init,
            output_b_init=mlp_output_b_init,
            layer_normalization=layer_normalization
        )

        self._positional_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=max_len
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
            input_dim=self._obs_dim,
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
        embedding = self._obs_embedding(observations) #(S_len, B, output_step)
        embedding_pos = self._positional_encoding(embedding)
        transformer_output = self._transformer_module(embedding_pos, hidden_states) #(T, B, target_output)
        dist = self._policy_head(transformer_output[-1:,:, :]) #get just the last hidden state as input for policy head
        return (dist, dict(mean=dist.mean, log_std=(dist.variance**.5).log()), hidden_states)
