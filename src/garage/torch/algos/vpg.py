"""Vanilla Policy Gradient (REINFORCE)."""
import collections
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance
from garage.np import discount_cumsum
from garage.np.algos import RLAlgorithm
from garage.sampler import RaySampler
from garage.torch import compute_advantages, filter_valids, np_to_torch
from garage.torch.optimizers import OptimizerWrapper


class VPG(RLAlgorithm):
    """Vanilla Policy Gradient (REINFORCE).

    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """

    def __init__(
        self,
        env_spec,
        policy,
        value_function,
        policy_optimizer=None,
        vf_optimizer=None,
        num_train_per_epoch=1,
        discount=0.99,
        gae_lambda=1,
        center_adv=True,
        positive_adv=False,
        policy_ent_coeff=0.0,
        use_softplus_entropy=False,
        stop_entropy_gradient=False,
        entropy_method='no_entropy',
    ):
        self._discount = discount
        self.policy = policy
        self.max_episode_length = env_spec.max_episode_length

        self._value_function = value_function
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._n_samples = num_train_per_epoch
        self._env_spec = env_spec

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=100)
        self.sampler_cls = RaySampler

        if policy_optimizer:
            self._policy_optimizer = policy_optimizer
        else:
            self._policy_optimizer = OptimizerWrapper(torch.optim.Adam, policy)
        if vf_optimizer:
            self._vf_optimizer = vf_optimizer
        else:
            self._vf_optimizer = OptimizerWrapper(torch.optim.Adam,
                                                  value_function)

        self._old_policy = copy.deepcopy(self.policy)

    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    @property
    def discount(self):
        """Discount factor used by the algorithm.

        Returns:
            float: discount factor.
        """
        return self._discount

    def train_once(self, itr, eps):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            eps (EpisodeBatch): A batch of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        obs = np_to_torch(eps.padded_observations)
        aug_obs = np_to_torch(eps.padded_aug_observations)
        hidden_states = np_to_torch(eps.padded_hidden_states)
        rewards = np_to_torch(eps.padded_rewards)
        returns = np_to_torch(
            np.stack([
                discount_cumsum(reward, self.discount)
                for reward in eps.padded_rewards
            ]))
        valids = eps.lengths

        vf_obs_input = aug_obs if self._value_function.shared_network() else obs
        with torch.no_grad():
            baselines = self._value_function(vf_obs_input, hidden_states)

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(aug_obs, hidden_states)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = np_to_torch(eps.observations)
        aug_obs_flat = np_to_torch(eps.augmented_observations)
        hidden_st_flat = np_to_torch(eps.hidden_states)
        actions_flat = np_to_torch(eps.actions)
        rewards_flat = np_to_torch(eps.rewards)
        returns_flat = torch.cat(filter_valids(returns, valids))
        advs_flat = self._compute_advantage(rewards, valids, baselines)

        vf_obs_input_flat = aug_obs_flat if self._value_function.shared_network() else obs_flat
        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(
                aug_obs_flat, hidden_st_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_before = self._value_function.compute_loss(
                vf_obs_input_flat, hidden_st_flat, returns_flat)
            kl_before = self._compute_kl_constraint(aug_obs, hidden_states)

        self._train(obs_flat, aug_obs_flat, hidden_st_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat)

        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                aug_obs_flat, hidden_st_flat, actions_flat, rewards_flat, advs_flat)
            vf_loss_after = self._value_function.compute_loss(
                vf_obs_input_flat, hidden_st_flat, returns_flat)
            kl_after = self._compute_kl_constraint(aug_obs, hidden_states)
            policy_entropy = self._compute_policy_entropy(aug_obs, hidden_states)

        with tabular.prefix(self.policy.name):
            tabular.record('/LossBefore', policy_loss_before.item())
            tabular.record('/LossAfter', policy_loss_after.item())
            tabular.record('/dLoss',
                           (policy_loss_before - policy_loss_after).item())
            tabular.record('/KLBefore', kl_before.item())
            tabular.record('/KL', kl_after.item())
            tabular.record('/Entropy', policy_entropy.mean().item())
            tabular.record('/lr', self._policy_optimizer._optimizer.param_groups[0]['lr'])

        with tabular.prefix(self._value_function.name):
            tabular.record('/LossBefore', vf_loss_before.item())
            tabular.record('/LossAfter', vf_loss_after.item())
            tabular.record('/dLoss',
                           vf_loss_before.item() - vf_loss_after.item())
            tabular.record('/lr', self._vf_optimizer._optimizer.param_groups[0]['lr'])

        self._old_policy.load_state_dict(self.policy.state_dict())

        undiscounted_returns = log_performance(itr,
                                               eps,
                                               discount=self._discount)
        return np.mean(undiscounted_returns)

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in trainer.step_epochs():
            for _ in range(self._n_samples):
                eps = trainer.obtain_episodes(trainer.step_itr)
                last_return = self.train_once(trainer.step_itr, eps)
                trainer.step_itr += 1

        return last_return

    def _train(self, obs, aug_obs, hidden_states, actions, rewards, returns, advs):
        r"""Train the policy and value function with minibatch.

        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.

        """
        for dataset in self._policy_optimizer.get_minibatch(
                aug_obs, hidden_states, actions, rewards, advs):
            self._train_policy(*dataset)
        
        vf_obs_input = aug_obs if self._value_function.shared_network() else obs
        for dataset in self._vf_optimizer.get_minibatch(vf_obs_input, hidden_states, returns):
            self._train_value_function(*dataset)

    def _train_policy(self, obs, hidden_states, actions, rewards, advantages):
        r"""Train the policy.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).

        """
        self._policy_optimizer.zero_grad()
        loss = self._compute_loss_with_adv(obs, hidden_states, actions, rewards, advantages)
        loss.backward()
        self._policy_optimizer.step()

        return loss

    def _train_value_function(self, obs, hidden_states, returns):
        r"""Train the value function.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            returns (torch.Tensor): Acquired returns
                with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated mean scalar value of value function loss
                (float).

        """
        self._vf_optimizer.zero_grad()
        loss = self._value_function.compute_loss(obs, hidden_states, returns)
        loss.backward()
        self._vf_optimizer.step()

        return loss

    def _compute_loss(self, obs, hidden_states, actions, rewards, valids, baselines):
        r"""Compute mean value of loss.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, P, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        obs_flat = torch.cat(filter_valids(obs, valids))
        hid_st_flat = torch.cat(filter_valids(hidden_states, valids))
        actions_flat = torch.cat(filter_valids(actions, valids))
        rewards_flat = torch.cat(filter_valids(rewards, valids))
        advantages_flat = self._compute_advantage(rewards, valids, baselines)

        return self._compute_loss_with_adv(obs_flat, hid_st_flat, actions_flat,
                                           rewards_flat, advantages_flat)

    def _compute_loss_with_adv(self, obs, hidden_states, actions, rewards, advantages):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of objective.

        """
        objectives = self._compute_objective(advantages, obs, hidden_states, actions, rewards)

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(obs, hidden_states)
            objectives += self._policy_ent_coeff * policy_entropies

        return -objectives.mean()

    def _compute_advantage(self, rewards, valids, baselines):
        r"""Compute mean value of loss.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.

        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.

        """
        advantages = compute_advantages(self._discount, self._gae_lambda,
                                        self.max_episode_length, baselines,
                                        rewards)
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat

    def _compute_kl_constraint(self, obs, hidden_states):
        r"""Compute KL divergence.

        Compute the KL divergence between the old policy distribution and
        current policy distribution.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.

        Returns:
            torch.Tensor: Calculated mean scalar value of KL divergence
                (float).

        """
        with torch.no_grad():
            old_dist = self._old_policy(obs, hidden_states)[0]

        new_dist = self.policy(obs, hidden_states)[0]

        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)

        return kl_constraint.mean()

    def _compute_policy_entropy(self, obs, hidden_states):
        r"""Compute entropy value of probability distribution.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.

        Returns:
            torch.Tensor: Calculated entropy values given observation
                with shape :math:`(N, P)`.

        """
        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self.policy(obs, hidden_states)[0].entropy()
        else:
            policy_entropy = self.policy(obs, hidden_states)[0].entropy()

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_objective(self, advantages, obs, hidden_states, actions, rewards):
        r"""Compute objective value.

        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        del rewards
        log_likelihoods = self.policy(obs, hidden_states)[0].log_prob(actions)

        return log_likelihoods * advantages
