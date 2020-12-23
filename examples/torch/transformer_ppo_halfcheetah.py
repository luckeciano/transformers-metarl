#!/usr/bin/env python3
"""Example script to run RL2 in HalfCheetah."""
# pylint: disable=no-value-for-parameter
import click
import torch
from garage.torch import set_gpu_mode

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.experiment import task_sampler
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.trainer import Trainer
from garage.torch.algos import RL2PPO
from garage.torch.algos.rl2 import RL2Env, RL2Worker
from garage.torch.policies import GaussianTransformerPolicy
from garage.torch.value_functions import GaussianMLPValueFunction

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


@click.command()
@click.option('--seed', default=1)
@click.option('--max_episode_length', default=100)
@click.option('--meta_batch_size', default=10)
@click.option('--n_epochs', default=1000)
@click.option('--episode_per_task', default=4)
@wrap_experiment
def transformer_ppo_halfcheetah(ctxt, seed, max_episode_length, meta_batch_size,
                        n_epochs, episode_per_task):
    """Train PPO with HalfCheetah environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_episode_length (int): Maximum length of a single episode.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    trainer = Trainer(ctxt)
    tasks = task_sampler.SetTaskSampler(
        HalfCheetahVelEnv,
        wrapper=lambda env, _: RL2Env(
            GymEnv(env, max_episode_length=max_episode_length)))

    env_spec = RL2Env(
        GymEnv(HalfCheetahVelEnv(),
                max_episode_length=max_episode_length)).spec
    policy = GaussianTransformerPolicy(name='policy',
                                env_spec=env_spec)
    count_parameters(policy)

    value_function = GaussianMLPValueFunction(env_spec=env_spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    count_parameters(value_function)

    algo = RL2PPO(meta_batch_size=meta_batch_size,
                    task_sampler=tasks,
                    env_spec=env_spec,
                    policy=policy,
                    value_function=value_function,
                    episodes_per_trial=episode_per_task,
                    discount=0.99,
                    gae_lambda=0.95,
                    lr_clip_range=0.2,
                    lr=2.5e-4,
                    minibatch_size=64,
                    max_opt_epochs=10,
                    stop_entropy_gradient=True,
                    entropy_method='max',
                    policy_ent_coeff=0.02,
                    center_adv=False)

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    algo.to()

    trainer.setup(algo,
                    tasks.sample(meta_batch_size),
                    sampler_cls=LocalSampler,
                    n_workers=meta_batch_size,
                    worker_class=RL2Worker,
                    worker_args=dict(n_episodes_per_trial=episode_per_task))

    trainer.train(n_epochs=n_epochs,
                    batch_size=episode_per_task * max_episode_length *
                    meta_batch_size)


transformer_ppo_halfcheetah()
