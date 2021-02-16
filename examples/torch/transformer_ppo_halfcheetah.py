#!/usr/bin/env python3
"""Example script to run RL2 in HalfCheetah."""
# pylint: disable=no-value-for-parameter
import click
import torch
from garage.torch import set_gpu_mode

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.experiment import task_sampler, MetaEvaluator, OnlineMetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.trainer import Trainer
from garage.torch.algos import RL2PPO
from garage.torch.algos.rl2 import RL2Env, RL2Worker
from garage.torch.policies import GaussianTransformerPolicy, GaussianTransformerEncoderPolicy, GaussianMemoryTransformerPolicy
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
@click.option('--max_episode_length', default=200)
@click.option('--meta_batch_size', default=10)
@click.option('--n_epochs', default=1000000)
@click.option('--episode_per_task', default=2)
@click.option('--wm_embedding_hidden_size', default=5)
@click.option('--n_heads', default=1)
@click.option('--d_model', default=4)
@click.option('--layers', default=2)
@click.option('--dropout', default=0.0)
@click.option('--wm_size', default=1)
@click.option('--em_size', default=1)
@click.option('--dim_ff', default=16)
@click.option('--discount', default=0.99)
@click.option('--gae_lambda', default=0.95)
@click.option('--lr_clip_range', default=0.2)
@click.option('--policy_lr', default=2.5e-4)
@click.option('--vf_lr', default=2.5e-4)
@click.option('--minibatch_size', default=32)
@click.option('--max_opt_epochs', default=10)
@click.option('--center_adv', is_flag=True)
@click.option('--positive_adv', is_flag=True)
@click.option('--policy_ent_coeff', default=0.02)
@click.option('--use_softplus_entropy', is_flag=True)
@click.option('--stop_entropy_gradient', is_flag=True)
@click.option('--entropy_method', default='max')
@click.option('--share_network', is_flag=True)
@click.option('--architecture', default="MemoryTransformer")
@click.option('--policy_head_input', default="full_memory")
@click.option('--dropatt', default=0.0)
@click.option('--attn_type', default=1)
@click.option('--pre_lnorm', is_flag=True)
@click.option('--init_params', is_flag=True)
@click.option('--gating', default="residual")
@click.option('--gpu_id', default=0)
@wrap_experiment
def transformer_ppo_halfcheetah(ctxt, seed, max_episode_length, meta_batch_size,
                        n_epochs, episode_per_task,
                        wm_embedding_hidden_size, n_heads, d_model, layers, dropout,
                        wm_size, em_size, dim_ff, discount, gae_lambda, lr_clip_range, policy_lr,
                        vf_lr, minibatch_size, max_opt_epochs, center_adv, positive_adv, 
                        policy_ent_coeff, use_softplus_entropy, stop_entropy_gradient, entropy_method,
                        share_network, architecture, policy_head_input, dropatt, attn_type,
                        pre_lnorm, init_params, gating, gpu_id):
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

    if architecture == "Encoder":
        policy = GaussianTransformerEncoderPolicy(name='policy',
                                    env_spec=env_spec,
                                    encoding_hidden_sizes=(wm_embedding_hidden_size,),
                                    nhead=n_heads,
                                    d_model=d_model,
                                    num_encoder_layers=layers,
                                    dropout=dropout,
                                    obs_horizon=wm_size,
                                    dim_feedforward=dim_ff,
                                    policy_head_input=policy_head_input)
    elif architecture == "Transformer":         
        policy = GaussianTransformerPolicy(name='policy',
                                    env_spec=env_spec,
                                    encoding_hidden_sizes=(wm_embedding_hidden_size,),
                                    nhead=n_heads,
                                    d_model=d_model,
                                    num_decoder_layers=layers,
                                    num_encoder_layers=layers,
                                    dropout=dropout,
                                    obs_horizon=wm_size,
                                    hidden_horizon=em_size,
                                    dim_feedforward=dim_ff)
    elif architecture == "MemoryTransformer":
        policy = GaussianMemoryTransformerPolicy(name='policy',
                                    env_spec=env_spec,
                                    encoding_hidden_sizes=(wm_embedding_hidden_size,),
                                    nhead=n_heads,
                                    d_model=d_model,
                                    num_encoder_layers=layers,
                                    dropout=dropout,
                                    dropatt=dropatt,
                                    obs_horizon=wm_size,
                                    mem_len=em_size,
                                    dim_feedforward=dim_ff,
                                    attn_type=attn_type,
                                    pre_lnorm=pre_lnorm,
                                    init_params=init_params,
                                    gating=gating,
                                    policy_head_input=policy_head_input)
                                    

    # count_parameters(policy)

    base_model = policy if share_network else None

    value_function = GaussianMLPValueFunction(env_spec=env_spec,
                                              base_model=base_model,
                                              hidden_sizes=(64, 64),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    # count_parameters(value_function)

    meta_evaluator = OnlineMetaEvaluator(test_task_sampler=tasks,
                                        n_test_tasks=5,
                                        worker_class=RL2Worker,
                                        worker_args=dict(n_episodes_per_trial=3))
    meta_evaluator = None

    algo = RL2PPO(meta_batch_size=meta_batch_size,
                    task_sampler=tasks,
                    env_spec=env_spec,
                    policy=policy,
                    value_function=value_function,
                    episodes_per_trial=episode_per_task,
                    discount=discount,
                    gae_lambda=gae_lambda,
                    lr_clip_range=lr_clip_range,
                    policy_lr=policy_lr,
                    vf_lr=vf_lr,
                    minibatch_size=minibatch_size,
                    max_opt_epochs=max_opt_epochs,
                    use_softplus_entropy=use_softplus_entropy,
                    stop_entropy_gradient=stop_entropy_gradient,
                    entropy_method=entropy_method,
                    policy_ent_coeff=policy_ent_coeff,
                    center_adv=center_adv,
                    positive_adv=positive_adv,
                    meta_evaluator=meta_evaluator,
                    n_epochs_per_eval=15)

    if torch.cuda.is_available():
        set_gpu_mode(True, gpu_id)
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
