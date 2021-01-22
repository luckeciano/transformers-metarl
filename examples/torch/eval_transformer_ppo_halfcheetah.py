#!/usr/bin/env python3
"""Example script to run RL2 in HalfCheetah."""
# pylint: disable=no-value-for-parameter
import click
import torch
import numpy as np
import time

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.experiment import task_sampler, OnlineMetaEvaluator, Snapshotter
from garage.experiment.deterministic import set_seed
from garage.torch.algos.rl2 import RL2Env, RL2Worker
from garage.np import stack_tensor_dict_list
from garage.torch import set_gpu_mode

from prettytable import PrettyTable

def rollout(env,
            agent,
            *,
            max_episode_length=np.inf,
            animated=False,
            pause_per_frame=None,
            deterministic=False):
    """Sample a single episode of the agent in the environment.

    Args:
        agent (Policy): Policy used to select actions.
        env (Environment): Environment to perform actions in.
        max_episode_length (int): If the episode reaches this many timesteps,
            it is truncated.
        animated (bool): If true, render the environment after each step.
        pause_per_frame (float): Time to sleep between steps. Only relevant if
            animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.

    Returns:
        dict[str, np.ndarray or dict]: Dictionary, with keys:
            * observations(np.array): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape
                :math:`(T + 1, S^*)`, i.e. the unflattened observation space of
                    the current environment.
            * actions(np.array): Non-flattened array of actions. Should have
                shape :math:`(T, S^*)`, i.e. the unflattened action space of
                the current environment.
            * rewards(np.array): Array of rewards of shape :math:`(T,)`, i.e. a
                1D array of length timesteps.
            * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(np.array): Array of termination signals.

    """
    env_steps = []
    agent_infos = []
    observations = []
    last_obs, episode_infos = env.reset()
    agent.reset()
    episode_length = 0
    if animated:
        env.visualize()
    while episode_length < (max_episode_length or np.inf):
        if pause_per_frame is not None:
            time.sleep(pause_per_frame)
        a, agent_info, _, _ = agent.get_action(last_obs)
        obs_emb, wm_emb, em_emb = agent.compute_current_embeddings()
        agent_info["obs_emb"] = obs_emb
        agent_info["wm_emb"] = wm_emb
        agent_info["em_emb"] = em_emb
        if deterministic and 'mean' in agent_info:
            a = agent_info['mean']
        es = env.step(a)
        env_steps.append(es)
        observations.append(last_obs)
        agent_infos.append(agent_info)
        episode_length += 1
        if es.last:
            break
        last_obs = es.observation

    return dict(
        episode_infos=episode_infos,
        observations=np.array(observations),
        actions=np.array([es.action for es in env_steps]),
        rewards=np.array([es.reward for es in env_steps]),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list([es.env_info for es in env_steps]),
        dones=np.array([es.terminal for es in env_steps]),
    )

@click.command()
@click.option('--path', default='/data/transformer-metarl/garage/examples/torch/data/local/experiment/transformer_ppo_halfcheetah_11')
def transformer_ppo_halfcheetah(path):
    """Eval policy with HalfCheetah environment.
    """
    snapshotter = Snapshotter()
    data = snapshotter.load(path)
    # tasks = task_sampler.SetTaskSampler(
    #     HalfCheetahVelEnv,
    #     wrapper=lambda env, _: RL2Env(
    #         GymEnv(env, max_episode_length=200)))

    policy = data['algo'].policy


    if torch.cuda.is_available():
        set_gpu_mode(True, gpu_id=0)
    else:
        set_gpu_mode(False)
    # algo.to()
    obs_emb_file = open("embeddings/obs_embeddings.tsv", "ab")
    wm_emb_file = open("embeddings/wm_embeddings.tsv", "ab")
    em_emb_file = open("embeddings/em_embeddings.tsv", "ab")
    metadata_file = open("embeddings/metadata.tsv", "ab")
    for velocity in np.arange(0.0, 2.01, 0.5):
        task = {'velocity': velocity}
        env = RL2Env(GymEnv(HalfCheetahVelEnv(task), max_episode_length=200))
        eps = rollout(env, policy, animated=True, deterministic=True)
        t = 0
        for obs_emb, wm_emb, em_emb in zip(eps["agent_infos"]["obs_emb"], eps["agent_infos"]["wm_emb"], eps["agent_infos"]["em_emb"]):
            np.savetxt(obs_emb_file, obs_emb.detach().cpu().numpy(), delimiter='\t')
            np.savetxt(wm_emb_file, wm_emb.detach().cpu().numpy(), delimiter='\t')
            # np.savetxt(em_emb_file, np.array([0.0, em_emb.detach().cpu().numpy()])[np.newaxis], delimiter='\t')
            np.savetxt(metadata_file, np.array([t, velocity])[np.newaxis], delimiter='\t')
            t = t + 1
        print(sum(eps['rewards']))
    obs_emb_file.close()
    wm_emb_file.close()
    em_emb_file.close()
    metadata_file.close()
        #env.visualize()

transformer_ppo_halfcheetah()