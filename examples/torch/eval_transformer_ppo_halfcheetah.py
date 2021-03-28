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
from garage.visualization.attention_heatmap import plot_attention

from prettytable import PrettyTable

def create_attention_dict(attn_weights, index_memory, timestep):
    wm_horizon = attn_weights[0].shape[0]
    att_dict = {}
    timestep_label = list(range(wm_horizon))
    timestep_label = [str(i) for i in timestep_label]
    timestep_label[index_memory] = timestep_label[index_memory] + ' (Index)'
    for i in range(len(attn_weights)):
        att_dict[i] = {}
        att_dict[i]['weights'] = attn_weights[i].detach().cpu().numpy()
        att_dict[i]['title'] = "Timestep " + str(timestep)
        att_dict[i]['x_label'] = timestep_label
        att_dict[i]['y_label'] = timestep_label
    return att_dict

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
    episode_length = 0
    if animated:
        env.visualize()
    while episode_length < (max_episode_length or np.inf):
        if pause_per_frame is not None:
            time.sleep(pause_per_frame)
        a, agent_info, _, _ = agent.get_action(last_obs)
        obs_emb, em_emb = agent.compute_current_embeddings()
        #attn_weights, index_memory = agent.compute_attention_weights()
        #agent_info["attention"] = create_attention_dict(attn_weights, index_memory, episode_length)
        agent_info["obs_emb"] = obs_emb
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

def get_env(env_name):
    m = __import__("garage")
    m = getattr(m, "envs")
    m = getattr(m, "mujoco")
    return getattr(m, env_name)

@click.command()
@click.option('--path', default='/data/transformer-metarl/garage/examples/torch/data/local/experiment/transformer_ppo_halfcheetah_241')
@click.option('--env_name', default="HalfCheetahVelEnv")
def transformer_ppo_halfcheetah(path, env_name):
    """Eval policy with HalfCheetah environment.
    """
    snapshotter = Snapshotter()
    data = snapshotter.load(path, itr=1200)
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

    env_class = get_env(env_name)
    env = RL2Env(GymEnv(env_class(), max_episode_length=200))
    tasks = env.sample_tasks(num_tasks=1)
    episodes_per_trial=2

    obs_emb_file = open("embeddings/obs_embeddings_{0}.tsv".format(env_name), "ab")
    em_emb_file = open("embeddings/em_embeddings_{0}.tsv".format(env_name), "ab")
    metadata_file = open("embeddings/metadata_{0}.tsv".format(env_name), "ab")
    for task in tasks:
        task_id = list(task.values())
        env.set_task(task)
        policy.reset()
        t = 0
        for ep in range(episodes_per_trial):
            policy.reset_observations()
            eps = rollout(env, policy, animated=False, deterministic=True)
        
        #attn_dict = eps["agent_infos"]["attention"][0]
        #for i in range(attn_dict['weights'].shape[0]):
        #    attn_weights = attn_dict['weights'][i]
        #    title = attn_dict['title'][i]
        #    title = title + " Velocity " + str(velocity) + " Layer " + str(i)
        #    x_lb = attn_dict['x_label'][i]
        #    y_lb = attn_dict['y_label'][i]
        #    plot_attention(attn_weights, x_lb, y_lb, title, "./visualization")
            for obs_emb, em_emb in zip(eps["agent_infos"]["obs_emb"], eps["agent_infos"]["em_emb"]):
                np.savetxt(obs_emb_file, obs_emb.detach().cpu().numpy(), delimiter='\t')
                # np.savetxt(wm_emb_file, wm_emb.detach().cpu().numpy(), delimiter='\t')
                np.savetxt(em_emb_file, em_emb.detach().cpu().numpy(), delimiter='\t')
                np.savetxt(metadata_file, np.array([t] + task_id)[np.newaxis], delimiter='\t')
                t = t + 1
            print(sum(eps['rewards']))
    obs_emb_file.close()
    em_emb_file.close()
    metadata_file.close()

transformer_ppo_halfcheetah()