import metaworld
import numpy as np
import random
import gym
from garage import EnvSpec

class ML1Env(gym.Env):
    """
    MetaWorld's ML1 Environemnts.
    """

    def __init__(self, task_name, max_episode_steps=500):
        self.ml1 = metaworld.ML1(task_name)
        self.env = self.ml1.train_classes[task_name]()
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=self._max_episode_steps)
        super(ML1Env, self).__init__()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def sample_tasks(self, num_tasks):
        return random.sample(self.ml1.train_tasks, num_tasks)

    def set_task(self, task):
        self.env.set_task(task)

    def get_task(self):
        self.env.task
    
    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def spec(self):
        return self._spec


class ML1ReachEnv(ML1Env):
    def __init__(self, max_episode_steps=500):
        super(ML1ReachEnv, self).__init__("reach-v2")

class ML1PushEnv(ML1Env):
    def __init__(self, max_episode_steps=500):
        super(ML1PushEnv, self).__init__("push-v2")

class ML1PickPlaceEnv(ML1Env):
    def __init__(self, max_episode_steps=500):
        super(ML1PickPlaceEnv, self).__init__("pick-place-v2")