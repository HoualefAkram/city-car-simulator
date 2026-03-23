import gymnasium as gym
from gymnasium.spaces import Discrete


class HandoverEnv(gym.Env):

    def __init__(self):
        super().__init__()
        self.observation_space = Discrete(12)
        self.action_space = Discrete(4)

    def step(self, action):
        return super().step(action)

    def reset(self, seed=None, options=None):
        return super().reset(seed=seed, options=options)
