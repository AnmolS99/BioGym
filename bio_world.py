import gymnasium as gym
from gymnasium import spaces
import numpy as np

from bio_environment import BioEnvironment


class BioGymWorld(gym.Env):

    def __init__(self, bio_env: BioEnvironment, renderer) -> None:
        super().__init__()
        self.grid_size = bio_env.get_grid_size()
        self.prot_unit_size = bio_env.get_prot_unit_size()
        self.renderer = renderer

        self.bio_env = bio_env

        species_dict = self.bio_env.init_species_populations(type="Box")

        # Creating the observation space
        self.observation_space = spaces.Dict(species_dict)

        # Creating the action space
        self.action_space = spaces.Discrete(self.bio_env.get_action_space())

    def _get_obs(self):
        return self.bio_env.get_obs()

    def _get_info(self):
        return {"info": "Placeholder for information"}

    def reset(self, seed=None, options=None):
        # We need the following line to seedself.np_random
        super().reset(seed=seed)

        # Choose species populations randomly
        self.bio_env.reset()

        # Reset protection units
        self.prot_units = []

        observations = self._get_obs()
        info = self._get_info()

        self.render()

        return observations, info

    def step(self, action):
        self.bio_env.step(action)

        terminated = False
        reward = 1 if terminated else 0
        observations = self._get_obs()
        info = self._get_info()

        self.render()

        return observations, reward, terminated, info

    def render(self):
        obs = self._get_obs()
        return self.renderer.render(obs)

    def close(self):
        self.renderer.close()