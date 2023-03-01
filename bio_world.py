import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bio_environment import BioEnvironment
from sns_renderer import SNS_Renderer


class BioGymWorld(gym.Env):

    def __init__(self, bio_env: BioEnvironment,
                 renderer: SNS_Renderer) -> None:
        super().__init__()
        self.grid_size = bio_env.get_grid_size()
        self.renderer = renderer

        self.bio_env = bio_env

        # Creating the observation space
        self.observation_space = spaces.Box(0,
                                            np.inf,
                                            shape=(self.bio_env.num_species,
                                                   self.grid_size,
                                                   self.grid_size),
                                            dtype=np.float64)

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

        self.renderer.reset()

        observations = self._get_obs()
        info = self._get_info()

        self.render()

        return observations, info

    def step(self, action):
        self.bio_env.step(action)

        terminated = self.bio_env.any_species_extinct()
        reward = self.calculate_reward(terminated)
        observations = self._get_obs()
        info = self._get_info()

        self.render()

        return observations, reward, terminated, info

    def calculate_reward(self, terminated):
        """
        Calculates the rewards, based on the current state of the BioEnvironment
        """
        # If simulation is terminated, at least one species has gone extinct
        if terminated:
            return -1000
        else:
            # Negative reward for each species below critical threshold
            reward = -1 * self.bio_env.get_num_species_critical()
            # Negative cost for placing action unit
            reward += -0.5 if self.bio_env.is_action_unit_placed() else 0
        return reward

    def render(self):
        obs = self._get_obs()
        return self.renderer.render(obs)

    def show_species_history(self):
        """
        Show the species population history
        """
        pop_history = self.bio_env.get_pop_history()
        critical_thresholds = self.bio_env.get_critical_thresholds()
        self.renderer.render_pop_history(pop_history, critical_thresholds)

    def show_score_history(self, score_history):
        """
        Show the score history
        """
        self.renderer.render_score_history(score_history)

    def close(self):
        self.renderer.close()