import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bio_environment import BioEnvironment
from sns_renderer import SNS_Renderer


class BioGymWorld(gym.Env):

    def __init__(self, bio_env: BioEnvironment, renderer: SNS_Renderer,
                 max_steps: int, reduced_actions: bool) -> None:
        super().__init__()
        self.grid_size = bio_env.get_grid_size()
        self.renderer = renderer

        self.bio_env = bio_env

        self.max_steps = max_steps

        self.reduced_actions = reduced_actions

        self.current_step = 0

        # Creating the observation space
        self.observation_space = spaces.Dict({
            "species_populations":
            spaces.Box(0,
                       np.inf,
                       shape=(self.bio_env.num_species, self.grid_size,
                              self.grid_size),
                       dtype=np.float64),
            "criticalness":
            spaces.Box(0,
                       np.inf,
                       shape=(self.bio_env.num_species, ),
                       dtype=np.float64),
            "criticalness_trend":
            spaces.Box(0,
                       np.inf,
                       shape=(self.bio_env.num_species, ),
                       dtype=np.float64)
        })

        # Creating the action space
        self.action_space = spaces.Discrete(self.bio_env.get_action_space())

    def _get_obs(self):
        return self.bio_env.get_obs()

    def _get_info(self):
        return {"info": "Placeholder for information"}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose species populations randomly
        self.bio_env.reset()

        self.renderer.reset()

        self.current_step = 0

        observations = self._get_obs()
        info = self._get_info()

        self.render()

        return observations, info

    def step(self, action):
        self.current_step += 1

        self.bio_env.step(action)

        terminated = self.bio_env.any_species_extinct()
        reward = self.calculate_reward(terminated)
        observations = self._get_obs()
        info = self._get_info()

        self.render()

        if self.current_step == self.max_steps:
            return observations, reward, terminated, True, info

        return observations, reward, terminated, False, info

    def calculate_reward(self, terminated):
        """
        Calculates the rewards, based on the current state of the BioEnvironment
        """
        # If simulation is terminated, at least one species has gone extinct
        if terminated:
            return -1000
        else:
            reward = 0
            # Negative reward for each species below critical threshold
            if self.bio_env.is_action_unit_placed():
                _, _, harvesting, _ = self.bio_env.get_action_unit()
                if harvesting:
                    reward += -0.25
                else:
                    reward += -0.25
            if self.bio_env.get_num_species_critical() > 0:
                reward += -1
            else:
                reward += 1
            # Negative cost for placing action unit
        return reward

    def render(self):
        obs = self.bio_env.get_render_obs()
        return self.renderer.render(obs)

    def show_episode_history(self):
        """
        Show the history for this episode
        """
        history = self.bio_env.get_history()
        critical_thresholds = self.bio_env.get_critical_thresholds()
        self.renderer.render_episode_history(history, critical_thresholds)

    def show_run_history(self, score_history, species_richness_history,
                         species_evenness_history):
        """
        Show the history for this run (over all episodes)
        """
        self.renderer.render_run_history(score_history,
                                         species_richness_history,
                                         species_evenness_history)

    def close(self):
        self.renderer.close()