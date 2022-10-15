import gym
from gym import spaces
import numpy as np
from renderer import Renderer


class BioGymWorld(gym.Env):

    def __init__(self,
                 render_mode=None,
                 sim_height=500,
                 render_pix_padding=50,
                 grid_size=5,
                 protection_unit_size=3,
                 num_species=3) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.protection_unit_size = protection_unit_size
        self.renderer = Renderer(render_mode, sim_height, render_pix_padding,
                                 num_species, grid_size, protection_unit_size)
        self.num_species = num_species
        self.species_populations = {}

        species_dict = {}

        # Creating the observation space
        for i in range(num_species):

            species_dict["species_" + str(i)] = spaces.Box(0,
                                                           np.inf,
                                                           shape=(grid_size,
                                                                  grid_size),
                                                           dtype=int)

        self.observation_space = spaces.Dict(species_dict)

        self.action_space = spaces.Discrete(
            (grid_size - protection_unit_size + 1)**2)

        self.protection_units = []

    def _action_to_coordinate(self, action):
        """
        Maps from Discrete action, which is int, to coordinates on grid of the upper left cell of the protection unit
        """
        x_coor = action % (self.grid_size - self.protection_unit_size + 1)
        y_coor = action // (self.grid_size - self.protection_unit_size + 1)
        return np.array([x_coor, y_coor])

    def _get_obs(self):
        return self.species_populations

    def _get_info(self):
        return {"info": "Placeholder for information"}

    def reset(self, seed=None, options=None):
        # We need the following line to seedself.np_random
        super().reset(seed=seed)

        # Choose species populations randomly
        for i in range(self.num_species):
            self.species_populations["species_" +
                                     str(i)] = self.np_random.integers(
                                         0,
                                         100,
                                         size=(self.grid_size, self.grid_size),
                                         dtype=int)

        # Reset protection units
        self.protection_units = []

        observations = self._get_obs()
        info = self._get_info()

        self.render()

        return observations, info

    def step(self, action):
        # Map the action
        protection_unit_coordinates = self._action_to_coordinate(action)
        self.protection_units.append(protection_unit_coordinates)

        terminated = False
        reward = 1 if terminated else 0
        observations = self._get_obs()
        info = self._get_info()

        self.render()

        return observations, reward, terminated, info

    def render(self):
        obs = self._get_obs()
        prot_units = self.protection_units
        return self.renderer.render(obs, prot_units)

    def close(self):
        self.renderer.close()