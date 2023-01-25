import gymnasium
from gymnasium import spaces
import numpy as np
from renderer import Renderer
from bio_environment import BioEnvironment


class BioGymWorld(gymnasium.Env):

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
        self.bio_environment = BioEnvironment(num_species, grid_size)

        species_dict = self.bio_environment.init_species_populations(
            type="Box")

        # Creating the observation space
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
        return self.bio_environment.species_populations

    def _get_info(self):
        return {"info": "Placeholder for information"}

    def reset(self, seed=None, options=None):
        # We need the following line to seedself.np_random
        super().reset(seed=seed)

        # Choose species populations randomly
        self.bio_environment.reset()

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
        self.bio_environment.step()

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