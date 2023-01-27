import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import time


class SNS_Renderer():

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode, sim_height, pix_padding, num_species,
                 grid_size, protection_unit_size, display_population) -> None:

        self.grid_size = grid_size  # Number of cells in a row/column in the grid
        self.protection_unit_size = protection_unit_size  # Number of row/column in the protection units
        self.pix_padding = pix_padding  # Padding between the different simulations
        self.sim_height = sim_height  # Height of simulation grids
        self.window_height = sim_height + self.pix_padding * 2  # Height of PyGame window
        self.window_width = (
            (sim_height + self.pix_padding) *
            num_species) + self.pix_padding  # Length of PyGame window
        self.pix_square_size = (self.sim_height / self.grid_size
                                )  # Size of a single grid square in pixels
        self.num_species = num_species

        self.display_population = display_population

        plt.ion()

        fig, axs = plt.subplots(
            ncols=num_species * 2,
            gridspec_kw=dict(width_ratios=[10, 1, 10, 1, 10, 1]),
            figsize=(20, 5))

        fig.tight_layout()

        self.fig = fig
        self.axs = axs

        plt.show()

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]
        self.render_mode = render_mode

    def render(self, obs, prot_units):
        if self.render_mode == "human":
            return self._render_frame(obs, prot_units)

    def _render_frame(self, obs, prot_units):

        # Unpack observations
        species_pop, species_max = obs

        p0 = sns.heatmap(species_pop[0],
                         ax=self.axs[0],
                         cbar_ax=self.axs[1],
                         cmap="Greens")
        p1 = sns.heatmap(species_pop[1],
                         ax=self.axs[2],
                         cbar_ax=self.axs[3],
                         cmap="Greens")
        p2 = sns.heatmap(species_pop[2],
                         ax=self.axs[4],
                         cbar_ax=self.axs[5],
                         cmap="Greens")
        # Draw protection unit
        for prot_unit_coordinates in prot_units:
            p0.add_patch(
                Rectangle(prot_unit_coordinates,
                          self.protection_unit_size,
                          self.protection_unit_size,
                          fill=False,
                          edgecolor='blue',
                          lw=3))

        self.fig.canvas.draw()
        time.sleep(2)
        self.fig.canvas.flush_events()

    def close(self):
        plt.close()