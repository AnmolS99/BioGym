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
            ncols=(num_species * 3) - 1,
            gridspec_kw=dict(width_ratios=[10, 1, 0.75, 10, 1, 0.75, 10, 1]),
            figsize=(20, 5))

        # Remove axes only used to add spacing between colourbar and next heatmap
        axs[2].remove()
        axs[5].remove()

        plt.show()

        self.fig = fig
        self.axs = axs

        self.heatmaps = [None] * num_species

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]
        self.render_mode = render_mode

    def _render_add_description(self):
        for i in range(self.num_species):
            self.axs[i * 3].set_title("species_" + str(i),
                                      fontdict={
                                          'fontsize': 15,
                                          'fontweight': 'medium'
                                      })

    def render(self, obs, prot_units):
        if self.render_mode == "human":
            return self._render_frame(obs, prot_units)

    def _render_frame(self, obs, prot_units):

        # Unpack observations
        species_pop, species_max = obs

        # Create heatmaps
        for i in range(self.num_species):
            # Clearing previous heatmaps
            self.axs[i * 3].cla()

            # Create heatmap with data from obs
            self.heatmaps[i] = sns.heatmap(species_pop[i],
                                           annot=self.display_population,
                                           ax=self.axs[i * 3],
                                           linewidths=0.5,
                                           xticklabels=False,
                                           yticklabels=False,
                                           cbar_ax=self.axs[(i * 3) + 1],
                                           cmap="Greens")

            self.heatmaps[i].axhline(y=0, color='k', linewidth=5)
            self.heatmaps[i].axhline(y=self.grid_size, color='k', linewidth=5)
            self.heatmaps[i].axvline(x=0, color='k', linewidth=5)
            self.heatmaps[i].axvline(x=self.grid_size, color='k', linewidth=5)

        self._render_add_description()

        # Draw protection unit
        for prot_unit_coordinates in prot_units:
            self.heatmaps[0].add_patch(
                Rectangle(prot_unit_coordinates,
                          self.protection_unit_size,
                          self.protection_unit_size,
                          fill=False,
                          edgecolor='blue',
                          lw=3))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(2)

    def close(self):
        plt.close()