import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class SNS_Renderer():

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode, sim_height, pix_padding, num_species,
                 grid_size, protection_unit_size, display_population) -> None:

        self.grid_size = grid_size  # Number of cells in a row/column in the grid
        self.protection_unit_size = protection_unit_size  # Number of row/column in the protection units
        self.pix_padding = pix_padding  # Padding between the different simulations
        self.sim_height = sim_height  # Height of simulation grids
        self.window_height = sim_height // 50  # Height of plt window
        self.window_width = self.window_height * (num_species + 1
                                                  )  # Length of plt window
        self.num_species = num_species

        self.display_population = display_population

        self.species_names = ["Prey", "Mesopredator", "Apex predator"]

        plt.ion()

        fig, axs = plt.subplots(
            ncols=(num_species * 3) - 1,
            gridspec_kw=dict(width_ratios=[10, 1, 0.75, 10, 1, 0.75, 10, 1]),
            figsize=(self.window_width, self.window_height))

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

    def _render_add_description(self, species_pop):
        for i in range(self.num_species):
            pop_max = species_pop[i].max()
            self.axs[i * 3].set_title(self.species_names[i] + " (max: " +
                                      str(round(pop_max, 2)) + ")",
                                      fontdict={
                                          'fontsize': 15,
                                          'fontweight': 'medium'
                                      })

    def render(self, obs):
        if self.render_mode == "human":
            return self._render_frame(obs)

    def _render_frame(self, obs):

        # Unpack observations
        species_pop, prot_units = obs

        # Create heatmaps
        for i in range(self.num_species):
            # Clearing previous heatmaps
            self.axs[i * 3].cla()

            # Create heatmap with data from obs
            self.heatmaps[i] = sns.heatmap(species_pop[i],
                                           annot=self.display_population,
                                           vmin=0,
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

        self._render_add_description(species_pop)

        # Draw protection unit
        for species, coordinates in prot_units:
            self.heatmaps[species].add_patch(
                Rectangle(coordinates,
                          self.protection_unit_size,
                          self.protection_unit_size,
                          fill=False,
                          edgecolor='blue',
                          lw=3))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        plt.close()