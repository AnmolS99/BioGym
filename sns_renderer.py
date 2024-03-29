import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class SNS_Renderer():

    def __init__(self, render_mode, sim_height, pix_padding, num_species,
                 grid_size, action_unit_size, display_population) -> None:

        self.grid_size = grid_size  # Number of cells in a row/column in the grid
        self.action_unit_size = action_unit_size  # Number of rows/columns in the action unit
        self.pix_padding = pix_padding  # Padding between the different simulations
        self.sim_height = sim_height  # Height of simulation grids
        self.window_height = sim_height // 40  # Height of plt window
        self.window_width = self.window_height * (num_species + 1
                                                  )  # Length of plt window
        self.num_species = num_species

        self.display_population = display_population

        self.species_names = ["Prey", "Mesopredator", "Apex predator"]

        assert render_mode in ["on", "off"]
        self.render_mode = render_mode

        if self.render_mode == "on":

            self.reset()

            self.heatmaps = [None] * num_species

    def reset(self):

        if self.render_mode == "on":

            plt.close()

            plt.ion()

            fig, axs = plt.subplots(
                ncols=(self.num_species * 3) - 1,
                gridspec_kw=dict(
                    width_ratios=[10, 1, 0.75, 10, 1, 0.75, 10, 1]),
                figsize=(self.window_width, self.window_height))

            # Remove axes only used to add spacing between colourbar and next heatmap
            axs[2].remove()
            axs[5].remove()

            self.fig = fig
            self.axs = axs

            plt.show()

    def _render_add_description(self, species_pop):
        for i in range(self.num_species):
            pop_max = species_pop[i].max()
            pop_sum = species_pop[i].sum()
            self.axs[i * 3].set_title(self.species_names[i] + "\nmax: " +
                                      str(round(pop_max, 2)) +
                                      "  --  total: " + str(round(pop_sum, 2)),
                                      fontdict={
                                          'fontsize': 15,
                                          'fontweight': 'medium'
                                      })

    def render(self, obs):
        if self.render_mode == "on":
            return self._render_frame(obs)

    def _render_frame(self, obs):

        # Unpack observations
        species_pop, action_unit, critical_species = obs

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

            if critical_species[i]:
                color = "r"
            else:
                color = "k"

            self.heatmaps[i].axhline(y=0, color=color, linewidth=5)
            self.heatmaps[i].axhline(y=self.grid_size,
                                     color=color,
                                     linewidth=5)
            self.heatmaps[i].axvline(x=0, color=color, linewidth=5)
            self.heatmaps[i].axvline(x=self.grid_size,
                                     color=color,
                                     linewidth=5)

        self._render_add_description(species_pop)

        # Draw action unit
        if action_unit is not None:
            species, coordinates, harvesting, population = action_unit
            if harvesting:
                edgecolor = "red"
            else:
                edgecolor = "blue"

            self.heatmaps[species].add_patch(
                Rectangle(coordinates,
                          self.action_unit_size,
                          self.action_unit_size,
                          fill=False,
                          edgecolor=edgecolor,
                          lw=3,
                          zorder=2))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render_episode_history(self, history, critical_thresholds):
        """
        Render history of different metrics
        """

        # Close window displaying heatmaps
        plt.close()

        fig, ax = plt.subplots(self.num_species + len(history) - 1, 1)

        for species_num in range(self.num_species):
            ax[species_num].plot(history["pop_history"][species_num])
            ax[species_num].set_title(str(self.species_names[species_num]))
            ax[species_num].axhline(critical_thresholds[species_num],
                                    linestyle='--',
                                    color="red")

        ax[3].plot(history["species_abundance"])
        ax[3].set_title("Species abundance (relative to critical thresholds)")
        ax[3].axhline(1, linestyle='--', color="red")

        ax[4].plot(history["shannon_index"])
        ax[4].set_title("Shannon index")

        fig.supylabel('Population')
        fig.supxlabel('Time step')

        fig.tight_layout()

        plt.show(block=True)
        self.reset()

    def render_run_history(self, score_history, species_abundance_history,
                           shannon_index_history):
        """
        Render species population history
        """
        # Close window displaying heatmaps
        plt.close()

        fig, ax = plt.subplots(3, 1)

        ax[0].plot(score_history)
        ax[0].set_title("Score (Total reward)")

        ax[1].plot(species_abundance_history)
        ax[1].set_title("Species abundance (relative to critical thresholds)")

        ax[2].plot(shannon_index_history)
        ax[2].set_title("Shannon Index")

        fig.supxlabel('Episode')

        fig.tight_layout()

        plt.show(block=True)
        self.reset()

    def close(self):
        plt.close()