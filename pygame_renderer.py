import pygame
import numpy as np


class Pygame_Renderer():

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

        self.display_population = display_population

        self.font_name = "Arial"

        self.window = None
        self.clock = None

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]
        self.render_mode = render_mode

    def render(self, obs):
        if self.render_mode == "human":
            return self._render_frame(obs)

    def _render_fill_square(self, canvas, rgb_color, coordinates, sim_num):
        """
        NB: Coordinates are taken on the format [x, y], where x goes leftwards and y goes downwards
        """
        # Coordinates for the top left corner of the appropriate cell
        x = (self.pix_square_size *
             (coordinates[0] + self.grid_size * sim_num) +
             self.pix_padding * sim_num) + self.pix_padding
        y = self.pix_square_size * coordinates[1] + self.pix_padding
        pygame.draw.rect(
            canvas, rgb_color,
            pygame.Rect(np.array([x, y]),
                        (self.pix_square_size, self.pix_square_size)))

    def _render_draw_protection_unit(self, canvas, speceis, coordinates):
        """
        Drawing the protection unit, given the coordinates of the top-left corner
        NB: Coordinates are taken on the format [x, y], where x goes leftwards and y goes downwards
        """
        # TODO: MUST ADD SUPPORT FOR PROT UNITS FOR DIFFERENT SPECIES
        pix_prot_unit_size = self.pix_square_size * self.protection_unit_size
        pix_x_start = (coordinates[0] *
                       self.pix_square_size) + self.pix_padding
        pix_y_start = (coordinates[1] *
                       self.pix_square_size) + self.pix_padding

        # Drawing the left vertical line
        pygame.draw.line(canvas, (0, 0, 255), (pix_x_start, pix_y_start),
                         (pix_x_start, pix_y_start + pix_prot_unit_size),
                         width=6)

        # Drawing the right vertical line
        pygame.draw.line(canvas, (0, 0, 255),
                         (pix_x_start + pix_prot_unit_size, pix_y_start),
                         (pix_x_start + pix_prot_unit_size,
                          pix_y_start + pix_prot_unit_size),
                         width=6)

        # Drawing the top horizontal line
        pygame.draw.line(canvas, (0, 0, 255), (pix_x_start, pix_y_start),
                         (pix_x_start + pix_prot_unit_size, pix_y_start),
                         width=6)

        # Drawing the bottom horizontal line
        pygame.draw.line(canvas, (0, 0, 255),
                         (pix_x_start, pix_y_start + pix_prot_unit_size),
                         (pix_x_start + pix_prot_unit_size,
                          pix_y_start + pix_prot_unit_size),
                         width=6)

    def _render_draw_gridlines(self, canvas, sim_num):
        for x in range(self.grid_size + 1):

            x_start = self.pix_padding * (sim_num + 1) + (self.sim_height *
                                                          sim_num)
            y_start = self.pix_padding
            x_offset_start = (self.pix_square_size * x) + x_start
            y_offset_start = (self.pix_square_size * x) + y_start
            x_end = (self.sim_height + self.pix_padding) * (sim_num + 1)
            y_end = self.sim_height + self.pix_padding

            # Draw horizontal lines
            pygame.draw.line(canvas,
                             0, (x_start, y_offset_start),
                             (x_end, y_offset_start),
                             width=3)

            # Draw vertical lines
            pygame.draw.line(canvas,
                             0, (x_offset_start, y_start),
                             (x_offset_start, y_end),
                             width=3)

    def _render_add_description(self, canvas, sim_num, max_sim):
        """
        Add information about specific simulation e.g. name
        """
        x = (self.pix_padding *
             (sim_num + 1)) + (self.sim_height // 2) + (self.sim_height *
                                                        sim_num)
        y_title = self.pix_padding // 2
        size = self.pix_padding // 2
        font = pygame.font.SysFont(self.font_name, size)
        title_text = font.render("species_" + str(sim_num), True, 0)
        title_textRect = title_text.get_rect()
        title_textRect.center = (x, y_title)
        canvas.blit(title_text, title_textRect)

        y_desc = int(self.pix_padding * 1.5 + self.sim_height)
        desc_text = font.render("max: " + str(round(max_sim, 2)), True, 0)
        desc_textRect = desc_text.get_rect()
        desc_textRect.center = (x, y_desc)
        canvas.blit(desc_text, desc_textRect)

    def _render_add_population_number(self, canvas, coordinates, sim_num,
                                      population):
        """
        NB: Coordinates are taken on the format [x, y], where x goes leftwards and y goes downwards
        """
        # Coordinates for the top left corner of the appropriate cell
        x = (self.pix_square_size *
             (coordinates[0] + self.grid_size * sim_num) +
             self.pix_padding * sim_num) + self.pix_padding
        y = self.pix_square_size * coordinates[1] + self.pix_padding

        size = int(self.pix_square_size // 4)

        font = pygame.font.SysFont(self.font_name, size)
        pop_text = font.render("" + str(round(population, 3)), True, 0)
        pop_textRect = pop_text.get_rect()
        pop_textRect.center = (x + self.pix_square_size // 2,
                               y + self.pix_square_size // 2)
        canvas.blit(pop_text, pop_textRect)

    def _render_frame(self, obs):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("Spatio-temporal Species Simulator")
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        # First we draw species 0
        green = (0, 100, 0)

        # Unpack observations
        species_pop, prot_units = obs

        for species_num in range(len(species_pop)):
            population = species_pop[species_num]
            population_max = population.max()

            population_range = population_max - population.min()

            for iy, ix in np.ndindex(population.shape):
                population_in_cell = population[iy, ix]
                cell_coordinates = np.array([ix, iy])

                if population_range == 0:  # If population is exactly the same in all cells
                    color_tuple = green
                else:
                    color_tuple = tuple(
                        elem + 20 -
                        int(((population_max - population_in_cell) /
                             population_range) * elem) for elem in green)

                self._render_fill_square(canvas, color_tuple, cell_coordinates,
                                         species_num)

                if self.display_population:
                    self._render_add_population_number(canvas,
                                                       cell_coordinates,
                                                       species_num,
                                                       population_in_cell)

            # Add some gridlines
            self._render_draw_gridlines(canvas, species_num)

            # Add descriptions
            self._render_add_description(canvas, species_num, population_max)

        # Draw protection unit
        for species, coordinates in prot_units:
            self._render_draw_protection_unit(canvas, species, coordinates)

        if self.render_mode == "human":
            # The following line copies our drawings from 'canvas' to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automaticaly add a delay to keep the framerate stable
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),
                                axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()