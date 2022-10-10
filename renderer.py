import pygame
import numpy as np


class Renderer():

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self, render_mode, window_size, grid_size,
                 protection_unit_size) -> None:

        self.grid_size = grid_size
        self.protection_unit_size = protection_unit_size
        self.window_size = 768  # Size of PyGame window
        self.pix_square_size = (self.window_size / self.grid_size
                                )  # Size of a single grid square in pixels

        self.window = None
        self.clock = None

        assert render_mode is None or render_mode in self.metadata[
            "render_modes"]
        self.render_mode = render_mode

    def render(self, obs, prot_units):
        if self.render_mode == "human":
            return self._render_frame(obs, prot_units)

    def _render_fill_square(self, canvas, rgb_color, coordinates):
        """
        NB: Coordinates are taken on the format [x, y], where x goes leftwards and y goes downwards
        """
        pygame.draw.rect(
            canvas, rgb_color,
            pygame.Rect(self.pix_square_size * coordinates,
                        (self.pix_square_size, self.pix_square_size)))

    def _render_draw_protection_unit(self, canvas, coordinates):
        """
        Drawing the protection unit, given the coordinates of the top-left corner
        NB: Coordinates are taken on the format [x, y], where x goes leftwards and y goes downwards
        """
        pix_prot_unit_size = self.pix_square_size * self.protection_unit_size
        pix_x_start = coordinates[0] * self.pix_square_size
        pix_y_start = coordinates[1] * self.pix_square_size

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

    def _render_draw_gridlines(self, canvas):
        for x in range(self.grid_size + 1):
            pygame.draw.line(canvas,
                             0, (0, self.pix_square_size * x),
                             (self.window_size, self.pix_square_size * x),
                             width=3)

            pygame.draw.line(canvas,
                             0, (self.pix_square_size * x, 0),
                             (self.pix_square_size * x, self.window_size),
                             width=3)

    def _render_frame(self, obs, prot_units):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # First we draw species 0
        green = (0, 100, 0)
        species_0_population = obs["species_0"]

        for iy, ix in np.ndindex(species_0_population.shape):
            population_in_cell = species_0_population[iy, ix]

            color_tuple = tuple(
                elem -
                int((elem / species_0_population.max()) * population_in_cell)
                for elem in green)

            self._render_fill_square(canvas, color_tuple, np.array([ix, iy]))

        # Finally, add some gridlines
        self._render_draw_gridlines(canvas)

        # Draw protection unit
        for prot_unit_coordinates in prot_units:
            self._render_draw_protection_unit(canvas, prot_unit_coordinates)

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