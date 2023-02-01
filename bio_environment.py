import numpy as np
from scipy.integrate import odeint
from gymnasium import spaces


class BioEnvironment():
    """
    Biological environment, containing all logic relating to the environment
    """

    def __init__(self,
                 num_species,
                 grid_size,
                 diagonal_neighbours=False) -> None:
        self.num_species = num_species
        self.grid_size = grid_size

        self.diagonal_neighbours = diagonal_neighbours
        self.migration_rate = np.array([0.10, 0.05,
                                        0.01])  # Migration rate between cells

        self.species_ranges = [[48, 50], [11, 12], [
            0.01, 0.011
        ]]  # Initial population ranges of the different species

        self.species_populations = self.init_species_populations(
        )  # Initialize species populations

        r = 3.33  # Maximum reproduction per prey
        k = 100  # Carrying capacity of prey in a cell
        a = 2  # Rate of prey consumption by a mesopredator
        b = 40  # The number at which the mesopredator consumption of the prey is half of its maximum

        e = 2.1  # Conversion of prey consumption to mesopredator offspring (was 0.476 in the wildlife book)
        d = 1  # Decrease in the mesopredator population due to natural reasons such as death

        a_2 = 12.3  # Rate of mesopredator consumption by a apex predator
        b_2 = 0.47  # The number at which the apex predator consumption of the mesopredator is half of its maximum
        e_2 = 0.1  # Conversion of mesopredator consumption to apex predator offspring
        d_2 = 0.6  # Decrease in the apex predator population due to natural reasons such as death
        s = 0.4  # Maximum rate of apex predators per capita
        gamma = 0.1  # Maximum apex predator density

        self.params = [r, k, a, b, e, d, a_2, b_2, e_2, d_2, s, gamma]

    def init_species_populations(self, type="numpy") -> dict:
        """
        Initialize populations for the different species (either as matrix (numpy) or Box (spaces))
        """
        if type != "numpy" and type != "Box":
            raise Exception(
                "Type not supported, needs to be either 'numpy' or 'Box'")

        # Creating the observation space
        if type == "Box":

            species_populations = {}

            # Initialize the species population as a Box
            for i in range(self.num_species):
                species_populations["species_" + str(i)] = spaces.Box(
                    0,
                    np.inf,
                    shape=(self.grid_size, self.grid_size),
                    dtype=float)
        else:
            # Initialize numpy ndarray
            species_populations = np.zeros(
                (self.num_species, self.grid_size, self.grid_size))

            for i in range(self.num_species):

                # Initialize the species population as a NumPy ndarray
                species_populations[i] = np.random.uniform(
                    self.species_ranges[i][0],
                    self.species_ranges[i][1],
                    size=(self.grid_size, self.grid_size))

        return species_populations

    def sim_ode(self, variables, t, params):

        if (self.num_species != 3):
            raise NotImplementedError()

        n_1 = variables[0]
        n_2 = variables[1]
        n_3 = variables[2]

        r = params[0]
        k = params[1]
        a = params[2]
        b = params[3]

        e = params[4]
        d = params[5]

        a_2 = params[6]
        b_2 = params[7]
        e_2 = params[8]
        d_2 = params[9]
        s = params[10]
        gamma = params[11]

        dn_1dt = r * (1 - (n_1 / k)) - ((a * n_1 * n_2) / (b + n_1))
        dn_2dt = ((a * e * n_1 * n_2) /
                  (b + n_1)) - (d * n_2) - ((a_2 * n_2 * n_3) / (b_2 + n_2))
        dn_3dt = ((a_2 * e_2 * n_2 * n_3) /
                  (b_2 + n_2)) - (d_2 * n_3) - ((s * (n_3)**2) / gamma)

        return [dn_1dt, dn_2dt, dn_3dt]

    def sim_cell_step(self, y0) -> np.array:
        """
        Simulating a step in a single cell of the grid
        """

        t = np.linspace(
            0, 2,
            num=2)  # Two timesteps: t0 (current state) and t1 (next state)
        y = odeint(self.sim_ode, y0, t, args=(self.params, ))
        return y[1]

    def get_num_cell_neighbours(self, coordinates):
        """
        Returns the number of neighbouring cells.
        Takes in cell coordinates as a tuple (x, y)
        """
        x = coordinates[0]
        y = coordinates[1]
        # If the cell is on the top/bottom edge
        if (x == 0 or x == (self.grid_size - 1)):

            # If the cell is also on left/right edge
            if (y == 0 or y == (self.grid_size - 1)):
                return 3 if self.diagonal_neighbours else 2

            return 5 if self.diagonal_neighbours else 3

        # If the cell is only on left/right edge
        elif (y == 0 or y == (self.grid_size - 1)):
            return 5 if self.diagonal_neighbours else 3

        # If cell isnt on any edge
        else:
            return 8 if self.diagonal_neighbours else 4

    def get_cell_neighbours(self, coordinates):
        """
        Returns the neighbouring cells as in a list.
        Takes in cell coordinates as a tuple (x, y)
        """
        x = coordinates[0]
        y = coordinates[1]

        neighbours = []
        # If there is a neighbour over
        if x > 0:
            neighbours.append((x - 1, y))
            # If diagonal neighbours count
            if self.diagonal_neighbours:
                # If there is also a neighbour to the left, meaning there is a neighbour top-left
                if y > 0:
                    neighbours.append((x - 1, y - 1))
                # If there is also a neighbour to the right, meaning there is a neighbour top-right
                if y < self.grid_size - 1:
                    neighbours.append((x - 1, y + 1))

        # If there is a neighbour below
        if x < self.grid_size - 1:
            neighbours.append((x + 1, y))
            # If diagonal neighbours count
            if self.diagonal_neighbours:
                # If there is also a neighbour to the left, meaning there is a neighbour bottom-left
                if y > 0:
                    neighbours.append((x + 1, y - 1))
                # If there is also a neighbour to the right, meaning there is a neighbour bottom-right
                if y < self.grid_size - 1:
                    neighbours.append((x + 1, y + 1))

        # If there is a neighbour to the left
        if y > 0:
            neighbours.append((x, y - 1))
        # If there is a neighbour to the right
        if y < self.grid_size - 1:
            neighbours.append((x, y + 1))
        return neighbours

    def sim_dispersal(self):
        "Simulates dispersal in and out the cell to neighbouring cells, for all cells in the grid"

        # Create copy of species_populations
        new_species_population = np.copy(self.species_populations)

        # Dispersal out from cell
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                new_species_population[:, x,
                                       y] *= 1 - (self.migration_rate *
                                                  self.get_num_cell_neighbours(
                                                      (x, y)))

        # Dispersal into cell from neighbours
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for neighbour in self.get_cell_neighbours((x, y)):
                    new_species_population[:, x,
                                           y] += self.species_populations[:, neighbour[
                                               0], neighbour[
                                                   1]] * self.migration_rate
        self.species_populations = new_species_population

    def step(self):
        """
        Simulating a step for the whole grid
        """
        # Simulating a step in the tri-trophic system for each cell in the grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                old_cell_state = self.species_populations[:, x, y]
                new_cell_state = self.sim_cell_step(old_cell_state)
                self.species_populations[:, x, y] = new_cell_state

        # Simulating dispersal for the whole grid
        self.sim_dispersal()

    def get_obs(self):
        """
        Returns detailed information about the current status of the BioEnvironment
        """
        return self.species_populations

    def reset(self):
        """
        Reseting environment
        """
        self.species_populations = self.init_species_populations()


def main():
    """
    Main function for running this python script.
    """
    b = BioEnvironment(num_species=3, grid_size=3, diagonal_neighbours=False)
    b.species_populations = np.array(
        [[[300, 500, 100], [100, 200, 400], [700, 600, 100]],
         [[300, 500, 100], [100, 200, 400], [700, 600, 100]],
         [[300, 500, 100], [100, 200, 400], [700, 600, 100]]],
        dtype=np.float64)
    print(b.species_populations)
    b.sim_dispersal()
    print(b.species_populations)


if __name__ == '__main__':
    main()