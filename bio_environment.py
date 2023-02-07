import numpy as np
from scipy.integrate import odeint
from gymnasium import spaces


class BioEnvironment():
    """
    Biological environment, containing all logic relating to the environment
    """

    def __init__(self, num_species, grid_size, prot_unit_size,
                 diagonal_neighbours, migration_rate, species_ranges, r, k, a,
                 b, e, d, a_2, b_2, e_2, d_2, s, gamma) -> None:
        self.num_species = num_species
        self.grid_size = grid_size
        self.prot_unit_size = prot_unit_size

        self.diagonal_neighbours = diagonal_neighbours

        self.migration_rate = np.array(
            migration_rate)  # Migration rate between cells

        self.species_ranges = species_ranges  # Initial population ranges of the different species

        self.params = [r, k, a, b, e, d, a_2, b_2, e_2, d_2, s, gamma]

        self.species_populations = self.init_species_populations(
        )  # Initialize species populations

        # Set the extinction thresholds
        self.extinction_threshold = np.mean(self.species_populations,
                                            axis=(1, 2)) * 0.05

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

    def sim_grid_step(self):
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                old_cell_state = self.species_populations[:, x, y]
                new_cell_state = self.sim_cell_step(old_cell_state)
                self.species_populations[:, x, y] = new_cell_state

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

        # Dispersal out from cell
        new_species_population = np.einsum("ijk,i->ijk",
                                           self.species_populations,
                                           (1 - self.migration_rate))

        # Dispersal into cell from neighbours
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for neighbour in self.get_cell_neighbours((x, y)):

                    new_species_population[:, x, y] += (
                        self.species_populations[:, neighbour[0], neighbour[1]]
                        * self.migration_rate) / self.get_num_cell_neighbours(
                            (neighbour[0], neighbour[1]))
        self.species_populations = new_species_population

    def sim_extiction(self):
        """
        Simulates extinction for each species, for cells with less population than the extinction threshold
        """
        for i in range(self.num_species):
            self.species_populations[i][
                self.species_populations[i] < self.extinction_threshold[i]] = 0

    def step(self):
        """
        Simulating a step for the whole grid
        """
        # Simulating a step in the tri-trophic system for each cell in the grid
        self.sim_grid_step()

        # Simulating dispersal for the whole grid
        self.sim_dispersal()

        # Simulating extinction
        self.sim_extiction()

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

    def get_grid_size(self):
        """
        Returns the size of the environment grid
        """
        return self.grid_size

    def get_prot_unit_size(self):
        """
        Returns the size of the environment protection unit
        """
        return self.prot_unit_size


def main():
    """
    Main function for running this python script.
    """
    b = BioEnvironment(num_species=3,
                       grid_size=3,
                       prot_unit_size=3,
                       diagonal_neighbours=False,
                       migration_rate=[0.10, 0.05, 0.01],
                       species_ranges=[[0, 70], [0, 20], [0, 1]],
                       r=3.33,
                       k=100,
                       a=2,
                       b=40,
                       e=2.1,
                       d=1,
                       a_2=12.3,
                       b_2=0.47,
                       e_2=0.1,
                       d_2=0.6,
                       s=0.4,
                       gamma=0.1)
    b.species_populations = np.array(
        [[[300, 300, 300], [300, 300, 300], [300, 300, 300]],
         [[500, 500, 500], [500, 500, 500], [500, 500, 500]],
         [[100, 100, 100], [100, 100, 100], [100, 100, 100]]],
        dtype=np.float64)
    print(b.species_populations)
    print(np.mean(b.species_populations, axis=(1, 2)) * [0.15, 0.15, 0.05])


if __name__ == '__main__':
    main()