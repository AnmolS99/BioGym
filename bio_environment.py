import numpy as np
from scipy.integrate import odeint


class BioEnvironment():
    """
    Biological environment, containing all logic relating to the environment
    """

    def __init__(self, num_species, grid_size) -> None:
        self.num_species = num_species
        self.grid_size = grid_size

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

    def init_species_populations(self) -> dict:
        """
        Initialize populations for the different species
        """

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

    def sim_step(self, y0) -> np.array:
        t = np.linspace(
            0, 2,
            num=2)  # Two timesteps: t0 (current state) and t1 (next state)
        y = odeint(self.sim_ode, y0, t, args=(self.params, ))
        return y[1]


def main():
    """
    Main function for running this python script.
    """
    b = BioEnvironment(3, 5)
    print(b.sim_step([50, 12, 0.01]))


if __name__ == '__main__':
    main()