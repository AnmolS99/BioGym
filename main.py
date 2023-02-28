from config_parser import ConfigParser
import numpy as np
import time

config_parser = ConfigParser("bio_env_configs/default3.ini")
env = config_parser.create_bio_gym_world()
np.set_printoptions(suppress=True, formatter={'float': "{0:0.3f}".format})


def main():
    """
    Main function for running this python script.
    """
    episodes = 3
    score_history = []

    start = time.time()
    for i in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        timestep = 0

        start_ep = time.time()

        while not done and timestep < 100:
            timestep += 1
            action = 0
            action = env.action_space.sample() if timestep % 1 == 0 else 0
            n_state, reward, done, info = env.step(action)
            score += reward
        score_history.append(score)

        duration = time.time() - start_ep
        print("---> Episode " + str(i) + ": " + str(duration))
        env.show_species_history()
    tot_duration = time.time() - start
    print("Total time: " + str(tot_duration))
    env.show_score_history(score_history)
    env.close()


if __name__ == '__main__':
    main()