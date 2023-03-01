from config_parser import ConfigParser
from stable_baselines3 import A2C
import numpy as np
import time

np.set_printoptions(suppress=True, formatter={'float': "{0:0.3f}".format})

config_parser = ConfigParser("bio_env_configs/default4.ini")
env = config_parser.create_bio_gym_world()


def train_model():
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10_000, progress_bar=True)
    model.save("trained_models/A2C_test")


def main():
    """
    Main function for running this python script.
    """
    episodes = 2
    score_history = []

    # model = A2C.load("trained_models/A2C_test")

    for i in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        score = 0

        timestep = 0

        while not done and timestep < 100:
            timestep += 1
            action = 0
            # action = env.action_space.sample() if timestep % 5 == 0 else 0
            # action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
        score_history.append(score)

        # env.show_species_history()
    env.show_score_history(score_history)
    env.close()


if __name__ == '__main__':
    # train_model()
    main()