from config_parser import ConfigParser
from stable_baselines3 import A2C
import numpy as np
from agents.no_action import NoAction
from agents.random_action import RandomAction
from agents.user_action import UserAction

np.set_printoptions(suppress=True, formatter={'float': "{0:0.3f}".format})

config_parser = ConfigParser("bio_env_configs/default4.ini")
env = config_parser.create_bio_gym_world()


def train_model():
    env.renderer.render_mode = "off"
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(
        total_timesteps=5_000,
        progress_bar=True,
    )
    model.save("trained_models/A2C_test")


def get_agent(name):
    if name == "no action":
        return NoAction()
    elif name == "random":
        return RandomAction(env)
    elif name == "user":
        return UserAction()
    else:
        raise NotImplementedError("Agent not implemented.")


def run(episodes, render_mode, show_species_history, agent_name):
    """
    Run episodes of environment with given RL agent.
    """
    score_history = []

    env.renderer.render_mode = render_mode

    if agent_name == "model":
        agent = A2C.load("trained_models/A2C_test")
    else:
        agent = get_agent(agent_name)

    for i in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        score = 0

        timestep = 0

        while not done and timestep < 100:
            timestep += 1

            if agent_name == "model":
                action, _state = agent.predict(obs, deterministic=True)
            else:
                action = agent.predict(obs, timestep)

            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
        score_history.append(score)

        if show_species_history:
            env.show_species_history()

    env.show_score_history(score_history)
    env.close()


if __name__ == '__main__':
    # train_model()
    run(episodes=5,
        render_mode="on",
        show_species_history=True,
        agent_name="no action")
