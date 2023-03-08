from config_parser import ConfigParser
from stable_baselines3 import A2C, DQN, PPO
import numpy as np
from agents.no_action import NoAction
from agents.random_action import RandomAction
from agents.user_action import UserAction

np.set_printoptions(suppress=True, formatter={'float': "{0:0.3f}".format})

config_parser = ConfigParser("bio_env_configs/default5.ini")
env = config_parser.create_bio_gym_world()

model_type = PPO
model_path = "trained_models/model_test"


def train_model():
    env.renderer.render_mode = "off"
    model = model_type("MultiInputPolicy",
                       env,
                       verbose=1,
                       tensorboard_log="./logs/")
    model.learn(
        total_timesteps=10_000,
        progress_bar=True,
    )
    model.save(model_path)


def get_agent(name):
    if name == "no action":
        return NoAction(env)
    elif name == "random":
        return RandomAction(env)
    elif name == "user":
        return UserAction(env)
    else:
        raise NotImplementedError("Agent not implemented.")


def run(episodes, render_mode, show_species_history, agent_name):
    """
    Run episodes of environment with given RL agent.
    """
    score_history = []

    env.renderer.render_mode = render_mode

    if agent_name == "model":
        agent = model_type.load(model_path)
    else:
        agent = get_agent(agent_name)

    for i in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        score = 0

        while not done:

            if agent_name == "model":
                action, _state = agent.predict(obs, deterministic=True)
            else:
                action = agent.predict(obs, env.current_step)

            obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated

        score_history.append(score)

        if show_species_history:
            env.show_species_history()
    print("Average episode score: " +
          str(sum(score_history) / len(score_history)))
    env.show_score_history(score_history)
    env.close()


if __name__ == '__main__':
    train_model()
    run(episodes=10,
        render_mode="off",
        show_species_history=False,
        agent_name="model")
