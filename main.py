from config_parser import ConfigParser
from stable_baselines3 import DQN, A2C, PPO
import numpy as np
from agents.no_action import NoAction
from agents.random_action import RandomAction
from agents.user_action import UserAction

np.set_printoptions(suppress=True, formatter={'float': "{0:0.3f}".format})

config_parser = ConfigParser("bio_env_configs/2x2_10x.ini")
env = config_parser.create_bio_gym_world()

model_type = PPO


def train_model(model_name, timesteps):
    env.renderer.render_mode = "off"
    model = model_type("MultiInputPolicy",
                       env,
                       verbose=1,
                       tensorboard_log="./logs/")
    model.learn(total_timesteps=timesteps,
                progress_bar=True,
                tb_log_name=model_name)
    model.save("trained_models/" + model_name)


def get_agent(name):
    if name == "no action":
        return NoAction(env)
    elif name == "random":
        return RandomAction(env)
    elif name == "user":
        return UserAction(env)
    else:
        raise NotImplementedError("Agent not implemented.")


def run(episodes,
        render_mode,
        show_episode_history,
        agent_name,
        model_name=None):
    """
    Run episodes of environment with given RL agent.
    """
    score_history = []
    species_abundance_history = []
    species_evenness_history = []

    env.renderer.render_mode = render_mode

    if agent_name == "model":
        agent = model_type.load("trained_models/" + model_name)
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
        species_abundance_history.append(
            env.bio_env.get_average_species_abundance())
        species_evenness_history.append(
            env.bio_env.get_average_species_evenness())

        if show_episode_history:
            env.show_episode_history()

    print("Average episode score: " +
          str(sum(score_history) / len(score_history)))
    print(
        "Average episode species abundance (relative to critical threshold): "
        + str(sum(species_abundance_history) / len(species_abundance_history)))
    print("Average episode species evenness: " +
          str(sum(species_evenness_history) / len(species_evenness_history)))

    env.show_run_history(score_history, species_abundance_history,
                         species_evenness_history)
    env.close()


if __name__ == '__main__':
    for i in range(1, 21):

        model_name = "PPO_2x2_10x_200k_" + str(i)
        train_model(model_name, 200_000)

    # run(episodes=2,
    #     render_mode="on",
    #     show_episode_history=True,
    #     agent_name="model",
    #     model_name="DQN/2x2_10x/DQN_2x2_10x_200k_4")
