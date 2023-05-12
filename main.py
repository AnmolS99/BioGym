from config_parser import ConfigParser
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from bio_world import BioGymWorld
from agents.no_action import NoAction
from agents.random_action import RandomAction
from agents.user_action import UserAction

np.set_printoptions(suppress=True, formatter={'float': "{0:0.3f}".format})

config_parser = ConfigParser("bio_env_configs/4x4_10x.ini")
bio_env, renderer, max_steps, reduced_actions = config_parser.create_bio_gym_world(
)

env = BioGymWorld(bio_env, renderer, max_steps, reduced_actions)


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env_instance = BioGymWorld(bio_env, renderer, max_steps,
                                   reduced_actions)
        env_instance.renderer.render_mode = "off"
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        # env_instance.reset(seed=seed + rank)
        env_instance.reset()
        return env_instance

    # set_random_seed(seed)
    return _init


def make_train_env():

    train_env = SubprocVecEnv(
        [make_env(i) for i in range(8)],
        start_method="fork",
    )

    train_env = VecMonitor(train_env)
    return train_env


model_type = DQN


def train_model(model_name, timesteps):
    # train_env = make_train_env()
    # print("train_env.n_envs = " + str(train_env.num_envs))
    train_env = env
    train_env.renderer.render_mode = "off"
    model = model_type("MultiInputPolicy",
                       train_env,
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
    shannon_index_history = []

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
        shannon_index_history.append(env.bio_env.get_average_shannon_index())

        if show_episode_history:
            env.show_episode_history()

    print("Average episode score: " +
          str(sum(score_history) / len(score_history)))
    print(
        "Average episode species abundance (relative to critical threshold): "
        + str(sum(species_abundance_history) / len(species_abundance_history)))
    print("Average episode Shannon index: " +
          str(sum(shannon_index_history) / len(shannon_index_history)))

    env.show_run_history(score_history, species_abundance_history,
                         shannon_index_history)
    env.close()


if __name__ == '__main__':
    for i in range(1, 21):

        model_name = "new_DQN_4x4_10x_200k_" + str(i)
        train_model(model_name, 200_000)

    # run(episodes=2,
    #     render_mode="on",
    #     show_episode_history=True,
    #     agent_name="model",
    #     model_name="PPO/1env/3x3_10x/PPO_3x3_10x_200k_20")
