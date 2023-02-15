from config_parser import ConfigParser

config_parser = ConfigParser("bio_env_configs/default2.ini")
env = config_parser.create_bio_gym_world()

episodes = 2

for i in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    timestep = 0

    while not done and timestep < 5:
        timestep += 1
        action = 0
        # action = env.action_space.sample() if timestep % 5 == 0 else 0
        n_state, reward, done, info = env.step(action)
        score += reward
    env.show_species_history()
env.close()