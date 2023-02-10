from config_parser import ConfigParser

config_parser = ConfigParser("bio_env_configs/default2.ini")
env = config_parser.create_bio_gym_world()
print("Exctinction thresholds: " + str(env.bio_env.extinction_threshold))

episodes = 1

for i in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    timestep = 0

    while not done and timestep < 100:
        timestep += 1
        # action = env.action_space.sample()
        action = 1
        n_state, reward, done, info = env.step(action)
        score += reward
env.close()