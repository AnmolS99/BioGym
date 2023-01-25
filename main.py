from bio_world import BioGymWorld

env = BioGymWorld(render_mode="human",
                  grid_size=5,
                  sim_height=300,
                  num_species=3)

episodes = 1

for i in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    timestep = 0

    while not done and timestep < 100:
        timestep += 1
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
env.close()