from bio_world import BioGymWorld

env = BioGymWorld(render_mode="human",
                  sns_renderer=True,
                  grid_size=5,
                  sim_height=300,
                  num_species=3,
                  display_population=True)

episodes = 1

for i in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    timestep = 0

    while not done and timestep < 5:
        timestep += 1
        #action = env.action_space.sample()
        action = 1
        n_state, reward, done, info = env.step(action)
        score += reward
env.close()