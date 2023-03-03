class RandomAction:

    def __init__(self, env) -> None:
        self.env = env
        self.interval = int(
            input("How many time steps between each action?: "))

    def predict(self, obs, timestep):
        if timestep % self.interval == 0:
            return self.env.action_space.sample()
        else:
            return self.env.bio_env.get_no_action()
