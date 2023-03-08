class UserAction:

    def __init__(self, env) -> None:
        self.env = env
        self.interval = int(
            input("How many time steps between each action?: "))

    def predict(self, obs, timestep):
        if timestep % self.interval == 0:
            return int(input("Action: "))
        else:
            return self.env.bio_env.get_no_action()
