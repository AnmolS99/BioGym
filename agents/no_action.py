class NoAction:

    def __init__(self, env) -> None:
        self.env = env

    def predict(self, obs, timestep):
        return self.env.bio_env.get_no_action()
