class UserAction:

    def __init__(self) -> None:
        self.interval = int(
            input("How many time steps between each action?: "))

    def predict(self, obs, timestep):
        if timestep % self.interval == 0:
            return int(input("Action: "))
        else:
            return 0
