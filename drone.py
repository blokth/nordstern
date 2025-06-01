import numpy as np
from signal_model import free_space_path_loss

class Drone:
    def __init__(self, env, position, jammer):
        self.env = env
        self.position = np.array(position)
        self.jammer = jammer

    def measure_signal(self):
        # Simulate RSSI measurement from the jammer at the drone's position
        return free_space_path_loss(self.jammer.position, self.position)
