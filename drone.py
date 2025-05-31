import simpy
import numpy as np
from signal_model import free_space_path_loss
import config

class Drone:
    def __init__(self, env, position, jammer):
        self.env = env
        self.position = position
        self.jammer = jammer
        self.rssi = None
        self.measurements = []
        self.process = env.process(self.run())

    def measure_signal(self):
        rssi = free_space_path_loss(self.jammer.position, self.position, config.JAMMER_POWER, config.NOISE_STD)
        self.rssi = rssi
        self.measurements.append((self.env.now, rssi))
        return rssi

    def run(self):
        while True:
            self.measure_signal()
            yield self.env.timeout(config.MEASUREMENT_INTERVAL)
