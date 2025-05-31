import gym
import numpy as np
from gym import spaces
from signal_model import free_space_path_loss

class JammerLocalizationEnv(gym.Env):
    def __init__(self, area_size, jammer_power, noise_std):
        super().__init__()
        self.area_size = area_size
        self.jammer_power = jammer_power
        self.noise_std = noise_std

        # State: [x, y, orientation, AoA, RSSI]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.pi, -np.pi, -150]),
            high=np.array([area_size, area_size, np.pi, np.pi, jammer_power]),
            dtype=np.float32
        )
        # Action: [delta_distance, delta_angle]
        self.action_space = spaces.Box(
            low=np.array([0, -np.pi/4]),
            high=np.array([10, np.pi/4]),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.jammer_pos = np.random.uniform(0, self.area_size, size=2)
        self.drone_pos = np.random.uniform(0, self.area_size, size=2)
        self.orientation = np.random.uniform(-np.pi, np.pi)
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        # AoA: angle from drone to jammer, relative to drone orientation
        vec = self.jammer_pos - self.drone_pos
        true_aoa = np.arctan2(vec[1], vec[0])
        aoa = true_aoa - self.orientation + np.random.normal(0, 0.05)  # add noise
        rssi = free_space_path_loss(self.jammer_pos, self.drone_pos, self.jammer_power, noise_std=self.noise_std)
        return np.array([*self.drone_pos, self.orientation, aoa, rssi], dtype=np.float32)

    def step(self, action):
        delta_dist, delta_angle = action
        self.orientation += delta_angle
        dx = delta_dist * np.cos(self.orientation)
        dy = delta_dist * np.sin(self.orientation)
        self.drone_pos += np.array([dx, dy])
        self.drone_pos = np.clip(self.drone_pos, 0, self.area_size)
        self.steps += 1

        obs = self._get_obs()
        # Reward: negative distance to jammer (encourage getting closer)
        dist = np.linalg.norm(self.drone_pos - self.jammer_pos)
        reward = -dist
        done = dist < 2 or self.steps >= 50
        info = {"jammer_pos": self.jammer_pos, "drone_pos": self.drone_pos}
        return obs, reward, done, info
