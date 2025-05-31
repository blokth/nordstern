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

        # State: [x1, y1, x2, y2, dx, dy, rssi1, rssi2, drssi, orientation, aoa1, aoa2]
        # This mimics the movement-based ML features
        low = np.array(
            [0, 0, 0, 0, -area_size, -area_size, -150, -150, -100, -np.pi, -np.pi, -np.pi]
        )
        high = np.array(
            [area_size, area_size, area_size, area_size, area_size, area_size, jammer_power, jammer_power, 100, np.pi, np.pi, np.pi]
        )
        self.observation_space = spaces.Box(
            low=low,
            high=high,
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

        # For movement-based features
        self.prev_drone_pos = self.drone_pos.copy()
        self.prev_orientation = self.orientation
        self.prev_rssi = self._get_rssi(self.prev_drone_pos)
        self.prev_aoa = self._get_aoa(self.prev_drone_pos, self.prev_orientation)
        return self._get_obs()

    def _get_rssi(self, pos):
        return free_space_path_loss(self.jammer_pos, pos, self.jammer_power, noise_std=self.noise_std)

    def _get_aoa(self, pos, orientation):
        vec = self.jammer_pos - pos
        true_aoa = np.arctan2(vec[1], vec[0])
        aoa = true_aoa - orientation + np.random.normal(0, 0.05)
        return aoa

    def _get_obs(self):
        # Features: [x1, y1, x2, y2, dx, dy, rssi1, rssi2, drssi, orientation, aoa1, aoa2]
        x1, y1 = self.prev_drone_pos
        x2, y2 = self.drone_pos
        dx, dy = x2 - x1, y2 - y1
        rssi1 = self.prev_rssi
        rssi2 = self._get_rssi(self.drone_pos)
        drssi = rssi2 - rssi1
        aoa1 = self.prev_aoa
        aoa2 = self._get_aoa(self.drone_pos, self.orientation)
        obs = np.array([x1, y1, x2, y2, dx, dy, rssi1, rssi2, drssi, self.orientation, aoa1, aoa2], dtype=np.float32)
        return obs

    def step(self, action):
        delta_dist, delta_angle = action
        self.orientation += delta_angle
        dx = delta_dist * np.cos(self.orientation)
        dy = delta_dist * np.sin(self.orientation)
        self.prev_drone_pos = self.drone_pos.copy()
        self.prev_rssi = self._get_rssi(self.prev_drone_pos)
        self.prev_aoa = self._get_aoa(self.prev_drone_pos, self.orientation)
        self.drone_pos += np.array([dx, dy])
        self.drone_pos = np.clip(self.drone_pos, 0, self.area_size)
        self.steps += 1

        obs = self._get_obs()
        dist = np.linalg.norm(self.drone_pos - self.jammer_pos)
        reward = -dist
        done = dist < 2 or self.steps >= 50
        info = {"jammer_pos": self.jammer_pos, "drone_pos": self.drone_pos}
        return obs, reward, done, info
