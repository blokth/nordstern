import simpy
import numpy as np
from jammer import Jammer
from drone import Drone
from signal_model import free_space_path_loss
from estimation import estimate_jammer_movement_ml
from visualization import animate_simulation
import config


def run_simulation():
    env = simpy.Environment()

    # Create jammer at fixed or random position
    jammer_pos = np.array([config.AREA_SIZE / 2, config.AREA_SIZE / 2])
    jammer = Jammer(jammer_pos)

    # Create drones at random positions
    drones = []
    for _ in range(config.NUM_DRONES):
        pos = np.random.uniform(0, config.AREA_SIZE, size=2)
        drone = Drone(env, pos, jammer)
        drones.append(drone)

    # Number of movement steps
    NUM_STEPS = 10
    MOVE_DIST = 10  # meters per step

    # Initialize histories
    drone_positions_hist = []
    estimate_hist = []

    # Initial positions and RSSI
    drone_positions1 = np.array([d.position.copy() for d in drones])
    rssi1 = np.array([d.measure_signal() for d in drones])

    for step in range(NUM_STEPS):
        # Move each drone in a random direction
        for d in drones:
            angle = np.random.uniform(0, 2 * np.pi)
            d.position += MOVE_DIST * np.array([np.cos(angle), np.sin(angle)])
            d.position = np.clip(d.position, 0, config.AREA_SIZE)

        drone_positions2 = np.array([d.position.copy() for d in drones])
        rssi2 = np.array([d.measure_signal() for d in drones])

        # Estimate jammer position using movement ML
        estimate = estimate_jammer_movement_ml(drone_positions1, drone_positions2, rssi1, rssi2)
        estimate_hist.append(estimate)
        drone_positions_hist.append(drone_positions2.copy())

        # Prepare for next step
        drone_positions1 = drone_positions2
        rssi1 = rssi2

    # Visualize the animation
    animate_simulation(drone_positions_hist, jammer, estimate_hist)
