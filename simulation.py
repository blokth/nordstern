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

    # Initialize running estimate and uncertainty (covariance)
    running_estimate = None
    running_cov = None
    alpha = 0.7  # blending factor for exponential moving average

    for step in range(NUM_STEPS):
        # Swarm-like movement: cohesion, separation, alignment
        positions = np.array([d.position for d in drones])
        velocities = []
        for i, d in enumerate(drones):
            # Cohesion: move toward center of mass
            center_of_mass = positions.mean(axis=0)
            cohesion_vec = center_of_mass - d.position

            # Separation: avoid crowding neighbors
            separation_vec = np.zeros(2)
            for j, other in enumerate(drones):
                if i != j:
                    diff = d.position - other.position
                    dist = np.linalg.norm(diff)
                    if dist < 15:  # separation threshold
                        if dist > 1e-3:
                            separation_vec += diff / dist

            # Alignment: match average direction (here, just use previous direction if available)
            # For simplicity, random small alignment
            alignment_vec = np.random.uniform(-1, 1, size=2)

            # Weighted sum of behaviors
            move_vec = (
                0.6 * cohesion_vec +
                1.2 * separation_vec +
                0.3 * alignment_vec
            )
            # Normalize and scale to MOVE_DIST
            norm = np.linalg.norm(move_vec)
            if norm > 1e-3:
                move_vec = (move_vec / norm) * MOVE_DIST
            else:
                move_vec = np.random.uniform(-1, 1, size=2)
                move_vec = (move_vec / np.linalg.norm(move_vec)) * MOVE_DIST

            velocities.append(move_vec)

        # Apply movement and clip to area
        for d, v in zip(drones, velocities):
            d.position += v
            d.position = np.clip(d.position, 0, config.AREA_SIZE)

        drone_positions2 = np.array([d.position.copy() for d in drones])
        rssi2 = np.array([d.measure_signal() for d in drones])

        # Estimate jammer position using movement ML
        current_estimate = estimate_jammer_movement_ml(drone_positions1, drone_positions2, rssi1, rssi2)

        # Improve estimate over time using exponential moving average
        if running_estimate is None:
            running_estimate = current_estimate
        else:
            running_estimate = alpha * running_estimate + (1 - alpha) * current_estimate

        estimate_hist.append(running_estimate.copy())
        drone_positions_hist.append(drone_positions2.copy())

        # Prepare for next step
        drone_positions1 = drone_positions2
        rssi1 = rssi2

    # Visualize the animation
    animate_simulation(drone_positions_hist, jammer, estimate_hist)
