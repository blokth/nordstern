import simpy
import numpy as np
from jammer import Jammer
from drone import Drone
from signal_model import free_space_path_loss
from estimation import estimate_jammer_movement_ml, make_movement_features_with_imu, load_jammer_estimator
from visualization import animate_simulation
import config


def run_simulation():
    env = simpy.Environment()

    # Create jammer at fixed or random position
    jammer_pos = np.array([config.AREA_SIZE / 2, config.AREA_SIZE / 2])
    jammer = Jammer(jammer_pos)

    # Create drones in a triangle formation (swarm)
    triangle_center = np.array([config.AREA_SIZE / 2, config.AREA_SIZE / 2])
    triangle_size = 5  # meters, side length of the triangle

    # Calculate triangle vertices (equilateral, not pointing at jammer)
    angle_offset = np.pi / 4  # rotate triangle so it's not tangent to jammer
    triangle_points = []
    for i in range(config.NUM_DRONES):
        angle = angle_offset + i * 2 * np.pi / config.NUM_DRONES
        point = triangle_center + triangle_size * np.array([np.cos(angle), np.sin(angle)])
        triangle_points.append(point)

    drones = []
    for pos in triangle_points:
        drone = Drone(env, pos, jammer)
        drones.append(drone)

    # Number of movement steps
    NUM_STEPS = 10
    MOVE_DIST = 10  # meters per step

    # Initialize histories
    drone_positions_hist = []
    estimate_hist = []

    # Initial positions, orientations, RSSI, AoA
    drone_positions1 = np.array([d.position.copy() for d in drones])
    rssi1 = np.array([d.measure_signal() for d in drones])
    # Simulate initial orientations (random for each drone)
    drone_orientations1 = np.random.uniform(-np.pi, np.pi, size=len(drones))
    # Simulate initial AoA
    def simulate_aoa(jammer_pos, drone_pos, orientation):
        vec = jammer_pos - drone_pos
        true_aoa = np.arctan2(vec[1], vec[0])
        aoa = true_aoa - orientation + np.random.normal(0, 0.05)
        return aoa
    aoa1 = np.array([simulate_aoa(jammer.position, d.position, drone_orientations1[i]) for i, d in enumerate(drones)])

    # Initialize running estimate and uncertainty (covariance)
    running_estimate = None
    running_cov = None
    alpha = 0.7  # blending factor for exponential moving average

    # Load ML model once
    ml_model = load_jammer_estimator()

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

        # Simulate IMU: orientation after movement (yaw), velocity, delta_yaw
        delta_pos = drone_positions2 - drone_positions1
        drone_orientations2 = []
        delta_orient = []
        for i in range(len(drones)):
            v_x, v_y = delta_pos[i]
            prev_yaw = drone_orientations1[i]
            yaw = np.arctan2(v_y, v_x) if np.linalg.norm(delta_pos[i]) > 1e-6 else prev_yaw
            drone_orientations2.append(yaw)
            delta_orient.append(yaw - prev_yaw)
        drone_orientations2 = np.array(drone_orientations2)
        delta_orient = np.array(delta_orient)

        # Simulate AoA at new positions
        aoa2 = np.array([simulate_aoa(jammer.position, drone_positions2[i], drone_orientations2[i]) for i in range(len(drones))])
        delta_aoa = aoa2 - aoa1
        delta_rssi = rssi2 - rssi1

        # Build feature vector for all drones concatenated:
        # For each drone: [x1, y1, yaw1, v_x, v_y, aoa1, rssi1, dx, dy, delta_yaw, delta_aoa, delta_rssi]
        features = []
        for i in range(len(drones)):
            yaw1 = drone_orientations1[i]
            v_x = delta_pos[i,0]
            v_y = delta_pos[i,1]
            features.extend([
                drone_positions1[i,0], drone_positions1[i,1],
                yaw1,
                v_x, v_y,
                aoa1[i],
                rssi1[i],
                delta_pos[i,0], delta_pos[i,1],
                delta_orient[i],
                delta_aoa[i],
                delta_rssi[i]
            ])
        features = np.array(features).reshape(1, -1)

        # Estimate jammer position using movement+IMU+AoA ML
        current_estimate = ml_model.predict(features).flatten()

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
        drone_orientations1 = drone_orientations2
        aoa1 = aoa2

    # Visualize the animation
    animate_simulation(drone_positions_hist, jammer, estimate_hist)
