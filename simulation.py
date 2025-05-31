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

    # Place drones in a triangle formation at 60-80m from the jammer, tangent to the circle
    jammer_pos = np.array([config.AREA_SIZE / 2, config.AREA_SIZE / 2])
    distance_from_jammer = np.random.uniform(60, 80)
    triangle_size = 5  # meters, side length of the triangle

    # Choose a random angle for the center of the triangle on the circle
    center_angle = np.random.uniform(0, 2 * np.pi)
    # The center of the triangle is at this point on the circle
    triangle_center = jammer_pos + distance_from_jammer * np.array([np.cos(center_angle), np.sin(center_angle)])

    # The triangle should be tangent to the circle at this point.
    # The tangent direction is perpendicular to the radius vector.
    tangent_angle = center_angle + np.pi / 2  # Perpendicular to radius

    # Place the triangle vertices around the center, rotated so the triangle's base is tangent to the circle
    triangle_points = []
    for i in range(config.NUM_DRONES):
        angle = tangent_angle + i * 2 * np.pi / config.NUM_DRONES
        point = triangle_center + triangle_size * np.array([np.cos(angle), np.sin(angle)])
        triangle_points.append(point)

    # Store relative offsets from the triangle center for each drone
    relative_offsets = [point - triangle_center for point in triangle_points]

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

    # Choose a fixed movement direction for the group (e.g., tangent to the circle)
    group_move_direction = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])
    group_move_direction = group_move_direction / np.linalg.norm(group_move_direction)  # ensure unit vector

    # Initialize triangle center
    current_triangle_center = triangle_center.copy()

    for step in range(NUM_STEPS):
        # Move the triangle center in the chosen direction
        current_triangle_center += group_move_direction * MOVE_DIST
        # Optionally, clip to area bounds
        current_triangle_center = np.clip(current_triangle_center, 0, config.AREA_SIZE)

        # Update drone positions to maintain triangle formation
        for i, d in enumerate(drones):
            d.position = current_triangle_center + relative_offsets[i]
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
