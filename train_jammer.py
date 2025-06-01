import numpy as np
from estimation import train_jammer_estimator
from signal_model import free_space_path_loss
from config import AREA_SIZE, NUM_DRONES, JAMMER_POWER, NOISE_STD

# Number of training samples
N_SAMPLES = 20000  # Increased for more robust training

def simulate_imu(drone_pos, prev_drone_pos, prev_orientation):
    """
    Simulate IMU readings: orientation (yaw), velocity (v_x, v_y), and delta orientation.
    For simplicity, assume 2D yaw only.
    """
    delta_pos = drone_pos - prev_drone_pos
    v_x, v_y = delta_pos  # velocity approx
    yaw = np.arctan2(v_y, v_x) if np.linalg.norm(delta_pos) > 1e-6 else prev_orientation
    delta_yaw = yaw - prev_orientation
    return yaw, v_x, v_y, delta_yaw

def simulate_aoa(jammer_pos, drone_pos, orientation):
    """
    Simulate AoA (azimuth) from phased array with noise.
    """
    vec = jammer_pos - drone_pos
    true_aoa = np.arctan2(vec[1], vec[0])
    aoa = true_aoa - orientation + np.random.normal(0, 0.05)
    return aoa

X_train = []
y_train = []

for _ in range(N_SAMPLES):
    # Random jammer position in the area
    jammer_pos = np.random.uniform(0, AREA_SIZE, size=2)
    drone_positions = np.random.uniform(0, AREA_SIZE, size=(NUM_DRONES, 2))
    # Simulate RSSI measurements for each drone
    rssi_measurements = []
    for drone_pos in drone_positions:
        rssi = free_space_path_loss(
            jammer_pos, drone_pos, JAMMER_POWER, noise_std=NOISE_STD
        )
        rssi_measurements.append(rssi)
    rssi_measurements = np.array(rssi_measurements)
    # Add small random noise to drone positions for more robustness
    drone_positions += np.random.normal(0, 0.5, drone_positions.shape)
    # Random initial drone positions
    # Build features and append to training set
    X_train.append(make_movement_features(drone_positions1, drone_positions2, rssi1, rssi2))
    # Simulate orientations after movement
    drone_orientations2 = []
    for i in range(NUM_DRONES):
        yaw, v_x, v_y, delta_yaw = simulate_imu(drone_positions2[i], drone_positions1[i], drone_orientations1[i])
        drone_orientations2.append(yaw)
    drone_orientations2 = np.array(drone_orientations2)
    # Simulate RSSI measurements at both positions
    rssi1 = np.array([free_space_path_loss(jammer_pos, pos, JAMMER_POWER, noise_std=NOISE_STD) for pos in drone_positions1])
    rssi2 = np.array([free_space_path_loss(jammer_pos, pos, JAMMER_POWER, noise_std=NOISE_STD) for pos in drone_positions2])
    # Simulate AoA measurements at both positions
    aoa1 = np.array([simulate_aoa(jammer_pos, drone_positions1[i], drone_orientations1[i]) for i in range(NUM_DRONES)])
    aoa2 = np.array([simulate_aoa(jammer_pos, drone_positions2[i], drone_orientations2[i]) for i in range(NUM_DRONES)])
    # Compute deltas
    delta_pos = drone_positions2 - drone_positions1
    delta_orient = drone_orientations2 - drone_orientations1
    delta_aoa = aoa2 - aoa1
    delta_rssi = rssi2 - rssi1

    # Sort drones by RSSI1 (descending: closest first)
    sorted_indices = np.argsort(-rssi1)
    drone_positions1 = drone_positions1[sorted_indices]
    drone_positions2 = drone_positions2[sorted_indices]
    drone_orientations1 = drone_orientations1[sorted_indices]
    drone_orientations2 = drone_orientations2[sorted_indices]
    aoa1 = aoa1[sorted_indices]
    aoa2 = aoa2[sorted_indices]
    rssi1 = rssi1[sorted_indices]
    rssi2 = rssi2[sorted_indices]
    delta_pos = delta_pos[sorted_indices]
    delta_orient = delta_orient[sorted_indices]
    delta_aoa = delta_aoa[sorted_indices]
    delta_rssi = delta_rssi[sorted_indices]

    # Build feature vector for all drones concatenated:
    # For each drone: [x1, y1, yaw1, v_x, v_y, aoa1, rssi1, dx, dy, delta_yaw, delta_aoa, delta_rssi]
    features = []
    for i in range(NUM_DRONES):
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
    X_train.append(np.array(features))
>>>>>>> eeb558e (feat: enhance jammer localization with IMU and phased array features and training updates)
    y_train.append(jammer_pos)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")

# Train and save the model
train_jammer_estimator(X_train, y_train)
print("Training complete. Model saved as jammer_mlp_model.joblib.")
