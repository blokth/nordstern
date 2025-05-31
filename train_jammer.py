import numpy as np
from estimation import train_jammer_estimator, make_movement_features
from signal_model import free_space_path_loss
from config import AREA_SIZE, NUM_DRONES, JAMMER_POWER, NOISE_STD

# Number of training samples
N_SAMPLES = 20000  # Increased for more robust training

X_train = []
y_train = []

for _ in range(N_SAMPLES):
    # Random jammer position in the area
    jammer_pos = np.random.uniform(0, AREA_SIZE, size=2)
    # Random drone positions in the area
<<<<<<< HEAD
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
=======
    drone_positions1 = np.random.uniform(0, AREA_SIZE, size=(NUM_DRONES, 2))
    # Move each drone 10m in a random direction
    angles = np.random.uniform(0, 2*np.pi, size=NUM_DRONES)
    move = np.stack([10*np.cos(angles), 10*np.sin(angles)], axis=1)
    drone_positions2 = drone_positions1 + move
    # Simulate RSSI measurements at both positions
    rssi1 = np.array([free_space_path_loss(jammer_pos, pos, JAMMER_POWER, noise_std=NOISE_STD) for pos in drone_positions1])
    rssi2 = np.array([free_space_path_loss(jammer_pos, pos, JAMMER_POWER, noise_std=NOISE_STD) for pos in drone_positions2])
    # Add small random noise to drone positions for robustness
    drone_positions1 += np.random.normal(0, 0.5, drone_positions1.shape)
    drone_positions2 += np.random.normal(0, 0.5, drone_positions2.shape)
>>>>>>> 1bc2def (feat: add movement-based features and training for improved jammer estimation)
    # Build features and append to training set
    X_train.append(make_movement_features(drone_positions1, drone_positions2, rssi1, rssi2))
    y_train.append(jammer_pos)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Training set shape: {X_train.shape}, Labels shape: {y_train.shape}")

# Train and save the model
train_jammer_estimator(X_train, y_train)
print("Training complete. Model saved as jammer_mlp_model.joblib.")
