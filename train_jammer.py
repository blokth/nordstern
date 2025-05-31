import numpy as np
from estimation import train_jammer_estimator, _flatten_features
from signal_model import free_space_path_loss
from config import AREA_SIZE, NUM_DRONES, JAMMER_POWER, NOISE_STD

# Number of training samples
N_SAMPLES = 5000

X_train = []
y_train = []

for _ in range(N_SAMPLES):
    # Random jammer position in the area
    jammer_pos = np.random.uniform(0, AREA_SIZE, size=2)
    # Random drone positions in the area
    drone_positions = np.random.uniform(0, AREA_SIZE, size=(NUM_DRONES, 2))
    # Simulate RSSI measurements for each drone
    rssi_measurements = []
    for drone_pos in drone_positions:
        rssi = free_space_path_loss(jammer_pos, drone_pos, JAMMER_POWER, noise_std=NOISE_STD)
        rssi_measurements.append(rssi)
    rssi_measurements = np.array(rssi_measurements)
    # Flatten features and append to training set
    X_train.append(_flatten_features(drone_positions, rssi_measurements))
    y_train.append(jammer_pos)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Train and save the model
train_jammer_estimator(X_train, y_train)
print("Training complete. Model saved as jammer_mlp_model.joblib.")
