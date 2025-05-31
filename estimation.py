import numpy as np

# --- ML-based Estimator using scikit-learn ---
from sklearn.neural_network import MLPRegressor
import joblib
import os

_MODEL_PATH = "jammer_mlp_model.joblib"

def _make_features(drone_positions, rssi_measurements):
    """
    Construct features using all drone positions and all pairwise RSSI differences.
    Returns a 1D feature vector.
    """
    drone_positions = np.asarray(drone_positions).flatten()
    rssi_measurements = np.asarray(rssi_measurements).flatten()
    # Compute all pairwise RSSI differences (i < j)
    diff_features = []
    N = len(rssi_measurements)
    for i in range(N):
        for j in range(i+1, N):
            diff_features.append(rssi_measurements[i] - rssi_measurements[j])
    diff_features = np.array(diff_features)
    return np.concatenate([drone_positions, diff_features])

def train_jammer_estimator(X_train, y_train, save_path=_MODEL_PATH):
    """
    Train an MLPRegressor to estimate jammer position.
    X_train: array of shape (n_samples, n_features)
    y_train: array of shape (n_samples, 2) -- true jammer positions
    """
    model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    return model

def load_jammer_estimator(model_path=_MODEL_PATH):
    """
    Load a trained jammer estimator model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please train it first.")
    return joblib.load(model_path)

def estimate_jammer_ml(drone_positions, rssi_measurements, model=None):
    """
    Estimate jammer position using a trained ML model.
    drone_positions: (N, 2)
    rssi_measurements: (N,)
    model: trained scikit-learn regressor (if None, loads from disk)
    Returns: (2,) estimated jammer position
    """
    if model is None:
        model = load_jammer_estimator()
    features = _make_features(drone_positions, rssi_measurements).reshape(1, -1)
    pred = model.predict(features)
    return pred.flatten()

# --- Reference: Weighted Centroid Estimator ---
def estimate_jammer(drone_positions, rssi_measurements):
    """
    Estimate jammer position using weighted centroid of drone positions,
    weights proportional to RSSI (higher RSSI means closer).
    """
    weights = 10 ** (rssi_measurements / 10)  # convert dBm to linear scale
    weighted_pos = np.average(drone_positions, axis=0, weights=weights)
    return weighted_pos
