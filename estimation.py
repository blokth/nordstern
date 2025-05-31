import numpy as np

def estimate_jammer(drone_positions, rssi_measurements):
    """
    Estimate jammer position using weighted centroid of drone positions,
    weights proportional to RSSI (higher RSSI means closer).
    """
    weights = 10 ** (rssi_measurements / 10)  # convert dBm to linear scale
    weighted_pos = np.average(drone_positions, axis=0, weights=weights)
    return weighted_pos
