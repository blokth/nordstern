import numpy as np


def free_space_path_loss(tx_pos, rx_pos, tx_power, noise_std=0):
    """
    Compute RSSI at rx_pos from tx_pos using free-space path loss.
    tx_power in dBm.
    """
    dist = np.linalg.norm(tx_pos - rx_pos)
    if dist < 1e-6:
        dist = 1e-6  # avoid log(0)
    # Free space path loss formula (dB)
    fspl = 20 * np.log10(dist) + 20 * np.log10(2.4e9) - 147.55  # freq=2.4GHz
    rssi = tx_power - fspl
    if noise_std > 0:
        rssi += np.random.normal(0, noise_std)
    return rssi
