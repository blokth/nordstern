import numpy as np

def free_space_path_loss(jammer_pos, drone_pos, jammer_power, noise_std=2.0):
    """
    Simulate RSSI at drone_pos from jammer_pos using free-space path loss model.
    jammer_power: dBm (transmit power)
    Returns: RSSI in dBm (with noise)
    """
    d = np.linalg.norm(np.array(jammer_pos) - np.array(drone_pos))
    if d < 1.0:
        d = 1.0  # avoid log(0)
    # Free-space path loss (FSPL) in dB: FSPL = 20*log10(d) + 20*log10(f) - 27.55
    # For simplicity, assume f=2.4 GHz, so 20*log10(f) - 27.55 is constant
    fspl = 20 * np.log10(d) + 40  # 40 is a rough constant for 2.4 GHz
    rssi = jammer_power - fspl
    noise = np.random.normal(0, noise_std)
    return rssi + noise
