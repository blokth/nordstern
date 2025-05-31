import matplotlib.pyplot as plt
import numpy as np

def plot_simulation(drones, jammer, estimated_position, rssi_field=None):
    plt.figure(figsize=(8,8))
    # Plot drones
    drone_positions = np.array([d.position for d in drones])
    plt.scatter(drone_positions[:,0], drone_positions[:,1], c='blue', label='Drones')
    # Plot jammer
    plt.scatter(jammer.position[0], jammer.position[1], c='red', marker='x', s=100, label='Jammer')
    # Plot estimated position
    plt.scatter(estimated_position[0], estimated_position[1], c='green', marker='o', s=100, label='Estimated Jammer')
    plt.legend()
    plt.xlim(0, max(drone_positions[:,0].max(), jammer.position[0]) + 10)
    plt.ylim(0, max(drone_positions[:,1].max(), jammer.position[1]) + 10)
    plt.title("Drone Jammer Localization")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True)
    plt.show()
