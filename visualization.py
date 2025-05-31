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

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def animate_simulation(drone_positions_hist, jammer, estimate_hist):
    fig, ax = plt.subplots(figsize=(8,8))
    # Set axis limits to cover the whole area (use config.AREA_SIZE if available)
    max_x = max([positions[:,0].max() for positions in drone_positions_hist] + [jammer.position[0]])
    max_y = max([positions[:,1].max() for positions in drone_positions_hist] + [jammer.position[1]])
    ax.set_xlim(0, max_x + 10)
    ax.set_ylim(0, max_y + 10)

    jammer_dot, = ax.plot([jammer.position[0]], [jammer.position[1]], 'rx', label='Jammer', markersize=12)
    drones_scatter = ax.scatter([], [], c='blue', label='Drones')
    estimate_dot, = ax.plot([], [], 'go', label='Estimate', markersize=8)

    def update(frame):
        drone_positions = drone_positions_hist[frame]
        estimate = np.asarray(estimate_hist[frame])
        drones_scatter.set_offsets(drone_positions)
        # set_data expects sequences, so wrap in list or np.array
        estimate_dot.set_data([estimate[0]], [estimate[1]])
        return drones_scatter, estimate_dot

    ax.legend()
    ani = FuncAnimation(fig, update, frames=len(drone_positions_hist), interval=700, blit=True)
    plt.title("Drone Jammer Localization (Animated)")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.grid(True)
    plt.show()
