import simpy
import numpy as np
from jammer import Jammer
from drone import Drone
from signal_model import free_space_path_loss
from estimation import estimate_jammer
from visualization import plot_simulation
import config

def run_simulation():
    env = simpy.Environment()

    # Create jammer at fixed or random position
    jammer_pos = np.array([config.AREA_SIZE/2, config.AREA_SIZE/2])
    jammer = Jammer(jammer_pos)

    # Create drones at random positions
    drones = []
    for _ in range(config.NUM_DRONES):
        pos = np.random.uniform(0, config.AREA_SIZE, size=2)
        drone = Drone(env, pos, jammer)
        drones.append(drone)

    # Run simulation
    env.run(until=config.SIM_TIME)

    # Collect data for estimation
    drone_positions = np.array([d.position for d in drones])
    rssi_measurements = np.array([d.rssi for d in drones])

    # Estimate jammer position
    estimated_pos = estimate_jammer(drone_positions, rssi_measurements)

    # Visualize results
    plot_simulation(drones, jammer, estimated_pos)

if __name__ == "__main__":
    run_simulation()
