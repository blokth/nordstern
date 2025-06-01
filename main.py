import matplotlib.pyplot as plt
import numpy as np
from config import MAP_SIZE
from drone import Drone
from jammer import Jammer


def main():
    pos_d = np.random.randint(low=0, high=MAP_SIZE, size=2)
    pos_j = np.random.randint(low=0, high=MAP_SIZE, size=2)

    d = Drone(tuple(pos_d))
    j = Jammer(tuple(pos_j))

    plt.plot(d.pos[0], d.pos[1], "bo")
    plt.plot(j.pos[0], j.pos[1], "ro")
    plt.xlim([0, MAP_SIZE])
    plt.ylim([0, MAP_SIZE])
    plt.show()


if __name__ == "__main__":
    main()
