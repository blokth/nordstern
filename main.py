import matplotlib.pyplot as plt
from config import MAP_SIZE
from drone import Drone
from jammer import Jammer


def main():
    d = Drone((50, 50))
    j = Jammer((20, 60))
    plt.plot(d.pos[0], d.pos[1], "bo")
    plt.plot(j.pos[0], j.pos[1], "ro")
    plt.xlim([0, MAP_SIZE])
    plt.ylim([0, MAP_SIZE])
    plt.show()


if __name__ == "__main__":
    main()
