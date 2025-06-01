class Drone:
    id = 0

    def __init__(self, pos: tuple[int, int]) -> None:
        self.id: int = Drone.id
        Drone.id += 1
        self.pos: tuple[int, int] = pos
