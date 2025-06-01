class Jammer:
    id = 0

    def __init__(self, pos: tuple[int, int]) -> None:
        self.id: int = Jammer.id
        Jammer.id += 1
        self.pos: tuple[int, int] = pos
