class VehicleTrace:
    def __init__(
        self,
        id: int,
        timestamp: float,
        x: float,
        y: float,
        angle: float,
        type: str,
        speed: float,
        pos: float,
        lane: str,
        slope: float,
    ) -> None:
        self.id = id
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.angle = angle
        self.type = type
        self.speed = speed
        self.pos = pos
        self.lane = lane
        self.slope = slope

    @property
    def latitude(self) -> float:
        return self.y

    @property
    def longitude(self) -> float:
        return self.x

    def __repr__(self) -> str:
        return (
            f"VehicleTrace(id={self.id}, timestamp={self.timestamp}, "
            f"x={self.x}, y={self.y}, angle={self.angle}, type={self.type!r}, "
            f"speed={self.speed}, pos={self.pos}, lane={self.lane!r}, slope={self.slope})"
        )
