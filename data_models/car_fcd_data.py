from data_models.latlng import LatLng


class CarFcdData:
    def __init__(
        self,
        car_id: int,
        latlng: LatLng,
        angle: float,
        speed: float,
        timestep: float,
    ):
        self.car_id = car_id
        self.latlng = latlng
        self.timestep = timestep
        self.angle = angle
        self.speed = speed
