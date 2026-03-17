from data_models.latlng import LatLng


class CarFcdData:
    def __init__(self, id: int, latlng: LatLng, timestep: float):
        self.id = id
        self.latlng = latlng
        self.timestep = timestep
