from data_models.user_equipment import UserEquipment
from data_models.base_tower import BaseTower
from data_models.latlng import LatLng
from utils.path_gen import PathGeneration
from utils.render import Render


bs1 = BaseTower(
    id=1,
    latlng=LatLng(35.717583, -0.540996),  # top left
    connected_ues=[],
)


bs2 = BaseTower(
    id=2,
    latlng=LatLng(35.717558, -0.539236),  # top right
    connected_ues=[],
)

bs3 = BaseTower(
    id=3,
    latlng=LatLng(35.717580, -0.540077),  # center top
    connected_ues=[],
)


car = UserEquipment(
    id=0,
    latlng=LatLng(35.717122, -0.540052),  # Home
    serving_bs=bs2,  # starts connected to bs2
    all_bs=[bs1, bs2, bs3],
    print_report_on_movement=True,
)

bs2.add_ue(ue=car)

path_gen = PathGeneration(stop_trip_generation_after=1)
path_gen.run()
