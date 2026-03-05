from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from latlng import LatLng
from ng_ran_report import NGRANReport

if TYPE_CHECKING:
    from user_equipment import UserEquipment


class BaseTower:

    def __init__(
        self,
        id: int,
        latlng: LatLng,
        connected_ues: list[UserEquipment],
        p_tx: float,
        frequency: float,  # 4G LTE (common)2100 MHz, 4G LTE (low band)800 MHz, 5G sub-6GHz 3500 MHz, 5G mmWave 28 GHz
        g_tx: float,  # +14 to +17 dBi
        ng_ran_report: Optional[NGRANReport] = None,
    ):
        self.id = id
        self.latlng: LatLng = latlng
        self.connected_ues: list[UserEquipment] = connected_ues
        self.last_report: NGRANReport = ng_ran_report
        self.p_tx = p_tx
        self.g_tx = g_tx
        self.frequency = frequency

    def receive_report(self, report: NGRANReport):
        self.last_report = report
