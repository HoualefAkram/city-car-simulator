import xml.etree.ElementTree as ET
from pathlib import Path
from data_models.latlng import LatLng


class TraceParser:

    @staticmethod
    def parse_fcd_trace(
        trace_file: str = "outputs/sumo/trace.xml",
    ) -> list[dict[int, LatLng]]:
        if not Path(trace_file).exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        tree = ET.parse(trace_file)
        root = tree.getroot()

        timesteps = []
        for timestep in root.findall("timestep"):
            snapshot = {}
            for vehicle in timestep.findall("vehicle"):
                veh_id = int(vehicle.get("id"))
                lon = float(vehicle.get("x"))
                lat = float(vehicle.get("y"))
                snapshot[veh_id] = LatLng(lat, lon)
            timesteps.append(snapshot)

        return timesteps
