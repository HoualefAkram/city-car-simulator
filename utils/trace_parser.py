import xml.etree.ElementTree as ET
from pathlib import Path
from data_models.latlng import LatLng


class TraceParser:

    @staticmethod
    def parse_fcd_trace(
        trace_file: int = "outputs/sumo/trace.xml",
    ) -> dict[int, list[LatLng]]:
        if not Path(trace_file).exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        tree = ET.parse(trace_file)
        root = tree.getroot()

        vehicle_paths = {}
        for timestep in root.findall("timestep"):
            for vehicle in timestep.findall("vehicle"):
                veh_id = int(vehicle.get("id"))

                lon = float(vehicle.get("x"))
                lat = float(vehicle.get("y"))

                if veh_id not in vehicle_paths:
                    vehicle_paths[veh_id] = []

                vehicle_paths[veh_id].append(LatLng(lat, lon))

        return vehicle_paths
