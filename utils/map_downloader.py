import requests
from pathlib import Path
from data_models.latlng import LatLng
import xml.etree.ElementTree as ET
import os


class MapDownloader:

    @staticmethod
    def __is_identical(val1: float, val2: float, tolerance: float = 0.000001) -> bool:
        return abs(float(val1) - float(val2)) <= tolerance

    @staticmethod
    def download_osm_by_bbox(
        top_left: LatLng,
        bottom_right: LatLng,
        output_file: str = "maps/map.osm",
    ) -> None:
        # left, bottom, right, top (min_lon, min_lat, max_lon, max_lat)
        min_lon = top_left.long
        min_lat = bottom_right.lat
        max_lon = bottom_right.long
        max_lat = top_left.lat

        # check if the map already exists
        if Path(output_file).exists():
            tree = ET.parse(output_file)
            root = tree.getroot()
            bounds = root.find("bounds")
            osm_minlon = bounds.get("minlon", False)
            osm_minlat = bounds.get("minlat", False)
            osm_maxlon = bounds.get("maxlon", False)
            osm_maxlat = bounds.get("maxlat", False)
            if (
                MapDownloader.__is_identical(min_lon, osm_minlon)
                and MapDownloader.__is_identical(min_lat, osm_minlat)
                and MapDownloader.__is_identical(max_lon, osm_maxlon)
                and MapDownloader.__is_identical(max_lat, osm_maxlat)
            ):
                print("Map already downloaded. Skipping...")
                return None

            print(
                f"Removing old map. missmatch map:\n{min_lon}:{osm_minlon}\n{min_lat}:{osm_minlat}\n{max_lon}:{osm_maxlon}\n{max_lat}:{osm_maxlat}"
            )
            os.remove(output_file)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        url = f"https://overpass-api.de/api/map?bbox={min_lon},{min_lat},{max_lon},{max_lat}"
        print(f"Downloading OSM data from Overpass API to {output_file}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_file, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print("Download complete!")
        else:
            raise ConnectionError(
                f"Failed to download map. HTTP Status: {response.status_code}\n"
                f"Response: {response.text}"
            )
