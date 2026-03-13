import requests
from pathlib import Path
from data_models.latlng import LatLng


class MapDownloader:

    @staticmethod
    def download_osm_by_bbox(
        top_left: LatLng,
        bottom_right: LatLng,
        output_file: str = "maps/map.osm",
    ) -> None:
        if Path(output_file).exists():
            print("Skipping download, file already exists")
            return
        # left, bottom, right, top (min_lon, min_lat, max_lon, max_lat)
        min_lon = top_left.long
        min_lat = bottom_right.lat
        max_lon = bottom_right.long
        max_lat = top_left.lat
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
