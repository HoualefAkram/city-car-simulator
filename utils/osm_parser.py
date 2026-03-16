import xml.etree.ElementTree as ET


class OsmParser:

    @staticmethod
    def parse_bounds(osm_file: str) -> dict[str, float]:
        tree = ET.parse(osm_file)
        root = tree.getroot()
        bounds = root.find("bounds")
        return {
            "minlon": float(bounds.get("minlon")),
            "minlat": float(bounds.get("minlat")),
            "maxlon": float(bounds.get("maxlon")),
            "maxlat": float(bounds.get("maxlat")),
        }
