import folium
from base_tower import BaseTower
from user_equipment import UserEquipment


class Render:
    @staticmethod
    def render_map(
        bs_list: list[BaseTower], ue: UserEquipment, output: str = "simulation.html"
    ):
        m = folium.Map(
            location=[bs_list[0].latlng.lat, bs_list[0].latlng.long], zoom_start=16
        )

        # Base Towers
        for bs in bs_list:
            folium.Marker(
                location=[bs.latlng.lat, bs.latlng.long],
                tooltip=f"BS {bs.id}",
                popup=f"BS{bs.id} | {bs.frequency/1e9:.1f}GHz | {bs.p_tx}dBm",
                icon=folium.Icon(color="red", icon="tower-cell", prefix="fa"),
            ).add_to(m)

        # UE path
        if len(ue.path_history) > 1:
            folium.PolyLine(
                locations=[[p.lat, p.long] for p in ue.path_history],
                color="blue",
                weight=2,
            ).add_to(m)

        # UE initial position
        folium.Marker(
            location=[ue.path_history[0].lat, ue.path_history[0].long],
            tooltip=f"UE {ue.id}",
            icon=folium.Icon(color="green", icon="car", prefix="fa"),
        ).add_to(m)

        # UE current position
        folium.Marker(
            location=[ue.latlng.lat, ue.latlng.long],
            tooltip=f"UE {ue.id}",
            icon=folium.Icon(color="red", icon="car", prefix="fa"),
        ).add_to(m)

        m.save(output)
