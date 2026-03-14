# 🚗 City Car Simulator — 5G Handover Simulation

A Python-based simulator that models a **User Equipment (UE)** — a car — moving through a city with multiple **Base Station Towers (BS)**. The simulator calculates real-time **RSRP** and **RSRQ** signal metrics using standard radio propagation models, laying the foundation for an **AI Reinforcement Learning agent** to optimize handover decisions.

---

## 📡 What is Handover?

In mobile networks, as a car moves through a city, it constantly measures signal strength from nearby base stations. When a neighboring BS provides a stronger signal than the current serving BS, the network triggers a **handover** — switching the UE's connection to the better tower.

This simulator models that process from first principles.

---

## 🗂️ Project Structure

```
city-car-simulator/
│
├── main.py               # Entry point — sets up BS, UE, runs simulation
├── base_tower.py         # BaseTower class (BS)
├── user_equipment.py     # UserEquipment class (UE / car)
├── wave_utils.py         # RSRP, RSRQ, RSSI calculations
├── location_utils.py     # Haversine distance, move_meters
├── latlng.py             # LatLng coordinate dataclass
└── ng_ran_report.py      # NGRANReport — measurement report from UE to BS
```

---

## ⚙️ Signal Model

### Path Loss (Log-Distance Model)

```
PL(d) = PL(d0) + 10·n·log10(d/d0)
```

| Parameter | Value | Description |
|---|---|---|
| `d0` | 1m | Reference distance |
| `n` (LOS) | 2.0 | Path loss exponent, clear line of sight |
| `n` (NLOS) | 3.0 | Path loss exponent, urban obstructions |
| LOS threshold | 20m | Distance under which LOS is assumed |

### RSRP

```
RSRP (dBm) = P_tx + G_tx + G_rx - PL(d)
```

| Parameter | Value | Description |
|---|---|---|
| `P_tx` | 43 dBm | BS transmit power (20W, typical 5G macro) |
| `G_tx` | 15 dBi | BS sector antenna gain |
| `G_rx` | 0 dBi | UE omnidirectional antenna gain |

### RSRQ

```
RSRQ (dB) = 10·log10(N) + RSRP - RSSI
```

| Parameter | Value | Description |
|---|---|---|
| `N` | 100 | Resource blocks (20 MHz bandwidth) |
| `RSSI` | Σ all BS signals + noise | Total received power |

### Thermal Noise Floor

```
noise (dBm) = -174 + 10·log10(bandwidth_hz) + noise_figure_db
```

| Parameter | Value |
|---|---|
| Bandwidth | 100 MHz (5G sub-6GHz) |
| Noise figure | 7 dB (typical UE) |
| Noise floor | ~-87 dBm |

---

## 📶 Signal Quality Reference

### RSRP
| Value | Quality |
|---|---|
| > -60 dBm | Excellent ✅ |
| -60 to -80 dBm | Good |
| -80 to -90 dBm | Medium |
| -90 to -100 dBm | Poor |
| < -100 dBm | Very bad ❌ |

### RSRQ
| Value | Quality |
|---|---|
| > -3 dB | Excellent ✅ |
| -3 to -10 dB | Good |
| -10 to -15 dB | Medium |
| < -20 dB | Poor ❌ |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/your-username/city-car-simulator.git
cd city-car-simulator
pip install numpy folium
```

### Run

```bash
python3 main.py
```

### Example Output

```
distance before move: 319.83m
rsrp1: -60.48 dBm
distance after move: 206.17m
rsrp2: -54.76 dBm
```

---

## 🗺️ Visualization

The simulator uses **Folium** to render an interactive HTML map showing BS towers, UE position, and travel path.

```python
plot_simulation(bs_list=[bs1, bs2, bs3], ue=car, ue_path=path)
# opens simulation.html in browser
```

---

## 🏙️ BS & UE Configuration

```python
bs1 = BaseTower(
    id=0,
    latlng=LatLng(36.7538, 3.0588),  # Algiers, Algeria
    p_tx=43.0,                        # dBm
    frequency=3.5e9,                  # 5G sub-6GHz
    bandwidth=100e6,                  # 100 MHz
    g_tx=15.0,                        # dBi
)

car = UserEquipment(
    id=0,
    latlng=LatLng(36.7520, 3.0560),
    g_rx=0.0,                         # dBi
    serving_bs=bs1,
)
```

---

## SUMO commands
netconvert --osm-files home_map.osm --output-file home_map.net.xml</br>
randomTrips.py -n home_map.net.xml -e 1000 -o home_trips.xml</br>
duarouter -n home_map.net.xml --route-files home_trips.xml -o home.rou.xml --ignore-errors</br>

---

## 🤖 Roadmap

- [x] Log-distance path loss model
- [x] RSRP calculation (per BS)
- [x] RSRQ / RSSI calculation
- [x] UE movement (haversine-based)
- [x] Map visualization (Folium)
- [x] A3 handover event trigger (hysteresis-based)
- [x] Multiple UEs
- [ ] Performance metrics (ping-pong rate, handover failures)
- [ ] Shadowing / slow fading (Long finger)

---

## 📐 Dependencies

| Library | Purpose |
|---|---|
| `numpy` | Math / signal calculations |
| `folium` | Interactive map visualization |

---

## 📚 References

- 3GPP TR 38.901 — Study on channel model for frequencies from 0.5 to 100 GHz
- 3GPP TS 36.214 — LTE physical layer measurements (RSRP, RSRQ definitions)
- Rappaport, T.S. — *Wireless Communications: Principles and Practice*

---

## 📄 License

MIT License — free to use, modify, and distribute.