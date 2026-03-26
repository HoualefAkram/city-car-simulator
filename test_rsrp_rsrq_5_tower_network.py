"""
==============================================================================
 RF Physics Validation: 5-Tower Mixed-Density Network
==============================================================================
 Purpose: Validate that wave_utils.py produces mathematically sound RSRP/RSRQ
          values for training an RL load-balancing agent.

 Tests:
   1. Inter-cell interference lowers RSRQ at cell edges
   2. Cell load (UE congestion) degrades RSRQ for connected UEs
   3. Load drop triggers RSRQ recovery
   4. Spatial isolation — Tower 5 (44km away) is unaffected by urban events

 Topology (real London coords):
   T1 (51.5050, -0.1350)  ─┐
   T2 (51.5050, -0.1300)   ├─ Urban Cluster (~350m apart)
   T3 (51.5075, -0.1325)  ─┘
   T4 (51.5050, -0.1000)  ── Suburban (~2.4km east)
   T5 (51.9000, -0.1350)  ── Rural/Isolated (~44km north)
==============================================================================
"""

import sys
import math
import numpy as np
from data_models.base_tower import BaseTower
from data_models.user_equipment import UserEquipment
from data_models.handover_algorithm import HandoverAlgorithm
from data_models.latlng import LatLng
from utils.wave_utils import WaveUtils
from utils.location_utils import LocationUtils

# ── Tower Topology ──────────────────────────────────────────────────────────

TOWERS_CONFIG = [
    # id, lat, lon, label
    (0, 51.5050, -0.1350, "T1-Urban-Core"),
    (1, 51.5050, -0.1300, "T2-Urban-East"),
    (2, 51.5075, -0.1325, "T3-Urban-North"),
    (3, 51.5050, -0.1000, "T4-Suburban"),
    (4, 51.9000, -0.1350, "T5-Rural-Isolated"),
]

TOWER_LABELS = {cfg[0]: cfg[3] for cfg in TOWERS_CONFIG}


def make_towers() -> list[BaseTower]:
    towers = []
    for tid, lat, lon, label in TOWERS_CONFIG:
        bs = BaseTower(
            id=tid,
            latlng=LatLng(lat=lat, long=lon),
            connected_ues=[],
            p_tx=43.0,
            frequency=3.5e9,
            bandwidth=100e6,
            g_tx=15.0,
            radio="NR",
        )
        towers.append(bs)
    return towers


# ── UE Factories ────────────────────────────────────────────────────────────

def make_urban_ues(n: int, towers: list[BaseTower], start_id: int = 0) -> list[UserEquipment]:
    """Spawn UEs uniformly scattered within the urban triangle."""
    ues = []
    # Centroid of urban cluster
    cx = (51.5050 + 51.5050 + 51.5075) / 3.0
    cy = (-0.1350 + -0.1300 + -0.1325) / 3.0
    rng = np.random.RandomState(42)
    for i in range(n):
        # Random offset ~0-200m in lat/lon
        dlat = rng.uniform(-0.0015, 0.0015)
        dlon = rng.uniform(-0.0015, 0.0015)
        ue = UserEquipment(
            id=start_id + i,
            all_bs=towers,
            latlng=LatLng(lat=cx + dlat, long=cy + dlon),
            handover_algorithm=HandoverAlgorithm.A3_RSRP_3GPP,
        )
        ues.append(ue)
    return ues


def make_rural_ue(towers: list[BaseTower], ue_id: int = 999) -> UserEquipment:
    """One UE sitting ~50m from Tower 5."""
    t5 = towers[4]
    return UserEquipment(
        id=ue_id,
        all_bs=towers,
        latlng=LatLng(lat=t5.latlng.lat + 0.00045, long=t5.latlng.long),  # ~50m north
        handover_algorithm=HandoverAlgorithm.A3_RSRP_3GPP,
    )


# ── Helpers ─────────────────────────────────────────────────────────────────

def raw_rsrp_for_ue(ue: UserEquipment, towers: list[BaseTower]) -> dict[int, float]:
    return {bs.id: WaveUtils.calculate_rsrp(bs=bs, ue=ue) for bs in towers}


def load_factors_for(towers: list[BaseTower]) -> list[float]:
    return [WaveUtils.calculate_load_factor(bs) for bs in towers]


def raw_rsrq_for_ue(ue: UserEquipment, towers: list[BaseTower]) -> dict[int, float]:
    rsrp_map = raw_rsrp_for_ue(ue, towers)
    all_rsrp = list(rsrp_map.values())
    lf = load_factors_for(towers)
    rsrq_map = {}
    for bs in towers:
        rsrq_map[bs.id] = WaveUtils.calculate_rsrq(
            serving_tower=bs,
            serving_rsrp=rsrp_map[bs.id],
            all_rsrp_dBm=all_rsrp,
            load_factors=lf,
        )
    return rsrq_map


def serving_rsrq(ue: UserEquipment, towers: list[BaseTower]) -> float:
    """RSRQ toward the tower with the strongest RSRP."""
    rsrp_map = raw_rsrp_for_ue(ue, towers)
    best_id = max(rsrp_map, key=rsrp_map.get)
    all_rsrp = list(rsrp_map.values())
    lf = load_factors_for(towers)
    best_tower = next(bs for bs in towers if bs.id == best_id)
    return WaveUtils.calculate_rsrq(
        serving_tower=best_tower,
        serving_rsrp=rsrp_map[best_id],
        all_rsrp_dBm=all_rsrp,
        load_factors=lf,
    )


def move_ue_toward(ue: UserEquipment, target: LatLng, fraction: float = 0.1):
    """Move UE a fraction of the way toward a target location."""
    new_lat = ue.latlng.lat + (target.lat - ue.latlng.lat) * fraction
    new_lon = ue.latlng.long + (target.long - ue.latlng.long) * fraction
    ue.latlng = LatLng(lat=new_lat, long=new_lon)


# ── Test 1: Inter-Cell Interference ────────────────────────────────────────

def test_intercell_interference(towers: list[BaseTower]) -> tuple[bool, str]:
    """
    A UE at the cell edge between T1 and T2 should have WORSE RSRQ than
    the same UE measured against T1 alone (no neighbor interference).
    """
    # Place UE exactly at midpoint between T1 and T2
    t1, t2 = towers[0], towers[1]
    mid_lat = (t1.latlng.lat + t2.latlng.lat) / 2
    mid_lon = (t1.latlng.long + t2.latlng.long) / 2

    ue = UserEquipment(
        id=8000,
        all_bs=towers,
        latlng=LatLng(lat=mid_lat, long=mid_lon),
        handover_algorithm=HandoverAlgorithm.A3_RSRP_3GPP,
    )

    # RSRQ with ALL 5 towers contributing to RSSI
    rsrp_all = raw_rsrp_for_ue(ue, towers)
    all_rsrp_list = list(rsrp_all.values())
    rsrq_with_neighbors = WaveUtils.calculate_rsrq(
        serving_tower=t1,
        serving_rsrp=rsrp_all[t1.id],
        all_rsrp_dBm=all_rsrp_list,
    )

    # RSRQ with ONLY T1 (no interference)
    rsrq_isolated = WaveUtils.calculate_rsrq(
        serving_tower=t1,
        serving_rsrp=rsrp_all[t1.id],
        all_rsrp_dBm=[rsrp_all[t1.id]],
    )

    delta = rsrq_isolated - rsrq_with_neighbors
    passed = delta > 0.5  # At least 0.5 dB degradation from neighbor interference

    msg = (
        f"  RSRQ (T1 only): {rsrq_isolated:.2f} dB\n"
        f"  RSRQ (all towers): {rsrq_with_neighbors:.2f} dB\n"
        f"  Interference degradation: {delta:.2f} dB\n"
        f"  {'PASS' if passed else 'FAIL'}: Neighbor towers {'do' if passed else 'do NOT'} degrade RSRQ at cell edge"
    )
    return passed, msg


# ── Test 2: Cell Load Degradation ──────────────────────────────────────────

def test_cell_load_degradation(towers: list[BaseTower]) -> tuple[bool, str]:
    """
    When 30 UEs pile onto T1, RSRQ should degrade compared to 1 UE on T1.

    CRITICAL FINDING: The current wave_utils.py does NOT model cell load in
    RSRQ. RSSI = sum(all_tower_signals) + noise, with no RB utilization factor.
    This test WILL FAIL, exposing the gap for the RL state space.
    """
    t1 = towers[0]

    # Lone UE near T1
    lone_ue = UserEquipment(
        id=9000,
        all_bs=towers,
        latlng=LatLng(lat=t1.latlng.lat + 0.0003, long=t1.latlng.long),
        handover_algorithm=HandoverAlgorithm.A3_RSRP_3GPP,
    )

    # Measure RSRQ with 0 other UEs on T1
    rsrq_empty = serving_rsrq(lone_ue, towers)

    # Now pile 30 UEs onto T1 (register them as connected)
    crowd = make_urban_ues(30, towers, start_id=9100)
    for c in crowd:
        c.latlng = LatLng(
            lat=t1.latlng.lat + np.random.uniform(-0.0002, 0.0002),
            long=t1.latlng.long + np.random.uniform(-0.0002, 0.0002),
        )
        t1.add_ue(c)
        c.serving_bs = t1

    # Re-measure RSRQ for the lone UE (same position)
    rsrq_loaded = serving_rsrq(lone_ue, towers)

    delta = rsrq_empty - rsrq_loaded

    # Clean up
    for c in crowd:
        t1.remove_ue(c.id)

    passed = delta > 0.1  # Any measurable degradation from load

    msg = (
        f"  RSRQ (T1 empty): {rsrq_empty:.4f} dB\n"
        f"  RSRQ (T1 loaded, 30 UEs): {rsrq_loaded:.4f} dB\n"
        f"  Load degradation delta: {delta:.4f} dB\n"
        f"  {'PASS' if passed else 'FAIL'}: Cell load {'does' if passed else 'does NOT'} impact RSRQ"
    )
    return passed, msg


# ── Test 3: Load Recovery ──────────────────────────────────────────────────

def test_load_recovery(towers: list[BaseTower]) -> tuple[bool, str]:
    """
    After removing 25 of 30 UEs from T1, RSRQ should recover.
    Same as Test 2 — this test validates the inverse direction.
    """
    t1 = towers[0]

    probe_ue = UserEquipment(
        id=9200,
        all_bs=towers,
        latlng=LatLng(lat=t1.latlng.lat + 0.0003, long=t1.latlng.long),
        handover_algorithm=HandoverAlgorithm.A3_RSRP_3GPP,
    )

    # Pile 30 UEs onto T1
    crowd = make_urban_ues(30, towers, start_id=9300)
    for c in crowd:
        c.latlng = LatLng(
            lat=t1.latlng.lat + np.random.uniform(-0.0002, 0.0002),
            long=t1.latlng.long + np.random.uniform(-0.0002, 0.0002),
        )
        t1.add_ue(c)
        c.serving_bs = t1

    rsrq_loaded = serving_rsrq(probe_ue, towers)

    # Remove 25 UEs
    for c in crowd[:25]:
        t1.remove_ue(c.id)

    rsrq_after_drop = serving_rsrq(probe_ue, towers)

    # Clean up remaining
    for c in crowd[25:]:
        t1.remove_ue(c.id)

    recovery = rsrq_after_drop - rsrq_loaded
    passed = recovery > 0.1

    msg = (
        f"  RSRQ (30 UEs on T1): {rsrq_loaded:.4f} dB\n"
        f"  RSRQ (5 UEs on T1):  {rsrq_after_drop:.4f} dB\n"
        f"  Recovery delta: {recovery:.4f} dB\n"
        f"  {'PASS' if passed else 'FAIL'}: Load drop {'triggers' if passed else 'does NOT trigger'} RSRQ recovery"
    )
    return passed, msg


# ── Test 4: Spatial Isolation (Tower 5 Test) ───────────────────────────────

def test_spatial_isolation(towers: list[BaseTower]) -> tuple[bool, str]:
    """
    The rural UE at Tower 5 (~44km away) must see ZERO change in RSRP/RSRQ
    regardless of what happens in the urban cluster.
    """
    rural_ue = make_rural_ue(towers)
    t5 = towers[4]

    # ── Baseline: measure raw RSRP and RSRQ at T5 ──
    rsrp_baseline = raw_rsrp_for_ue(rural_ue, towers)
    rsrq_baseline = raw_rsrq_for_ue(rural_ue, towers)

    baseline_rsrp_t5 = rsrp_baseline[t5.id]
    baseline_rsrq_t5 = rsrq_baseline[t5.id]

    # ── Scenario A: Pile 30 UEs onto T1 (should not change T5 measurement) ──
    t1 = towers[0]
    crowd_a = make_urban_ues(30, towers, start_id=7000)
    for c in crowd_a:
        c.latlng = LatLng(lat=t1.latlng.lat + 0.0001, long=t1.latlng.long)
        t1.add_ue(c)
        c.serving_bs = t1

    rsrp_after_load = raw_rsrp_for_ue(rural_ue, towers)
    rsrq_after_load = raw_rsrq_for_ue(rural_ue, towers)

    rsrp_delta = abs(rsrp_after_load[t5.id] - baseline_rsrp_t5)
    rsrq_delta = abs(rsrq_after_load[t5.id] - baseline_rsrq_t5)

    # Clean up
    for c in crowd_a:
        t1.remove_ue(c.id)

    # ── Scenario B: Move urban UEs around (handovers) — still no T5 impact ──
    urban_ues = make_urban_ues(10, towers, start_id=7500)
    rsrq_readings_t5 = []
    for step in range(20):
        for ue in urban_ues:
            # Random walk within urban area
            ue.latlng = LatLng(
                lat=51.505 + np.random.uniform(-0.002, 0.002),
                long=-0.133 + np.random.uniform(-0.002, 0.002),
            )
            ue.move_to(ue.latlng, timestep=float(step))
        # Measure rural UE
        rsrq_readings_t5.append(raw_rsrq_for_ue(rural_ue, towers)[t5.id])

    # Check that T5 RSRQ didn't change across all timesteps
    t5_variance = max(rsrq_readings_t5) - min(rsrq_readings_t5)

    # Threshold: 0.001 dB — 500x below the 3GPP RSRQ quantization step (0.5 dB).
    # Any leakage below this is physically negligible and invisible to the RL agent.
    ISOLATION_THRESHOLD = 1e-3

    passed_rsrp = rsrp_delta < ISOLATION_THRESHOLD
    passed_rsrq = rsrq_delta < ISOLATION_THRESHOLD
    passed_variance = t5_variance < ISOLATION_THRESHOLD
    passed = passed_rsrp and passed_rsrq and passed_variance

    msg = (
        f"  T5 baseline RSRP: {baseline_rsrp_t5:.6f} dBm\n"
        f"  T5 baseline RSRQ: {baseline_rsrq_t5:.6f} dB\n"
        f"  RSRP delta after urban load: {rsrp_delta:.2e} (threshold: <{ISOLATION_THRESHOLD})\n"
        f"  RSRQ delta after urban load: {rsrq_delta:.2e} (threshold: <{ISOLATION_THRESHOLD})\n"
        f"  T5 RSRQ variance across 20 urban movement steps: {t5_variance:.2e}\n"
        f"  3GPP quantization step: 0.5 dB (any leakage below this is invisible to RL)\n"
        f"  {'PASS' if passed else 'FAIL'}: Tower 5 is {'effectively' if passed else 'NOT'} isolated from urban events"
    )
    return passed, msg


# ── Multi-Timestep Simulation with Tabular Log ─────────────────────────────

def run_simulation(towers: list[BaseTower]):
    """
    150-step simulation:
      Steps 1-50:   Urban UEs roam between T1/T2/T3
      Steps 50-100: Traffic jam — all 30 UEs pile under T1
      Steps 100-150: 25 UEs removed, only 5 remain
    """
    NUM_URBAN = 30
    urban_ues = make_urban_ues(NUM_URBAN, towers, start_id=100)
    rural_ue = make_rural_ue(towers)

    # Initial connection
    for ue in urban_ues + [rural_ue]:
        ue.move_to(ue.latlng, timestep=0.0)

    t1 = towers[0]
    t5 = towers[4]
    rng = np.random.RandomState(7)

    # Log storage
    log_rows = []

    def log_step(step: int, active_urban: list[UserEquipment]):
        # T1 load = number of UEs connected to T1
        t1_load = len(t1.connected_ues)
        t5_load = len(t5.connected_ues)

        # Average RSRQ toward serving tower for urban UEs on T1
        t1_rsrqs = []
        for ue in active_urban:
            if ue.serving_bs and ue.serving_bs.id == t1.id:
                rsrq_map = raw_rsrq_for_ue(ue, towers)
                t1_rsrqs.append(rsrq_map[t1.id])

        avg_t1_rsrq = np.mean(t1_rsrqs) if t1_rsrqs else float("nan")

        # Rural UE RSRQ at T5
        t5_rsrq = raw_rsrq_for_ue(rural_ue, towers)[t5.id]

        log_rows.append({
            "step": step,
            "t1_load": t1_load,
            "avg_t1_rsrq": avg_t1_rsrq,
            "t5_load": t5_load,
            "t5_rsrq": t5_rsrq,
        })

    active_urban = list(urban_ues)

    # ── Phase 1: Roaming (steps 1-50) ──
    for step in range(1, 51):
        for ue in active_urban:
            # Random walk within urban triangle
            target_tower = towers[rng.randint(0, 3)]
            move_ue_toward(ue, target_tower.latlng, fraction=0.15)
            ue.move_to(ue.latlng, timestep=float(step))
        rural_ue.move_to(rural_ue.latlng, timestep=float(step))
        if step % 10 == 0 or step == 1:
            log_step(step, active_urban)

    # ── Phase 2: Traffic jam under T1 (steps 51-100) ──
    for step in range(51, 101):
        for ue in active_urban:
            # Push all UEs toward T1
            move_ue_toward(ue, t1.latlng, fraction=0.3)
            ue.move_to(ue.latlng, timestep=float(step))
        rural_ue.move_to(rural_ue.latlng, timestep=float(step))
        if step % 10 == 0 or step == 51:
            log_step(step, active_urban)

    # ── Phase 3: Drop 25 UEs (steps 101-150) ──
    removed = active_urban[:25]
    active_urban = active_urban[25:]
    for ue in removed:
        if ue.serving_bs:
            ue.serving_bs.remove_ue(ue.id)

    for step in range(101, 151):
        for ue in active_urban:
            # Slight movement near T1
            move_ue_toward(ue, t1.latlng, fraction=0.05)
            ue.move_to(ue.latlng, timestep=float(step))
        rural_ue.move_to(rural_ue.latlng, timestep=float(step))
        if step % 10 == 0 or step == 101:
            log_step(step, active_urban)

    return log_rows


def print_table(rows: list[dict]):
    header = f"{'Step':>6} | {'T1 Load':>8} | {'Avg T1 RSRQ':>12} | {'T5 Load':>8} | {'T5 RSRQ':>10} | {'Isolation'}"
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        t5_rsrq_str = f"{r['t5_rsrq']:.4f}"
        t1_rsrq_str = f"{r['avg_t1_rsrq']:.4f}" if not np.isnan(r["avg_t1_rsrq"]) else "N/A"
        # Isolation check: T5 RSRQ should match first row
        iso = "OK" if rows and abs(r["t5_rsrq"] - rows[0]["t5_rsrq"]) < 1e-6 else "DRIFT!"
        print(
            f"{r['step']:>6} | {r['t1_load']:>8} | {t1_rsrq_str:>12} | {r['t5_load']:>8} | {t5_rsrq_str:>10} | {iso}"
        )
    print(sep)


# ── Validation Summary ─────────────────────────────────────────────────────

def validate_all() -> bool:
    print("=" * 72)
    print(" RF PHYSICS VALIDATION: 5-Tower Mixed-Density Network")
    print("=" * 72)

    towers = make_towers()

    # Print topology
    print("\n[Topology]")
    for bs in towers:
        print(f"  {TOWER_LABELS[bs.id]:20s} @ ({bs.latlng.lat:.4f}, {bs.latlng.long:.4f})")

    # Print inter-tower distances
    print("\n[Inter-Tower Distances]")
    for i in range(len(towers)):
        for j in range(i + 1, len(towers)):
            d = LocationUtils.haversine(towers[i].latlng, towers[j].latlng)
            print(f"  {TOWER_LABELS[towers[i].id]} <-> {TOWER_LABELS[towers[j].id]}: {d:.0f} m")

    all_passed = True
    warnings = []

    # ── Test 1 ──
    print("\n" + "-" * 72)
    print("[TEST 1] Inter-Cell Interference at Cell Edge")
    print("-" * 72)
    p, msg = test_intercell_interference(towers)
    print(msg)
    if not p:
        warnings.append("TEST 1 FAILED: Neighbor interference not reflected in RSRQ")
    all_passed &= p

    # ── Test 2 ──
    # Reset tower state
    towers = make_towers()
    print("\n" + "-" * 72)
    print("[TEST 2] Cell Load Degradation (30 UEs on T1)")
    print("-" * 72)
    p, msg = test_cell_load_degradation(towers)
    print(msg)
    if not p:
        warnings.append(
            "TEST 2 FAILED: Cell load has NO effect on RSRQ. "
            "wave_utils.calculate_rssi() does not factor in RB utilization. "
            "The RL agent CANNOT observe congestion through RSRQ alone."
        )
    all_passed &= p

    # ── Test 3 ──
    towers = make_towers()
    print("\n" + "-" * 72)
    print("[TEST 3] Load Recovery (30 -> 5 UEs on T1)")
    print("-" * 72)
    p, msg = test_load_recovery(towers)
    print(msg)
    if not p:
        warnings.append(
            "TEST 3 FAILED: Load recovery not observable in RSRQ. "
            "Same root cause as Test 2 — RSSI ignores cell load."
        )
    all_passed &= p

    # ── Test 4 ──
    towers = make_towers()
    print("\n" + "-" * 72)
    print("[TEST 4] Spatial Isolation (Tower 5 @ 44km)")
    print("-" * 72)
    p, msg = test_spatial_isolation(towers)
    print(msg)
    if not p:
        warnings.append(
            "TEST 4 FAILED: Urban events are leaking into Tower 5's RSRQ! "
            "Check if calculate_rssi sums signals from ALL towers globally."
        )
    all_passed &= p

    # ── Simulation Table ──
    towers = make_towers()
    print("\n" + "-" * 72)
    print("[SIMULATION] 150-Step Multi-Phase Run")
    print("-" * 72)
    rows = run_simulation(towers)
    print_table(rows)

    # ── Final Verdict ──
    print("\n" + "=" * 72)
    if all_passed:
        print(" ALL TESTS PASSED — RF physics are sound for RL training.")
    else:
        print(f" {len(warnings)} TEST(S) FAILED — RL state space has issues:")
        for w in warnings:
            print(f"   WARNING: {w}")
    print("=" * 72)

    return all_passed


if __name__ == "__main__":
    ok = validate_all()
    sys.exit(0 if ok else 1)
