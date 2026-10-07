from __future__ import annotations

"""Mechanics-based stacked wood shear-wall deflection engine.

Basis
-----
Implements the FPInnovations/CWC mechanics-based stacked shear-wall formulation:
  Δ_i = Δ_b,i + Δ_s,i + Δ_n,i + Δ_a,i + Δ_r,i
with continuous-rod transformed bending stiffness.

The material/connector lookup values in this module are transcribed from the user's
validated shearwallanalysis_R2_faster.py database. The calculation uses service-level
forces supplied by the app; no strength load factors are applied internally.

Storey order
------------
Input DataFrames are ordered BOTTOM -> TOP (Storey 1 is the lowest storey).
"""

from typing import Dict, Tuple
import math
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Embedded databases transcribed from shearwallanalysis_R2_faster.py
# -----------------------------------------------------------------------------
SHEATHING_BV = pd.DataFrame([
    {"Panel Type": "DFP", "Thickness (mm)": 9.5,  "Bv (N/mm)": 5500},
    {"Panel Type": "DFP", "Thickness (mm)": 12.5, "Bv (N/mm)": 6900},
    {"Panel Type": "DFP", "Thickness (mm)": 15.5, "Bv (N/mm)": 8400},
    {"Panel Type": "DFP", "Thickness (mm)": 18.5, "Bv (N/mm)": 9800},
    {"Panel Type": "CSP", "Thickness (mm)": 9.5,  "Bv (N/mm)": 4300},
    {"Panel Type": "CSP", "Thickness (mm)": 12.5, "Bv (N/mm)": 5700},
    {"Panel Type": "CSP", "Thickness (mm)": 15.5, "Bv (N/mm)": 7100},
    {"Panel Type": "CSP", "Thickness (mm)": 18.5, "Bv (N/mm)": 8600},
    {"Panel Type": "OSB", "Thickness (mm)": 9.5,  "Bv (N/mm)": 10000},
    {"Panel Type": "OSB", "Thickness (mm)": 11.0, "Bv (N/mm)": 11000},
    {"Panel Type": "OSB", "Thickness (mm)": 12.0, "Bv (N/mm)": 11000},
    {"Panel Type": "OSB", "Thickness (mm)": 15.0, "Bv (N/mm)": 11000},
    {"Panel Type": "OSB", "Thickness (mm)": 18.0, "Bv (N/mm)": 12000},
])

LUMBER_E = pd.DataFrame([
    {"Species": "D.Fir-L", "Grade": "S.S.",       "E parallel (N/mm²)": 12500, "E perp (N/mm²)": 625},
    {"Species": "D.Fir-L", "Grade": "No.1/No.2", "E parallel (N/mm²)": 11000, "E perp (N/mm²)": 550},
    {"Species": "D.Fir-L", "Grade": "No.3/Stud", "E parallel (N/mm²)": 10000, "E perp (N/mm²)": 500},
    {"Species": "Hem-Fir", "Grade": "S.S.",       "E parallel (N/mm²)": 12000, "E perp (N/mm²)": 600},
    {"Species": "Hem-Fir", "Grade": "No.1/No.2", "E parallel (N/mm²)": 11000, "E perp (N/mm²)": 550},
    {"Species": "Hem-Fir", "Grade": "No.3/Stud", "E parallel (N/mm²)": 10000, "E perp (N/mm²)": 500},
    {"Species": "S-P-F",   "Grade": "S.S.",       "E parallel (N/mm²)": 10500, "E perp (N/mm²)": 525},
    {"Species": "S-P-F",   "Grade": "No.1/No.2", "E parallel (N/mm²)": 9500,  "E perp (N/mm²)": 475},
    {"Species": "S-P-F",   "Grade": "No.3/Stud", "E parallel (N/mm²)": 9000,  "E perp (N/mm²)": 450},
    {"Species": "Northern", "Grade": "S.S.",       "E parallel (N/mm²)": 7500,  "E perp (N/mm²)": 375},
    {"Species": "Northern", "Grade": "No.1/No.2", "E parallel (N/mm²)": 7000,  "E perp (N/mm²)": 350},
    {"Species": "Northern", "Grade": "No.3/Stud", "E parallel (N/mm²)": 6500,  "E perp (N/mm²)": 325},
])

TAKEUP_DEVICES = pd.DataFrame({
    "Model No.": ['ATUD6-2', 'ATUD9', 'ATUD9-2', 'ATUD9-3', 'ATUD14', 'TUD10',
                  'RTUD4B', 'RTUD5', 'RTUD6', 'RTUD7', 'RTUD8',
                  'CTUD55', 'CTUD65', 'CTUD66', 'CTUD75', 'CTUD76', 'CTUD77',
                  'CTUD87', 'CTUD88', 'CTUD97', 'CTUD98', 'CTUD99'],
    "Diameter (in.)": [0.750, 1.125, 1.125, 1.125, 1.750, 1.250,
                       0.500, 0.625, 0.750, 0.875, 1.000,
                       0.625, 0.750, 0.750, 0.875, 0.875, 0.875,
                       1.000, 1.000, 1.125, 1.125, 1.125],
    "Seating Increment DR (in.)": [0.004, 0.002, 0.002, 0.002, 0.005, 0.001,
                                    0.048, 0.056, 0.057, 0.059, 0.066,
                                    0.004, 0.003, 0.003, 0.003, 0.003, 0.003,
                                    0.003, 0.003, 0.003, 0.003, 0.003],
    "Deflection at Factored Resistance DF (in.)": [0.025, 0.014, 0.044, 0.040, 0.017, 0.033,
                                                    0.012, 0.007, 0.011, 0.013, 0.037,
                                                    0.008, 0.070, 0.070, 0.070, 0.070, 0.070,
                                                    0.038, 0.038, 0.038, 0.038, 0.038],
    "Seating Increment DR (mm)": [0.102, 0.051, 0.051, 0.051, 0.127, 0.025,
                                  1.219, 1.422, 1.448, 1.499, 1.676,
                                  0.102, 0.076, 0.076, 0.076, 0.076, 0.076,
                                  0.076, 0.076, 0.076, 0.076, 0.076],
    "Deflection at Factored Resistance DF (mm)": [0.635, 0.356, 1.118, 1.016, 0.432, 0.838,
                                                  0.305, 0.178, 0.279, 0.330, 0.940,
                                                  0.203, 1.778, 1.778, 1.778, 1.778, 1.778,
                                                  0.965, 0.965, 0.965, 0.965, 0.965],
})

ROD_RESISTANCE = pd.DataFrame({
    "Model no.": ['SR4', 'SR5', 'SR6', 'SR7', 'SR8', 'SR9', 'SR10',
                  'SR4H', 'SR5H', 'SR6H', 'SR7H', 'SR8H', 'SR9H',
                  'SR10H', 'SR9H150', 'SR10H150'],
    "Rod diameter (in.)": [0.5, 0.625, 0.75, 0.875, 1, 1.125, 1.25,
                           0.5, 0.625, 0.75, 0.875, 1, 1.125,
                           1.25, 1.125, 1.25],
    "Tr (kN)": [26.33, 42.45, 62.61, 85.79, 112.12, 141.96, 175.35,
                56.33, 90.6, 133.9, 183.54, 239.87, 303.74,
                375.14, 382.36, 485.59],
})

ROD_GEOMETRY = pd.DataFrame([
    {"Rod Diameter": '1/2"',   "Strong Rod Standard": "SR4",  "Strong Rod High Strength": "SR4H",  "Ag (mm²)": 98.7,   "Ane (mm²)": 91.6,   "Tr Standard (kN)": 26,  "Tr High (kN)": 56},
    {"Rod Diameter": '5/8"',   "Strong Rod Standard": "SR5",  "Strong Rod High Strength": "SR5H",  "Ag (mm²)": 158.7,  "Ane (mm²)": 145.8,  "Tr Standard (kN)": 43,  "Tr High (kN)": 91},
    {"Rod Diameter": '3/4"',   "Strong Rod Standard": "SR6",  "Strong Rod High Strength": "SR6H",  "Ag (mm²)": 234.2,  "Ane (mm²)": 215.5,  "Tr Standard (kN)": 63,  "Tr High (kN)": 134},
    {"Rod Diameter": '7/8"',   "Strong Rod Standard": "SR7",  "Strong Rod High Strength": "SR7H",  "Ag (mm²)": 321.3,  "Ane (mm²)": 298.1,  "Tr Standard (kN)": 86,  "Tr High (kN)": 184},
    {"Rod Diameter": '1"',     "Strong Rod Standard": "SR8",  "Strong Rod High Strength": "SR8H",  "Ag (mm²)": 419.4,  "Ane (mm²)": 391.0,  "Tr Standard (kN)": 112, "Tr High (kN)": 240},
    {"Rod Diameter": '1 1/8"', "Strong Rod Standard": "SR9",  "Strong Rod High Strength": "SR9H",  "Ag (mm²)": 531.6,  "Ane (mm²)": 492.3,  "Tr Standard (kN)": 142, "Tr High (kN)": 304},
    {"Rod Diameter": '1 1/4"', "Strong Rod Standard": "SR10", "Strong Rod High Strength": "SR10H", "Ag (mm²)": 670.3,  "Ane (mm²)": 625.2,  "Tr Standard (kN)": 179, "Tr High (kN)": 383},
    {"Rod Diameter": '1 3/8"', "Strong Rod Standard": "SR11", "Strong Rod High Strength": "SR11H", "Ag (mm²)": 811.0,  "Ane (mm²)": 745.2,  "Tr Standard (kN)": 217, "Tr High (kN)": 464},
    {"Rod Diameter": '1 1/2"', "Strong Rod Standard": "SR12", "Strong Rod High Strength": "SR12H", "Ag (mm²)": 981.9,  "Ane (mm²)": 906.4,  "Tr Standard (kN)": 262, "Tr High (kN)": 561},
    {"Rod Diameter": '1 3/4"', "Strong Rod Standard": "SR14", "Strong Rod High Strength": "SR14H", "Ag (mm²)": 1332.9, "Ane (mm²)": 1225.8, "Tr Standard (kN)": 356, "Tr High (kN)": 762},
    {"Rod Diameter": '2"',     "Strong Rod Standard": "SR16", "Strong Rod High Strength": "SR16H", "Ag (mm²)": 1810.3, "Ane (mm²)": 1611.6, "Tr Standard (kN)": 483, "Tr High (kN)": 1033},
])

SHEAR_CLIPS = pd.DataFrame([
    {"Stud Species": "D.Fir-L", "A35": 4.25, "LTP4": 3.63, "LTP5": 3.85, "LS70": 4.60, "LS90": 5.52},
    {"Stud Species": "Hem-Fir", "A35": 3.00, "LTP4": 2.58, "LTP5": 2.74, "LS70": 3.58, "LS90": 5.00},
    {"Stud Species": "S-P-F", "A35": 3.00, "LTP4": 2.58, "LTP5": 2.74, "LS70": 3.58, "LS90": 5.00},
    {"Stud Species": "Northern", "A35": 2.55, "LTP4": 2.19, "LTP5": 2.33, "LS70": 3.04, "LS90": 4.25},
])

BEARING_PLATES = pd.DataFrame([
    {"Model No.": "BPRTUD3-4B", "D.Fir-L (lb.)": 6120, "D.Fir-L (kN)": 27.26, "S-P-F (lb.)": 6120, "S-P-F (kN)": 27.26},
    {"Model No.": "BPRTUD5-6A", "D.Fir-L (lb.)": 7060, "D.Fir-L (kN)": 31.45, "S-P-F (lb.)": 7060, "S-P-F (kN)": 31.45},
    {"Model No.": "BPRTUD5-6B", "D.Fir-L (lb.)": 18110, "D.Fir-L (kN)": 80.67, "S-P-F (lb.)": 13705, "S-P-F (kN)": 61.05},
    {"Model No.": "BPRTUD5-6C", "D.Fir-L (lb.)": 23400, "D.Fir-L (kN)": 104.23, "S-P-F (lb.)": 17705, "S-P-F (kN)": 78.86},
    {"Model No.": "BPRTUD5-8",  "D.Fir-L (lb.)": 5200, "D.Fir-L (kN)": 23.13, "S-P-F (lb.)": 5200, "S-P-F (kN)": 23.13},
    {"Model No.": "BPRTUD7-8A", "D.Fir-L (lb.)": 17555, "D.Fir-L (kN)": 78.09, "S-P-F (lb.)": 13285, "S-P-F (kN)": 59.10},
    {"Model No.": "BPRTUD7-8B", "D.Fir-L (lb.)": 18740, "D.Fir-L (kN)": 83.36, "S-P-F (lb.)": 18740, "S-P-F (kN)": 83.36},
    {"Model No.": "BPRTUD7-8C", "D.Fir-L (lb.)": 30175, "D.Fir-L (kN)": 134.23, "S-P-F (lb.)": 22835, "S-P-F (kN)": 101.58},
])

PANEL_BUCKLING = pd.DataFrame([
    {"Nominal thickness (mm)": 7.5, "No. plies": 3, "Ba0": 55000, "Ba90": 24000, "Bv": 3400},
    {"Nominal thickness (mm)": 9.5, "No. plies": 3, "Ba0": 55000, "Ba90": 28000, "Bv": 4300},
    {"Nominal thickness (mm)": 12.5, "No. plies": 3, "Ba0": 81000, "Ba90": 39000, "Bv": 5700},
    {"Nominal thickness (mm)": 15.5, "No. plies": 4, "Ba0": 59000, "Ba90": 75000, "Bv": 7100},
    {"Nominal thickness (mm)": 18.5, "No. plies": 5, "Ba0": 83000, "Ba90": 69000, "Bv": 8600},
    {"Nominal thickness (mm)": 20.5, "No. plies": 5, "Ba0": 100000, "Ba90": 89000, "Bv": 9500},
    {"Nominal thickness (mm)": 22.5, "No. plies": 6, "Ba0": 130000, "Ba90": 69000, "Bv": 10000},
    {"Nominal thickness (mm)": 25.5, "No. plies": 7, "Ba0": 120000, "Ba90": 98000, "Bv": 12000},
])

SILL_BOLT_CAPACITY = pd.DataFrame([
    {"Species": sp, "Bolt diameter (mm)": d, "Capacity (kN)": cap}
    for (sp, d), cap in {
        ("D.Fir-L", 16): 8.09, ("D.Fir-L", 19): 10.90,
        ("Hem-Fir", 16): 7.80, ("Hem-Fir", 19): 10.60,
        ("S-P-F", 16): 7.41, ("S-P-F", 19): 9.84,
        ("Northern", 16): 6.68, ("Northern", 19): 8.20,
    }.items()
])

SILL_NAIL_A = pd.DataFrame([
    {"Species": "D.Fir-L", "Capacity (kN)": 0.786},
    {"Species": "Hem-Fir", "Capacity (kN)": 0.753},
    {"Species": "S-P-F", "Capacity (kN)": 0.707},
    {"Species": "Northern", "Capacity (kN)": 0.623},
])
SILL_NAIL_B = pd.DataFrame([
    {"Species": "D.Fir-L", "Capacity (kN)": 0.950},
    {"Species": "Hem-Fir", "Capacity (kN)": 0.910},
    {"Species": "S-P-F", "Capacity (kN)": 0.854},
    {"Species": "Northern", "Capacity (kN)": 0.752},
])

DATABASES: Dict[str, pd.DataFrame] = {
    "Sheathing Bv": SHEATHING_BV,
    "Lumber E": LUMBER_E,
    "Take-up devices": TAKEUP_DEVICES,
    "Rod resistance": ROD_RESISTANCE,
    "Rod geometry / area": ROD_GEOMETRY,
    "Shear clips": SHEAR_CLIPS,
    "Bearing plates": BEARING_PLATES,
    "Panel buckling properties": PANEL_BUCKLING,
    "Sill bolt capacity": SILL_BOLT_CAPACITY,
    "Sill Nail A": SILL_NAIL_A,
    "Sill Nail B": SILL_NAIL_B,
}


# -----------------------------------------------------------------------------
# Lookup / geometry helpers
# -----------------------------------------------------------------------------
STUD_DEPTH_MM = {"2x4": 89.0, "2x6": 140.0}
STUD_WIDTH_MM = 38.0
DEFAULT_CAVITY_MM = 9.0 * 25.4  # 228.6 mm; user-confirmed symmetric assumption
ROD_E_N_PER_MM2 = 200000.0
DEFAULT_COMPRESSION_BEARING_LENGTH_MM = 3.0 * 38.0


def get_bv(panel_type: str, thickness_mm: float, panel_sides: str = "S.S") -> float:
    q = SHEATHING_BV[(SHEATHING_BV["Panel Type"] == panel_type) & np.isclose(SHEATHING_BV["Thickness (mm)"], float(thickness_mm))]
    if q.empty:
        raise ValueError(f"No Bv value for {panel_type} {thickness_mm:g} mm.")
    base = float(q.iloc[0]["Bv (N/mm)"])
    return base * (2.0 if str(panel_sides).upper() == "B.S" else 1.0)


def get_lumber_e(species: str, grade: str) -> Tuple[float, float]:
    q = LUMBER_E[(LUMBER_E["Species"] == species) & (LUMBER_E["Grade"] == grade)]
    if q.empty:
        raise ValueError(f"No E values for {species}, {grade}.")
    r = q.iloc[0]
    return float(r["E parallel (N/mm²)"]), float(r["E perp (N/mm²)"])


def get_rod_properties(model: str) -> Dict[str, float | str]:
    for _, r in ROD_GEOMETRY.iterrows():
        if model == r["Strong Rod Standard"]:
            return {
                "Rod Diameter": r["Rod Diameter"], "Ag_mm2": float(r["Ag (mm²)"]), "Ane_mm2": float(r["Ane (mm²)"]),
                "At_mm2": 0.4 * float(r["Ag (mm²)"]) + 0.6 * float(r["Ane (mm²)"]),
                "Tr_kN": float(r["Tr Standard (kN)"]), "Strength Class": "Standard",
            }
        if model == r["Strong Rod High Strength"]:
            return {
                "Rod Diameter": r["Rod Diameter"], "Ag_mm2": float(r["Ag (mm²)"]), "Ane_mm2": float(r["Ane (mm²)"]),
                "At_mm2": 0.4 * float(r["Ag (mm²)"]) + 0.6 * float(r["Ane (mm²)"]),
                "Tr_kN": float(r["Tr High (kN)"]), "Strength Class": "High Strength",
            }
    # SR9H150 / SR10H150 exist in the resistance table but not the geometry table.
    raise ValueError(f"Rod model {model!r} does not have Ag/Ane data in the embedded rod geometry table.")


def get_takeup(model: str) -> Dict[str, float]:
    q = TAKEUP_DEVICES[TAKEUP_DEVICES["Model No."] == model]
    if q.empty:
        raise ValueError(f"Take-up device {model!r} is not in the embedded database.")
    r = q.iloc[0]
    return {
        "DR_mm": float(r["Seating Increment DR (mm)"]),
        "DF_mm": float(r["Deflection at Factored Resistance DF (mm)"]),
        "Diameter_in": float(r["Diameter (in.)"]),
    }


def calculate_lc_mm(wall_length_m: float, chord_studs_per_end: int, cavity_mm: float = DEFAULT_CAVITY_MM) -> float:
    """Symmetric rod/chord spacing used by the user's validated spreadsheet.

    Lc = Ls - (n_chord * 38 + 228.6), where n_chord is the number of chord studs
    at EACH end. This makes the old total-stud/divide-by-two convention explicit.
    """
    if wall_length_m <= 0:
        raise ValueError("Wall length must be > 0.")
    if int(chord_studs_per_end) < 1:
        raise ValueError("Chord studs per end must be >= 1.")
    lc = float(wall_length_m) * 1000.0 - (int(chord_studs_per_end) * STUD_WIDTH_MM + float(cavity_mm))
    if lc <= 0:
        raise ValueError("Calculated Lc is <= 0. Increase wall length or reduce chord-pack/cavity dimensions.")
    return lc


def transformed_section(
    wall_length_m: float,
    chord_studs_per_end: int,
    stud_size: str,
    species: str,
    grade: str,
    rod_model: str,
    cavity_mm: float = DEFAULT_CAVITY_MM,
) -> Dict[str, float]:
    if stud_size not in STUD_DEPTH_MM:
        raise ValueError(f"Unsupported stud size {stud_size!r}.")
    Ec, Eperp = get_lumber_e(species, grade)
    Ac = int(chord_studs_per_end) * STUD_WIDTH_MM * STUD_DEPTH_MM[stud_size]
    Lc = calculate_lc_mm(wall_length_m, chord_studs_per_end, cavity_mm)
    rod = get_rod_properties(rod_model)
    At = float(rod["At_mm2"])
    n = ROD_E_N_PER_MM2 / Ec
    At_tr = At * n
    y_tr = Ac * Lc / (At_tr + Ac)
    I_tr = At_tr * y_tr**2 + Ac * (Lc - y_tr)**2
    EI_tr = Ec * I_tr
    return {
        "Ec (N/mm²)": Ec, "Eperp (N/mm²)": Eperp, "Ac (mm²)": Ac,
        "Lc (mm)": Lc, "At (mm²)": At, "n = Et/Ec": n, "At,tr (mm²)": At_tr,
        "ytr (mm)": y_tr, "Itr (mm⁴)": I_tr, "EItr (N·mm²)": EI_tr,
        "Tr (kN)": float(rod["Tr_kN"]),
    }


def nail_response(V_kN: float, wall_length_m: float, nail_spacing_mm: float, nail_diameter_mm: float, panel_sides: str) -> Dict[str, float]:
    if wall_length_m <= 0 or nail_spacing_mm <= 0 or nail_diameter_mm <= 0:
        raise ValueError("Wall length, nail spacing and nail diameter must be > 0.")
    v_N_per_mm = abs(float(V_kN)) / float(wall_length_m)  # kN/m == N/mm
    divisor = 2.0 if str(panel_sides).upper() == "B.S" else 1.0
    force_per_nail_N = v_N_per_mm * float(nail_spacing_mm) / divisor
    en_mm = ((0.013 * force_per_nail_N) / float(nail_diameter_mm)**2) ** 2
    return {"v (N/mm)": v_N_per_mm, "Force per nail (N)": force_per_nail_N, "en (mm)": en_mm}


def anchorage_response(
    tension_kN: float,
    compression_kN: float,
    takeup_model: str,
    rod_model: str,
    Eperp_N_per_mm2: float,
    Ac_mm2: float,
    compression_bearing_length_mm: float = DEFAULT_COMPRESSION_BEARING_LENGTH_MM,
) -> Dict[str, float]:
    if Eperp_N_per_mm2 <= 0 or Ac_mm2 <= 0:
        raise ValueError("Eperp and Ac must be > 0.")
    tud = get_takeup(takeup_model)
    rod = get_rod_properties(rod_model)
    Tr = float(rod["Tr_kN"])
    if Tr <= 0:
        raise ValueError("Rod resistance Tr must be > 0.")
    T = max(float(tension_kN), 0.0)
    C = max(float(compression_kN), 0.0)
    device_def = (T / Tr) * tud["DF_mm"]
    seating = tud["DR_mm"]
    compression_def = (C * 1000.0 / (float(Eperp_N_per_mm2) * float(Ac_mm2))) * float(compression_bearing_length_mm)
    da = device_def + seating + compression_def
    return {
        "Tension chord force (kN)": T,
        "Compression chord force (kN)": C,
        "Rod utilization T/Tr": T / Tr,
        "Take-up device load deflection (mm)": device_def,
        "Take-up seating increment (mm)": seating,
        "Compression perp. deformation (mm)": compression_def,
        "da total (mm)": da,
    }


# -----------------------------------------------------------------------------
# Storey model
# -----------------------------------------------------------------------------
REQUIRED_STOREY_COLUMNS = [
    "Storey", "Floor lateral force (kN)", "Height (m)", "Wall length (m)",
    "Panel Type", "Panel thickness (mm)", "Panel sides", "Nail diameter (mm)",
    "Nail spacing (mm)", "Species", "Grade", "Stud size", "Chord studs / end",
    "Rod model", "Take-up device", "Service dead line load (kN/m)",
    "Service live line load (kN/m)",
]


def default_storeys(n: int = 1) -> pd.DataFrame:
    rows = []
    for i in range(int(n)):
        rows.append({
            "Storey": i + 1,
            "Floor lateral force (kN)": 46.0 if i == n - 1 else 0.0,
            "Height (m)": 3.0,
            "Wall length (m)": 4.8,
            "Panel Type": "CSP",
            "Panel thickness (mm)": 12.5,
            "Panel sides": "S.S",
            "Nail diameter (mm)": 3.25,
            "Nail spacing (mm)": 150.0,
            "Species": "S-P-F",
            "Grade": "No.1/No.2",
            "Stud size": "2x6",
            "Chord studs / end": 4,
            "Rod model": "SR8H",
            "Take-up device": "CTUD87",
            "Service dead line load (kN/m)": 0.0,
            "Service live line load (kN/m)": 0.0,
        })
    return pd.DataFrame(rows)


def normalize_storeys(storeys: pd.DataFrame) -> pd.DataFrame:
    df = storeys.copy()
    missing = [c for c in REQUIRED_STOREY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing storey inputs: {missing}")
    df = df.iloc[:].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("Add at least one storey.")
    df["Storey"] = np.arange(1, len(df) + 1)
    numeric = [
        "Floor lateral force (kN)", "Height (m)", "Wall length (m)", "Panel thickness (mm)",
        "Nail diameter (mm)", "Nail spacing (mm)", "Chord studs / end",
        "Service dead line load (kN/m)", "Service live line load (kN/m)",
    ]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[numeric].isna().any().any():
        raise ValueError("All numeric storey inputs must be valid numbers.")
    if (df["Height (m)"] <= 0).any() or (df["Wall length (m)"] <= 0).any():
        raise ValueError("Storey height and wall length must be > 0.")
    if (df["Nail diameter (mm)"] <= 0).any() or (df["Nail spacing (mm)"] <= 0).any():
        raise ValueError("Nail diameter and spacing must be > 0.")
    if (df["Chord studs / end"] < 1).any():
        raise ValueError("Chord studs per end must be >= 1.")
    df["Chord studs / end"] = df["Chord studs / end"].astype(int)
    return df


def analyze_stacked_wall(
    storeys: pd.DataFrame,
    include_lower_storey_rotation: bool = True,
    live_load_fraction_in_compression: float = 0.5,
    cavity_mm: float = DEFAULT_CAVITY_MM,
    compression_bearing_length_mm: float = DEFAULT_COMPRESSION_BEARING_LENGTH_MM,
) -> Dict[str, object]:
    """Analyze a continuous-rod stacked wall using service-level app inputs.

    Input rows are Storey 1 (bottom) through Storey n (top).
    """
    df = normalize_storeys(storeys)
    nst = len(df)
    if not (0.0 <= float(live_load_fraction_in_compression) <= 1.0):
        raise ValueError("Live-load fraction must be between 0 and 1.")

    F = df["Floor lateral force (kN)"].to_numpy(float)
    H_m = df["Height (m)"].to_numpy(float)
    L_m = df["Wall length (m)"].to_numpy(float)
    H = H_m * 1000.0
    L = L_m * 1000.0

    # FPInnovations Eq. definitions: storey shear V_i is sum of lateral forces at i and above.
    V_kN = np.cumsum(F[::-1])[::-1]

    # Moment at TOP of each storey: M_i = sum_{j=i+1}^n V_j H_j.
    M_top_kNm = np.zeros(nst, dtype=float)
    for i in range(nst - 2, -1, -1):
        M_top_kNm[i] = M_top_kNm[i + 1] + V_kN[i + 1] * H_m[i + 1]
    M_base_kNm = M_top_kNm + V_kN * H_m

    # Service gravity resultants above each storey including current storey.
    dead_storey = df["Service dead line load (kN/m)"].to_numpy(float) * L_m
    live_storey = df["Service live line load (kN/m)"].to_numpy(float) * L_m
    dead_cum = np.cumsum(dead_storey[::-1])[::-1]
    live_cum = np.cumsum(live_storey[::-1])[::-1]

    rows = []
    theta = np.zeros(nst)
    alpha = np.zeros(nst)

    # First pass: local storey deformations and rotation increments.
    for i, r in df.iterrows():
        sec = transformed_section(
            float(r["Wall length (m)"]), int(r["Chord studs / end"]), str(r["Stud size"]),
            str(r["Species"]), str(r["Grade"]), str(r["Rod model"]), cavity_mm,
        )
        Lc_m = sec["Lc (mm)"] / 1000.0
        # Service chord forces based on overturning at base of the storey; no strength factors.
        T_kN = max(M_base_kNm[i] / Lc_m - dead_cum[i] / 2.0, 0.0)
        C_kN = M_base_kNm[i] / Lc_m + (dead_cum[i] + float(live_load_fraction_in_compression) * live_cum[i]) / 2.0
        anch = anchorage_response(
            T_kN, C_kN, str(r["Take-up device"]), str(r["Rod model"]),
            sec["Eperp (N/mm²)"], sec["Ac (mm²)"], compression_bearing_length_mm,
        )
        nail = nail_response(V_kN[i], float(r["Wall length (m)"]), float(r["Nail spacing (mm)"]), float(r["Nail diameter (mm)"]), str(r["Panel sides"]))
        Bv = get_bv(str(r["Panel Type"]), float(r["Panel thickness (mm)"]), str(r["Panel sides"]))

        V_N = V_kN[i] * 1000.0
        M_top_Nmm = M_top_kNm[i] * 1.0e6
        EI = sec["EItr (N·mm²)"]
        db = V_N * H[i]**3 / (3.0 * EI) + M_top_Nmm * H[i]**2 / (2.0 * EI)
        ds = V_N * H[i] / (L[i] * Bv)
        dn = 0.0025 * H[i] * nail["en (mm)"]
        da_lat = H[i] / L[i] * anch["da total (mm)"]

        theta[i] = M_top_Nmm * H[i] / EI + V_N * H[i]**2 / (2.0 * EI)
        alpha[i] = anch["da total (mm)"] / L[i]

        rows.append({
            **r.to_dict(),
            "Storey shear V (kN)": V_kN[i],
            "M top (kN·m)": M_top_kNm[i],
            "M base (kN·m)": M_base_kNm[i],
            "Cumulative dead (kN)": dead_cum[i],
            "Cumulative live (kN)": live_cum[i],
            **sec,
            "Effective Bv (N/mm)": Bv,
            **nail,
            **anch,
            "Δ bending (mm)": db,
            "Δ panel shear (mm)": ds,
            "Δ nail slip (mm)": dn,
            "Δ anchorage (mm)": da_lat,
            "theta increment (rad)": theta[i],
            "alpha increment (rad)": alpha[i],
        })

    out = pd.DataFrame(rows)

    # Eq. 13: only rotations from STOREYS BELOW current storey (exclude current storey).
    rotation_below = np.zeros(nst)
    running = 0.0
    for i in range(nst):
        rotation_below[i] = running
        running += theta[i] + alpha[i]
    if not include_lower_storey_rotation:
        rotation_below[:] = 0.0
    d_rot = H * rotation_below
    out["Rotation below (rad)"] = rotation_below
    out["Δ rotation from below (mm)"] = d_rot
    out["Δ total inter-storey (mm)"] = (
        out["Δ bending (mm)"] + out["Δ panel shear (mm)"] + out["Δ nail slip (mm)"] +
        out["Δ anchorage (mm)"] + out["Δ rotation from below (mm)"]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        out["k secant (kN/m)"] = np.where(
            out["Δ total inter-storey (mm)"] > 0,
            out["Storey shear V (kN)"] / (out["Δ total inter-storey (mm)"] / 1000.0),
            np.inf,
        )
        out["Inter-storey drift ratio"] = out["Δ total inter-storey (mm)"] / (out["Height (m)"] * 1000.0)
    out["Cumulative lateral displacement (mm)"] = np.cumsum(out["Δ total inter-storey (mm)"].to_numpy(float))

    # Fractions for visualization.
    total = out["Δ total inter-storey (mm)"].replace(0, np.nan)
    for col, name in [
        ("Δ bending (mm)", "Bending fraction"),
        ("Δ panel shear (mm)", "Panel shear fraction"),
        ("Δ nail slip (mm)", "Nail slip fraction"),
        ("Δ anchorage (mm)", "Anchorage fraction"),
        ("Δ rotation from below (mm)", "Lower-storey rotation fraction"),
    ]:
        out[name] = out[col] / total

    return {
        "storeys": out,
        "meta": {
            "include_lower_storey_rotation": bool(include_lower_storey_rotation),
            "live_load_fraction_in_compression": float(live_load_fraction_in_compression),
            "cavity_mm": float(cavity_mm),
            "compression_bearing_length_mm": float(compression_bearing_length_mm),
            "storey_order": "bottom_to_top",
        },
    }


def single_storey_response_from_row(row: pd.Series | Dict[str, object], V_kN: float | None = None, **kwargs) -> Dict[str, float]:
    d = dict(row)
    d["Storey"] = 1
    if V_kN is not None:
        d["Floor lateral force (kN)"] = float(V_kN)
    result = analyze_stacked_wall(pd.DataFrame([d]), **kwargs)
    return result["storeys"].iloc[0].to_dict()


def run_mechanics_self_tests() -> pd.DataFrame:
    tests = []
    base = default_storeys(1)
    r = analyze_stacked_wall(base)["storeys"].iloc[0]
    tests.append({"Test": "Single-storey lower rotation = 0", "Value": float(r["Δ rotation from below (mm)"]), "Pass": abs(float(r["Δ rotation from below (mm)"])) < 1e-12})

    lc = calculate_lc_mm(4.8, 4)
    expected_lc = 4800.0 - (4 * 38.0 + 228.6)
    tests.append({"Test": "Symmetric Lc geometry", "Value": lc, "Pass": abs(lc - expected_lc) < 1e-12})

    nr = nail_response(46.0, 4.8, 150.0, 3.25, "S.S")
    expected_en = ((0.013 * nr["Force per nail (N)"]) / 3.25**2) ** 2
    tests.append({"Test": "Validated nail-slip equation", "Value": nr["en (mm)"], "Pass": abs(nr["en (mm)"] - expected_en) < 1e-12})

    # Two-storey check: bottom storey has no lower rotation; top storey must include bottom rotation.
    two = default_storeys(2)
    two.loc[0, "Floor lateral force (kN)"] = 10.0
    two.loc[1, "Floor lateral force (kN)"] = 20.0
    rr = analyze_stacked_wall(two)["storeys"]
    tests.append({"Test": "Eq.13 excludes current storey rotation", "Value": float(rr.loc[0, "Rotation below (rad)"]), "Pass": abs(float(rr.loc[0, "Rotation below (rad)"])) < 1e-12 and float(rr.loc[1, "Rotation below (rad)"]) > 0})

    # Service-force linearity sanity check at very small zero-gravity condition is not exact because nail slip is nonlinear;
    # instead verify force equilibrium definition V_i = sum F at and above.
    tests.append({"Test": "Storey shear accumulation", "Value": float(rr.loc[0, "Storey shear V (kN)"]), "Pass": abs(float(rr.loc[0, "Storey shear V (kN)"]) - 30.0) < 1e-12 and abs(float(rr.loc[1, "Storey shear V (kN)"]) - 20.0) < 1e-12})

    return pd.DataFrame(tests)
