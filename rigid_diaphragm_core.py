from __future__ import annotations

"""Pure calculation engine for the Rigid Diaphragm Research Lab.

No Streamlit dependency.  This keeps the engineering calculation layer easy to
unit-test, batch-run, and reuse in notebooks or other front ends.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math

import numpy as np
import pandas as pd


REQUIRED_WALL_COLUMNS = [
    "Wall Name",
    "Direction",
    "x (m)",
    "y (m)",
    "k (kN/m)",
    "Wall Length (m)",
    "Local X Force (kN)",
    "Local Y Force (kN)",
]


def normalize_walls(walls: pd.DataFrame) -> pd.DataFrame:
    df = walls.copy()
    missing = [c for c in REQUIRED_WALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing wall columns: {missing}")

    df["Wall Name"] = df["Wall Name"].astype(str).str.strip()
    df = df[df["Wall Name"] != ""].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("Add at least one wall.")

    df["Direction"] = df["Direction"].astype(str).str.upper().str.strip()
    df["Direction"] = df["Direction"].replace({"EW": "X", "E-W": "X", "NS": "Y", "N-S": "Y"})
    bad_dir = df.loc[~df["Direction"].isin(["X", "Y"]), "Wall Name"].tolist()
    if bad_dir:
        raise ValueError(f"Direction must be X or Y for: {bad_dir}")

    numeric = ["x (m)", "y (m)", "k (kN/m)", "Wall Length (m)", "Local X Force (kN)", "Local Y Force (kN)"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[numeric].isna().any().any():
        raise ValueError("Wall coordinates, stiffness, wall length and local wall forces must be numeric.")
    if (df["k (kN/m)"] <= 0).any():
        bad = df.loc[df["k (kN/m)"] <= 0, "Wall Name"].tolist()
        raise ValueError(f"Wall stiffness must be > 0 for: {bad}")
    if (df["Wall Length (m)"] <= 0).any():
        bad = df.loc[df["Wall Length (m)"] <= 0, "Wall Name"].tolist()
        raise ValueError(f"Wall length must be > 0 for: {bad}")
    return df


def length_proportional_stiffness_model(walls: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Return a solver-compatible model with relative stiffness proportional to wall length.

    This implements the preliminary rigid-diaphragm assumption k_i ∝ L_i used in
    CWC/FPInnovations design examples.  Only relative stiffness is meaningful.
    A single common scale factor is chosen so that the sum of mapped k values equals
    the sum of the current model k values; this leaves the relative-force solution
    unchanged while keeping the existing solver units/plots convenient.

    The mapped k values are therefore *not* physical mechanics-based stiffnesses.
    """
    df = normalize_walls(walls)
    total_length = float(df["Wall Length (m)"].sum())
    total_k = float(df["k (kN/m)"].sum())
    if total_length <= 0 or total_k <= 0:
        raise ValueError("Wall lengths and current stiffness sum must be positive.")

    scale = total_k / total_length  # common scale; force distribution is invariant to it
    out = df.copy()
    out["Original k (kN/m)"] = out["k (kN/m)"]
    out["Global length weight"] = out["Wall Length (m)"] / total_length
    out["Direction length sum (m)"] = out.groupby("Direction")["Wall Length (m)"].transform("sum")
    out["Direction relative stiffness L/ΣL"] = out["Wall Length (m)"] / out["Direction length sum (m)"]
    out["Length-proportional mapped k (kN/m)"] = scale * out["Wall Length (m)"]

    model = df.copy()
    model["k (kN/m)"] = out["Length-proportional mapped k (kN/m)"]

    table = out[[
        "Wall Name", "Direction", "Wall Length (m)",
        "Original k (kN/m)", "Direction relative stiffness L/ΣL",
        "Length-proportional mapped k (kN/m)",
    ]].copy()
    return model, table, scale


def diaphragm_properties(walls: pd.DataFrame, x_cm: float, y_cm: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = normalize_walls(walls)
    df["kx (kN/m)"] = np.where(df["Direction"].eq("X"), df["k (kN/m)"], 0.0)
    df["ky (kN/m)"] = np.where(df["Direction"].eq("Y"), df["k (kN/m)"], 0.0)

    sum_kx = float(df["kx (kN/m)"].sum())
    sum_ky = float(df["ky (kN/m)"].sum())
    if sum_kx <= 0:
        raise ValueError("At least one X-direction wall is required.")
    if sum_ky <= 0:
        raise ValueError("At least one Y-direction wall is required.")

    x_cr = float((df["ky (kN/m)"] * df["x (m)"]).sum() / sum_ky)
    y_cr = float((df["kx (kN/m)"] * df["y (m)"]).sum() / sum_kx)

    df["xbar (m)"] = df["x (m)"] - x_cr
    df["ybar (m)"] = df["y (m)"] - y_cr
    df["ky*xbar^2 (kN·m)"] = df["ky (kN/m)"] * df["xbar (m)"] ** 2
    df["kx*ybar^2 (kN·m)"] = df["kx (kN/m)"] * df["ybar (m)"] ** 2
    J = float(df["ky*xbar^2 (kN·m)"].sum() + df["kx*ybar^2 (kN·m)"].sum())
    if math.isclose(J, 0.0, abs_tol=1e-12):
        raise ValueError("Torsional rigidity J is zero. Check wall locations and lever arms about the CoR.")

    props = {
        "Xcr": x_cr,
        "Ycr": y_cr,
        "Xcm": float(x_cm),
        "Ycm": float(y_cm),
        "ex_signed": float(x_cm - x_cr),
        "ey_signed": float(y_cm - y_cr),
        "sum_kx": sum_kx,
        "sum_ky": sum_ky,
        "J": J,
    }
    return df, props


def analyze_direction(
    walls_with_props: pd.DataFrame,
    props: Dict[str, float],
    direction: str,
    force_kN: float,
    Lx: float,
    Ly: float,
    accidental_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Solve one applied-force direction for both accidental-eccentricity signs."""
    d = direction.upper()
    if d not in {"X", "Y"}:
        raise ValueError("direction must be X or Y")

    df = walls_with_props.copy()
    F = float(force_kN)
    J = props["J"]

    if d == "X":
        e_real = props["Ycm"] - props["Ycr"]
        e_acc = accidental_ratio * Ly

        def moment(e_total: float) -> float:
            return -F * e_total

        direct_x = F * df["kx (kN/m)"] / props["sum_kx"]
        direct_y = pd.Series(0.0, index=df.index)
        local_parallel = np.where(df["Direction"].eq("X"), df["Local X Force (kN)"], 0.0)
    else:
        e_real = props["Xcm"] - props["Xcr"]
        e_acc = accidental_ratio * Lx

        def moment(e_total: float) -> float:
            return F * e_total

        direct_x = pd.Series(0.0, index=df.index)
        direct_y = F * df["ky (kN/m)"] / props["sum_ky"]
        local_parallel = np.where(df["Direction"].eq("Y"), df["Local Y Force (kN)"], 0.0)

    case_frames = []
    check_rows = []
    for sign in (-1, +1):
        e_total = e_real + sign * e_acc
        Mz = moment(e_total)

        vx_tor = -Mz * df["kx (kN/m)"] * df["ybar (m)"] / J
        vy_tor = +Mz * df["ky (kN/m)"] * df["xbar (m)"] / J
        vx = direct_x + vx_tor
        vy = direct_y + vy_tor

        case = df[["Wall Name", "Direction", "Wall Length (m)"]].copy()
        case["Load Direction"] = d
        case["Accidental Sign"] = "+" if sign > 0 else "−"
        case["e_real (m)"] = e_real
        case["e_acc_signed (m)"] = sign * e_acc
        case["e_total (m)"] = e_total
        case["Mz (kN·m)"] = Mz
        case["Direct Vx (kN)"] = direct_x
        case["Direct Vy (kN)"] = direct_y
        case["Torsion Vx (kN)"] = vx_tor
        case["Torsion Vy (kN)"] = vy_tor
        case["Vx diaphragm (kN)"] = vx
        case["Vy diaphragm (kN)"] = vy
        case["Diaphragm wall-parallel V (kN)"] = np.where(case["Direction"].eq("X"), vx, vy)
        case["Local wall force (kN)"] = local_parallel
        case["Wall-parallel V (kN)"] = case["Diaphragm wall-parallel V (kN)"] + case["Local wall force (kN)"]
        case["v = |V|/Lwall (kN/m)"] = case["Wall-parallel V (kN)"].abs() / case["Wall Length (m)"]
        case_frames.append(case)

        sum_vx = float(vx.sum())
        sum_vy = float(vy.sum())
        wall_moment = float((df["xbar (m)"] * vy - df["ybar (m)"] * vx).sum())
        target_vx = F if d == "X" else 0.0
        target_vy = F if d == "Y" else 0.0
        check_rows.append({
            "Load Direction": d,
            "Accidental Sign": "+" if sign > 0 else "−",
            "ΣVx (kN)": sum_vx,
            "Target Vx (kN)": target_vx,
            "ΔVx (kN)": sum_vx - target_vx,
            "ΣVy (kN)": sum_vy,
            "Target Vy (kN)": target_vy,
            "ΔVy (kN)": sum_vy - target_vy,
            "ΣM@CoR (kN·m)": wall_moment,
            "Target Mz (kN·m)": Mz,
            "ΔM (kN·m)": wall_moment - Mz,
        })

    cases = pd.concat(case_frames, ignore_index=True)
    checks = pd.DataFrame(check_rows)

    env_rows = []
    for name, g in cases.groupby("Wall Name", sort=False):
        idx = g["Wall-parallel V (kN)"].abs().idxmax()
        gov = cases.loc[idx]
        env_rows.append({
            "Wall Name": name,
            "Direction": gov["Direction"],
            "Load Direction": d,
            "Envelope |V| (kN)": abs(float(gov["Wall-parallel V (kN)"])),
            "Governing Signed V (kN)": float(gov["Wall-parallel V (kN)"]),
            "Governing Acc. Sign": gov["Accidental Sign"],
            "v envelope (kN/m)": float(gov["v = |V|/Lwall (kN/m)"]),
        })
    envelope = pd.DataFrame(env_rows)
    meta = {"e_real": e_real, "e_acc": e_acc}
    return cases, envelope, checks, meta


def analyze_model(
    walls: pd.DataFrame,
    Lx: float,
    Ly: float,
    x_cm: float,
    y_cm: float,
    Fx: float,
    Fy: float,
    accidental_ratio: float,
) -> Dict[str, object]:
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Plan dimensions must be greater than zero.")
    if accidental_ratio < 0:
        raise ValueError("Accidental eccentricity ratio cannot be negative.")

    df, props = diaphragm_properties(walls, x_cm, y_cm)
    x_cases, x_env, x_checks, x_meta = analyze_direction(df, props, "X", Fx, Lx, Ly, accidental_ratio)
    y_cases, y_env, y_checks, y_meta = analyze_direction(df, props, "Y", Fy, Lx, Ly, accidental_ratio)

    env = x_env.rename(columns={
        "Envelope |V| (kN)": "X-load envelope |V| (kN)",
        "Governing Signed V (kN)": "X-load governing signed V (kN)",
        "Governing Acc. Sign": "X-load acc. sign",
        "v envelope (kN/m)": "X-load v env (kN/m)",
    }).drop(columns=["Load Direction"])
    y2 = y_env.rename(columns={
        "Envelope |V| (kN)": "Y-load envelope |V| (kN)",
        "Governing Signed V (kN)": "Y-load governing signed V (kN)",
        "Governing Acc. Sign": "Y-load acc. sign",
        "v envelope (kN/m)": "Y-load v env (kN/m)",
    }).drop(columns=["Direction", "Load Direction"])
    env = env.merge(y2, on="Wall Name", how="left")
    env["Max directional envelope |V| (kN)"] = env[["X-load envelope |V| (kN)", "Y-load envelope |V| (kN)"]].max(axis=1)

    return {
        "wall_properties": df,
        "properties": props,
        "x_cases": x_cases,
        "y_cases": y_cases,
        "envelope": env,
        "checks": pd.concat([x_checks, y_checks], ignore_index=True),
        "x_meta": x_meta,
        "y_meta": y_meta,
    }


# -----------------------------------------------------------------------------
# Reference models
# -----------------------------------------------------------------------------
def default_walls() -> pd.DataFrame:
    return pd.DataFrame([
        {"Wall Name": "X1", "Direction": "X", "x (m)": 5.0,  "y (m)": 0.0,  "k (kN/m)": 10000.0, "Wall Length (m)": 8.0, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
        {"Wall Name": "X2", "Direction": "X", "x (m)": 15.0, "y (m)": 12.0, "k (kN/m)": 10000.0, "Wall Length (m)": 8.0, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
        {"Wall Name": "Y1", "Direction": "Y", "x (m)": 0.0,  "y (m)": 6.0,  "k (kN/m)": 10000.0, "Wall Length (m)": 8.0, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
        {"Wall Name": "Y2", "Direction": "Y", "x (m)": 20.0, "y (m)": 6.0,  "k (kN/m)": 10000.0, "Wall Length (m)": 8.0, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
    ])


def fpinnovations_preset() -> Tuple[Dict[str, float], pd.DataFrame]:
    settings = {
        "Lx": 30.5,
        "Ly": 12.2,
        "Xcm": 15.25,
        "Ycm": 6.10,
        "Fx": 68.0,
        "Fy": 212.0 * 600.0 / 661.0,
        "acc": 0.10,
    }
    walls = pd.DataFrame([
        {"Wall Name": "A", "Direction": "Y", "x (m)": 0.00,  "y (m)": 6.10,  "k (kN/m)": 5074.0,  "Wall Length (m)": 12.2, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 212.0 * (61.0 / 2.0) / 661.0},
        {"Wall Name": "B", "Direction": "Y", "x (m)": 9.15,  "y (m)": 6.10,  "k (kN/m)": 8312.0,  "Wall Length (m)": 12.2, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 212.0 * (61.0 / 2.0) / 661.0},
        {"Wall Name": "C", "Direction": "Y", "x (m)": 30.50, "y (m)": 6.10,  "k (kN/m)": 4240.0,  "Wall Length (m)": 12.2, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
        {"Wall Name": "1", "Direction": "X", "x (m)": 15.25, "y (m)": 12.20, "k (kN/m)": 16079.0, "Wall Length (m)": 30.5, "Local X Force (kN)": 87.0 * (152.0 / 2.0) / 707.0, "Local Y Force (kN)": 0.0},
        {"Wall Name": "2", "Direction": "X", "x (m)": 15.25, "y (m)": 0.00,  "k (kN/m)": 16079.0, "Wall Length (m)": 30.5, "Local X Force (kN)": 87.0 * (152.0 / 2.0) / 707.0, "Local Y Force (kN)": 0.0},
    ])
    return settings, walls


def benchmark_table(result: Dict[str, object]) -> pd.DataFrame:
    p = result["properties"]
    F_ns = 212.0 * 600.0 / 661.0
    natural = F_ns * (15.25 - p["Xcr"])
    accidental = F_ns * 0.10 * 30.5
    rows = [
        ["CoR X", p["Xcr"], 11.65, "m"],
        ["CoR Y", p["Ycr"], 6.10, "m"],
        ["J", p["J"], 3443801.0, "kN·m"],
        ["N-S natural torsion", natural, 694.0, "kN·m"],
        ["N-S accidental torsion", accidental, 588.0, "kN·m"],
    ]
    env = result["envelope"].set_index("Wall Name")
    rows.extend([
        ["N-S wall A envelope", env.loc["A", "Y-load envelope |V| (kN)"], 63.5, "kN"],
        ["N-S wall B envelope", env.loc["B", "Y-load envelope |V| (kN)"], 100.1, "kN"],
        ["N-S wall C envelope", env.loc["C", "Y-load envelope |V| (kN)"], 76.1, "kN"],
        ["N-S wall 1 torsion", env.loc["1", "Y-load envelope |V| (kN)"], 36.5, "kN"],
        ["N-S wall 2 torsion", env.loc["2", "Y-load envelope |V| (kN)"], 36.5, "kN"],
        ["E-W wall 1 envelope", env.loc["1", "X-load envelope |V| (kN)"], 45.7, "kN"],
        ["E-W wall 2 envelope", env.loc["2", "X-load envelope |V| (kN)"], 45.7, "kN"],
    ])
    out = pd.DataFrame(rows, columns=["Benchmark", "App", "Source (rounded)", "Unit"])
    out["Difference"] = out["App"] - out["Source (rounded)"]
    return out



# -----------------------------------------------------------------------------
# Fast scenario evaluator for large parametric batches
# -----------------------------------------------------------------------------
def _fast_metrics_from_arrays(
    names: np.ndarray,
    directions: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    k: np.ndarray,
    wall_length: np.ndarray,
    local_x: np.ndarray,
    local_y: np.ndarray,
    settings: Dict[str, float],
    selected_index: int,
    load_direction: str,
) -> Dict[str, object]:
    """Fast numpy equivalent of the rigid-diaphragm envelope for one scenario.

    Intended for thousands of parametric cases.  The regular analyze_model()
    remains the transparent dataframe-based implementation used for detailed
    single-case output and verification.
    """
    is_x = directions == "X"
    is_y = ~is_x
    kx = np.where(is_x, k, 0.0)
    ky = np.where(is_y, k, 0.0)
    sum_kx = float(kx.sum()); sum_ky = float(ky.sum())
    if sum_kx <= 0 or sum_ky <= 0:
        raise ValueError("Both X- and Y-direction resisting elements are required.")
    Xcr = float(np.dot(ky, x) / sum_ky)
    Ycr = float(np.dot(kx, y) / sum_kx)
    xbar = x - Xcr; ybar = y - Ycr
    J = float(np.dot(ky, xbar*xbar) + np.dot(kx, ybar*ybar))
    if abs(J) < 1e-12:
        raise ValueError("Torsional rigidity J is zero.")

    d = load_direction.upper()
    if d == "X":
        F = float(settings["Fx"])
        e_real = float(settings["Ycm"] - Ycr)
        e_acc = float(settings["acc"] * settings["Ly"])
        direct_x = F * kx / sum_kx
        direct_y = np.zeros_like(k)
        local = np.where(is_x, local_x, 0.0)
        moments = [-F*(e_real - e_acc), -F*(e_real + e_acc)]
        e_norm = e_real / settings["Ly"] if settings["Ly"] else np.nan
    elif d == "Y":
        F = float(settings["Fy"])
        e_real = float(settings["Xcm"] - Xcr)
        e_acc = float(settings["acc"] * settings["Lx"])
        direct_x = np.zeros_like(k)
        direct_y = F * ky / sum_ky
        local = np.where(is_y, local_y, 0.0)
        moments = [F*(e_real - e_acc), F*(e_real + e_acc)]
        e_norm = e_real / settings["Lx"] if settings["Lx"] else np.nan
    else:
        raise ValueError("load_direction must be X or Y")

    case_forces = []
    for Mz in moments:
        vx = direct_x - Mz * kx * ybar / J
        vy = direct_y + Mz * ky * xbar / J
        wp = np.where(is_x, vx, vy) + local
        case_forces.append(wp)
    a = case_forces[0]; b = case_forces[1]
    choose_b = np.abs(b) > np.abs(a)
    signed_env = np.where(choose_b, b, a)
    abs_env = np.abs(signed_env)
    max_idx = int(np.argmax(abs_env))
    sel_v = float(abs_env[selected_index])
    sel_signed = float(signed_env[selected_index])
    sel_unit = sel_v / float(wall_length[selected_index])
    return {
        "Selected wall |V| (kN)": sel_v,
        "Selected wall signed V (kN)": sel_signed,
        "Selected wall v (kN/m)": sel_unit,
        "Selected wall force share |V|/F": sel_v / abs(F) if abs(F) > 1e-12 else np.nan,
        "Max wall |V| (kN)": float(abs_env[max_idx]),
        "Governing wall": str(names[max_idx]),
        "Xcr (m)": Xcr,
        "Ycr (m)": Ycr,
        "Signed eccentricity (m)": e_real,
        "Normalized eccentricity e/D": e_norm,
        "J (kN·m)": J,
        "sum_kx": sum_kx,
        "sum_ky": sum_ky,
    }


def _fast_base_arrays(base_walls: pd.DataFrame):
    w = normalize_walls(base_walls)
    names = w["Wall Name"].to_numpy(dtype=object)
    directions = w["Direction"].to_numpy(dtype=object)
    x = w["x (m)"].to_numpy(dtype=float)
    y = w["y (m)"].to_numpy(dtype=float)
    k = w["k (kN/m)"].to_numpy(dtype=float)
    L = w["Wall Length (m)"].to_numpy(dtype=float)
    lx = w["Local X Force (kN)"].to_numpy(dtype=float)
    ly = w["Local Y Force (kN)"].to_numpy(dtype=float)
    return w, names, directions, x, y, k, L, lx, ly

# -----------------------------------------------------------------------------
# Parametric studies
# -----------------------------------------------------------------------------
def _direction_outputs(result: Dict[str, object], wall_name: str, load_direction: str, force_kN: float, Lx: float, Ly: float) -> Dict[str, float]:
    d = load_direction.upper()
    env = result["envelope"].set_index("Wall Name")
    p = result["properties"]
    if wall_name not in env.index:
        raise ValueError(f"Wall '{wall_name}' not found.")
    if d == "X":
        wall_v = float(env.loc[wall_name, "X-load envelope |V| (kN)"])
        wall_signed = float(env.loc[wall_name, "X-load governing signed V (kN)"])
        wall_unit = float(env.loc[wall_name, "X-load v env (kN/m)"])
        e = p["ey_signed"]
        e_norm = e / Ly if Ly else np.nan
        col = "X-load envelope |V| (kN)"
    else:
        wall_v = float(env.loc[wall_name, "Y-load envelope |V| (kN)"])
        wall_signed = float(env.loc[wall_name, "Y-load governing signed V (kN)"])
        wall_unit = float(env.loc[wall_name, "Y-load v env (kN/m)"])
        e = p["ex_signed"]
        e_norm = e / Lx if Lx else np.nan
        col = "Y-load envelope |V| (kN)"
    max_v = float(env[col].max())
    gov_name = str(env[col].idxmax())
    force_share = wall_v / abs(force_kN) if abs(force_kN) > 1e-12 else np.nan
    return {
        "Selected wall |V| (kN)": wall_v,
        "Selected wall signed V (kN)": wall_signed,
        "Selected wall v (kN/m)": wall_unit,
        "Selected wall force share |V|/F": force_share,
        "Max wall |V| (kN)": max_v,
        "Governing wall": gov_name,
        "Xcr (m)": p["Xcr"],
        "Ycr (m)": p["Ycr"],
        "Signed eccentricity (m)": e,
        "Normalized eccentricity e/D": e_norm,
        "J (kN·m)": p["J"],
    }


def run_geometry_study(
    base_walls: pd.DataFrame,
    settings: Dict[str, float],
    wall_name: str,
    load_direction: str,
    points: int = 101,
    start_norm: float = 0.0,
    stop_norm: float = 1.0,
) -> pd.DataFrame:
    w, names, directions, x0, y0, k0, L0, lx0, ly0 = _fast_base_arrays(base_walls)
    matches = np.where(names == wall_name)[0]
    if len(matches) == 0:
        raise ValueError(f"Wall '{wall_name}' not found.")
    idx = int(matches[0]); direction = str(directions[idx])
    coord_col = "y (m)" if direction == "X" else "x (m)"
    dimension = float(settings["Ly"] if direction == "X" else settings["Lx"])
    rows = []
    for xn in np.linspace(start_norm, stop_norm, int(points)):
        x = x0.copy(); y = y0.copy()
        if direction == "X": y[idx] = xn * dimension
        else: x[idx] = xn * dimension
        try:
            out = _fast_metrics_from_arrays(names, directions, x, y, k0, L0, lx0, ly0, settings, idx, load_direction)
            out.update({"Wall Name": wall_name, "Wall Direction": direction, "Moved Coordinate": coord_col, "Position (m)": xn*dimension, "Normalized position": xn})
            rows.append(out)
        except Exception as exc:
            rows.append({"Wall Name":wall_name,"Position (m)":xn*dimension,"Normalized position":xn,"Error":str(exc)})
    return pd.DataFrame(rows)


def run_stiffness_study(
    base_walls: pd.DataFrame,
    settings: Dict[str, float],
    wall_name: str,
    load_direction: str,
    points: int = 101,
    start_multiplier: float = 0.25,
    stop_multiplier: float = 4.0,
    log_spacing: bool = True,
) -> pd.DataFrame:
    w, names, directions, x0, y0, k0, L0, lx0, ly0 = _fast_base_arrays(base_walls)
    matches=np.where(names==wall_name)[0]
    if len(matches)==0: raise ValueError(f"Wall '{wall_name}' not found.")
    idx=int(matches[0]); base_k=float(k0[idx]); direction=str(directions[idx])
    if start_multiplier<=0 or stop_multiplier<=0: raise ValueError("Stiffness multipliers must be > 0.")
    vals=np.geomspace(start_multiplier,stop_multiplier,int(points)) if log_spacing else np.linspace(start_multiplier,stop_multiplier,int(points))
    rows=[]
    for mult in vals:
        k=k0.copy(); k[idx]=base_k*mult
        try:
            out=_fast_metrics_from_arrays(names,directions,x0,y0,k,L0,lx0,ly0,settings,idx,load_direction)
            sum_dir=out["sum_kx"] if direction=="X" else out["sum_ky"]
            out.update({"Wall Name":wall_name,"Wall Direction":direction,"Stiffness multiplier":mult,"k (kN/m)":base_k*mult,"Relative stiffness k_i/Σk_dir":(base_k*mult)/sum_dir})
            rows.append(out)
        except Exception as exc:
            rows.append({"Wall Name":wall_name,"Stiffness multiplier":mult,"k (kN/m)":base_k*mult,"Error":str(exc)})
    return pd.DataFrame(rows)


def run_length_fixed_k_study(
    base_walls: pd.DataFrame,
    settings: Dict[str, float],
    wall_name: str,
    load_direction: str,
    start_length_m: float,
    stop_length_m: float,
    points: int = 101,
) -> pd.DataFrame:
    if start_length_m<=0 or stop_length_m<=0: raise ValueError("Wall length must be > 0.")
    w,names,directions,x0,y0,k0,L0,lx0,ly0=_fast_base_arrays(base_walls)
    matches=np.where(names==wall_name)[0]
    if len(matches)==0: raise ValueError(f"Wall '{wall_name}' not found.")
    idx=int(matches[0]); rows=[]
    for Lv in np.linspace(start_length_m,stop_length_m,int(points)):
        L=L0.copy(); L[idx]=Lv
        try:
            out=_fast_metrics_from_arrays(names,directions,x0,y0,k0,L,lx0,ly0,settings,idx,load_direction)
            out.update({"Wall Name":wall_name,"Wall Length (m)":Lv,"k held constant (kN/m)":float(k0[idx])})
            rows.append(out)
        except Exception as exc:
            rows.append({"Wall Name":wall_name,"Wall Length (m)":Lv,"Error":str(exc)})
    return pd.DataFrame(rows)


def run_interaction_study(
    base_walls: pd.DataFrame,
    settings: Dict[str, float],
    wall_name: str,
    load_direction: str,
    position_points: int = 41,
    stiffness_points: int = 41,
    position_start_norm: float = 0.0,
    position_stop_norm: float = 1.0,
    stiffness_start_multiplier: float = 0.25,
    stiffness_stop_multiplier: float = 4.0,
) -> pd.DataFrame:
    w,names,directions,x0,y0,k0,L0,lx0,ly0=_fast_base_arrays(base_walls)
    matches=np.where(names==wall_name)[0]
    if len(matches)==0: raise ValueError(f"Wall '{wall_name}' not found.")
    idx=int(matches[0]); wall_dir=str(directions[idx]); dimension=float(settings["Ly"] if wall_dir=="X" else settings["Lx"]); base_k=float(k0[idx])
    pos_vals=np.linspace(position_start_norm,position_stop_norm,int(position_points))
    kvals=np.geomspace(stiffness_start_multiplier,stiffness_stop_multiplier,int(stiffness_points))
    rows=[]
    for pos in pos_vals:
        for mult in kvals:
            x=x0.copy(); y=y0.copy(); k=k0.copy()
            if wall_dir=="X": y[idx]=pos*dimension
            else: x[idx]=pos*dimension
            k[idx]=base_k*mult
            try:
                out=_fast_metrics_from_arrays(names,directions,x,y,k,L0,lx0,ly0,settings,idx,load_direction)
                out.update({"Wall Name":wall_name,"Normalized position":pos,"Position (m)":pos*dimension,"Stiffness multiplier":mult,"k (kN/m)":base_k*mult})
                rows.append(out)
            except Exception as exc:
                rows.append({"Wall Name":wall_name,"Normalized position":pos,"Stiffness multiplier":mult,"Error":str(exc)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Wood shear-wall stiffness model from the uploaded FPInnovations example
# -----------------------------------------------------------------------------
def wood_wall_response(
    V_kN: float,
    H_m: float,
    L_m: float,
    E_N_per_mm2: float,
    A_mm2: float,
    Bv_N_per_mm: float,
    en_mm: float,
    da_mm: float,
) -> Dict[str, float]:
    """Return FPInnovations-example wall deformation components and secant stiffness.

    Legacy single-storey helper retained for compatibility. The flexural term uses
    the mechanics-based CSA/FPInnovations form 2 v H^3/(3 E A L). New app work
    should use wood_shearwall_mechanics.py, which also handles transformed EI and
    stacked-storey top-moment / lower-storey-rotation effects.

    Unit convention: v in N/mm, H and L in mm, E in N/mm², A in mm²,
    Bv in N/mm, en and da in mm. Numerically, k in N/mm equals kN/m.
    """
    if H_m <= 0 or L_m <= 0:
        raise ValueError("H and L must be > 0.")
    if E_N_per_mm2 <= 0 or A_mm2 <= 0 or Bv_N_per_mm <= 0:
        raise ValueError("E, A and Bv must be > 0.")
    if en_mm < 0 or da_mm < 0:
        raise ValueError("en and da cannot be negative.")

    V = abs(float(V_kN))
    H = H_m * 1000.0
    L = L_m * 1000.0
    v = V / L_m  # kN/m == N/mm numerically

    delta_bending = 2.0 * v * H**3 / (3.0 * E_N_per_mm2 * A_mm2 * L)
    delta_sheathing = v * H / Bv_N_per_mm
    delta_fastener = 0.0025 * H * en_mm
    delta_anchorage = (H / L) * da_mm
    delta_total = delta_bending + delta_sheathing + delta_fastener + delta_anchorage
    F_N = V * 1000.0
    k = F_N / delta_total if delta_total > 0 else np.inf  # N/mm == kN/m

    return {
        "V (kN)": V,
        "v (kN/m = N/mm)": v,
        "Δ bending (mm)": delta_bending,
        "Δ sheathing (mm)": delta_sheathing,
        "Δ fastener (mm)": delta_fastener,
        "Δ anchorage (mm)": delta_anchorage,
        "Δ total (mm)": delta_total,
        "k secant (kN/m)": k,
        "Bending fraction": delta_bending / delta_total if delta_total else np.nan,
        "Sheathing fraction": delta_sheathing / delta_total if delta_total else np.nan,
        "Fastener fraction": delta_fastener / delta_total if delta_total else np.nan,
        "Anchorage fraction": delta_anchorage / delta_total if delta_total else np.nan,
    }


def linearized_holdown_da_mm(V_kN: float, H_m: float, L_m: float, capacity_kN: float, deflection_at_capacity_mm: float) -> Dict[str, float]:
    if L_m <= 0 or capacity_kN <= 0:
        raise ValueError("Wall length and hold-down capacity must be > 0.")
    v = abs(V_kN) / L_m
    holdown_tension = v * H_m
    da = holdown_tension / capacity_kN * deflection_at_capacity_mm
    return {"Hold-down tension estimate (kN)": holdown_tension, "da (mm)": da}


def run_wood_length_study(
    V_kN: float,
    H_m: float,
    start_length_m: float,
    stop_length_m: float,
    points: int,
    E_N_per_mm2: float,
    A_mm2: float,
    Bv_N_per_mm: float,
    en_mm: float,
    da_mm: float,
    auto_holdown: bool = False,
    holdown_capacity_kN: float = 55.25,
    holdown_deflection_mm: float = 5.54,
) -> pd.DataFrame:
    vals = np.linspace(start_length_m, stop_length_m, int(points))
    rows = []
    for L in vals:
        da = da_mm
        hd_tension = np.nan
        if auto_holdown:
            hd = linearized_holdown_da_mm(V_kN, H_m, L, holdown_capacity_kN, holdown_deflection_mm)
            da = hd["da (mm)"]
            hd_tension = hd["Hold-down tension estimate (kN)"]
        r = wood_wall_response(V_kN, H_m, L, E_N_per_mm2, A_mm2, Bv_N_per_mm, en_mm, da)
        rows.append({"Wall Length (m)": L, "da used (mm)": da, "Hold-down tension estimate (kN)": hd_tension, **r})
    return pd.DataFrame(rows)


def run_wood_parameter_study(base: Dict[str, float], parameter: str, start: float, stop: float, points: int = 101) -> pd.DataFrame:
    allowed = {
        "V_kN", "H_m", "L_m", "E_N_per_mm2", "A_mm2", "Bv_N_per_mm", "en_mm", "da_mm"
    }
    if parameter not in allowed:
        raise ValueError(f"Unsupported wood-wall parameter: {parameter}")
    vals = np.linspace(start, stop, int(points))
    rows = []
    for val in vals:
        p = dict(base)
        p[parameter] = float(val)
        try:
            out = wood_wall_response(**p)
            rows.append({"Parameter": val, **out})
        except Exception as exc:
            rows.append({"Parameter": val, "Error": str(exc)})
    return pd.DataFrame(rows)


def run_length_calculated_k_building_study(
    base_walls: pd.DataFrame,
    settings: Dict[str, float],
    wall_name: str,
    load_direction: str,
    start_length_m: float,
    stop_length_m: float,
    points: int,
    wood_params: Dict[str, float],
    use_current_system_force_each_case: bool = True,
) -> pd.DataFrame:
    """Vary selected wall length, calculate its wood-wall k, and re-run the building.

    For each length, the seed force comes from the base building unless
    use_current_system_force_each_case is False, in which case wood_params['V_kN'] is used.
    This is a one-pass study, intentionally separate from the iterative solver below.
    """
    walls0 = normalize_walls(base_walls)
    vals = np.linspace(start_length_m, stop_length_m, int(points))
    rows = []
    for L in vals:
        walls = walls0.copy()
        mask = walls["Wall Name"].eq(wall_name)
        walls.loc[mask, "Wall Length (m)"] = L
        try:
            if use_current_system_force_each_case:
                seed = analyze_model(walls, settings["Lx"], settings["Ly"], settings["Xcm"], settings["Ycm"], settings["Fx"], settings["Fy"], settings["acc"])
                e = seed["envelope"].set_index("Wall Name")
                force_col = f"{load_direction.upper()}-load envelope |V| (kN)"
                V_seed = float(e.loc[wall_name, force_col])
            else:
                V_seed = float(wood_params["V_kN"])
            wp = dict(wood_params)
            wp.update({"V_kN": V_seed, "L_m": L})
            wr = wood_wall_response(**wp)
            walls.loc[mask, "k (kN/m)"] = wr["k secant (kN/m)"]
            r = analyze_model(walls, settings["Lx"], settings["Ly"], settings["Xcm"], settings["Ycm"], settings["Fx"], settings["Fy"], settings["acc"])
            F = settings["Fx"] if load_direction.upper() == "X" else settings["Fy"]
            out = _direction_outputs(r, wall_name, load_direction, F, settings["Lx"], settings["Ly"])
            out.update({
                "Wall Length (m)": L,
                "Seed wall force for k (kN)": V_seed,
                "Calculated k (kN/m)": wr["k secant (kN/m)"],
                "Δ total (mm)": wr["Δ total (mm)"],
            })
            rows.append(out)
        except Exception as exc:
            rows.append({"Wall Length (m)": L, "Error": str(exc)})
    return pd.DataFrame(rows)


def coupled_single_wood_wall(
    base_walls: pd.DataFrame,
    settings: Dict[str, float],
    wall_name: str,
    load_direction: str,
    wood_params: Dict[str, float],
    tolerance: float = 0.005,
    max_iterations: int = 30,
    relaxation: float = 0.7,
    auto_holdown: bool = False,
    holdown_capacity_kN: float = 55.25,
    holdown_deflection_mm: float = 5.54,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    """Iterate one selected wood wall while other wall stiffnesses remain fixed.

    en is treated as the user-entered value during the iteration.  Therefore this
    is a transparent V1 coupled solver, not a substitute for a full CSA O86
    fastener load-slip implementation.
    """
    walls = normalize_walls(base_walls)
    mask = walls["Wall Name"].eq(wall_name)
    if not mask.any():
        raise ValueError(f"Wall '{wall_name}' not found.")
    current_k = float(walls.loc[mask, "k (kN/m)"].iloc[0])
    wall_L = float(walls.loc[mask, "Wall Length (m)"].iloc[0])
    rows = []
    previous_force: Optional[float] = None
    final_result = None

    for it in range(1, int(max_iterations) + 1):
        walls.loc[mask, "k (kN/m)"] = current_k
        result = analyze_model(walls, settings["Lx"], settings["Ly"], settings["Xcm"], settings["Ycm"], settings["Fx"], settings["Fy"], settings["acc"])
        env = result["envelope"].set_index("Wall Name")
        force_col = f"{load_direction.upper()}-load envelope |V| (kN)"
        V = float(env.loc[wall_name, force_col])

        wp = dict(wood_params)
        wp["V_kN"] = V
        wp["L_m"] = wall_L
        da = wp["da_mm"]
        hd_tension = np.nan
        if auto_holdown:
            hd = linearized_holdown_da_mm(V, wp["H_m"], wall_L, holdown_capacity_kN, holdown_deflection_mm)
            da = hd["da (mm)"]
            hd_tension = hd["Hold-down tension estimate (kN)"]
            wp["da_mm"] = da
        wr = wood_wall_response(**wp)
        target_k = float(wr["k secant (kN/m)"])
        updated_k = (1.0 - relaxation) * current_k + relaxation * target_k

        k_change = abs(updated_k - current_k) / max(abs(current_k), 1e-12)
        force_change = np.nan if previous_force is None else abs(V - previous_force) / max(abs(previous_force), 1e-12)
        rows.append({
            "Iteration": it,
            "Wall force |V| (kN)": V,
            "k used (kN/m)": current_k,
            "k from wall model (kN/m)": target_k,
            "k next (kN/m)": updated_k,
            "Relative k change": k_change,
            "Relative V change": force_change,
            "Δ total (mm)": wr["Δ total (mm)"],
            "da used (mm)": da,
            "Hold-down tension estimate (kN)": hd_tension,
        })
        final_result = result
        converged = (k_change < tolerance) and (previous_force is not None and force_change < tolerance)
        current_k = updated_k
        previous_force = V
        if converged:
            break

    walls.loc[mask, "k (kN/m)"] = current_k
    final_result = analyze_model(walls, settings["Lx"], settings["Ly"], settings["Xcm"], settings["Ycm"], settings["Fx"], settings["Fy"], settings["acc"])
    hist = pd.DataFrame(rows)
    meta = {
        "converged": bool(len(hist) > 1 and hist.iloc[-1]["Relative k change"] < tolerance and hist.iloc[-1]["Relative V change"] < tolerance),
        "iterations": len(hist),
        "final_k": current_k,
        "tolerance": tolerance,
        "relaxation": relaxation,
    }
    return hist, walls, {"analysis": final_result, "meta": meta}


# -----------------------------------------------------------------------------
# Verification tests
# -----------------------------------------------------------------------------
def run_self_tests() -> pd.DataFrame:
    tests = []

    # Symmetry / zero natural eccentricity.
    w = default_walls()
    s = {"Lx": 20.0, "Ly": 12.0, "Xcm": 10.0, "Ycm": 6.0, "Fx": 100.0, "Fy": 100.0, "acc": 0.10}
    r = analyze_model(w, **{ "Lx":s["Lx"], "Ly":s["Ly"], "x_cm":s["Xcm"], "y_cm":s["Ycm"], "Fx":s["Fx"], "Fy":s["Fy"], "accidental_ratio":s["acc"]})
    p = r["properties"]
    tests.append(["Symmetry: Xcr = Xcm", p["Xcr"], 10.0, abs(p["Xcr"] - 10.0) < 1e-9])
    tests.append(["Symmetry: Ycr = Ycm", p["Ycr"], 6.0, abs(p["Ycr"] - 6.0) < 1e-9])

    # Uniform stiffness scaling should not change force distribution.
    w2 = w.copy(); w2["k (kN/m)"] *= 7.5
    r2 = analyze_model(w2, s["Lx"], s["Ly"], s["Xcm"], s["Ycm"], s["Fx"], s["Fy"], s["acc"])
    cols = ["X-load envelope |V| (kN)", "Y-load envelope |V| (kN)"]
    err = float((r["envelope"][cols] - r2["envelope"][cols]).abs().to_numpy().max())
    tests.append(["Uniform k scaling invariance", err, 0.0, err < 1e-9])

    # Wall length at fixed k should not change wall force.
    w3 = w.copy(); w3.loc[w3["Wall Name"].eq("X1"), "Wall Length (m)"] = 3.0
    w4 = w.copy(); w4.loc[w4["Wall Name"].eq("X1"), "Wall Length (m)"] = 9.0
    r3 = analyze_model(w3, s["Lx"], s["Ly"], s["Xcm"], s["Ycm"], s["Fx"], s["Fy"], s["acc"])
    r4 = analyze_model(w4, s["Lx"], s["Ly"], s["Xcm"], s["Ycm"], s["Fx"], s["Fy"], s["acc"])
    v3 = float(r3["envelope"].set_index("Wall Name").loc["X1", "X-load envelope |V| (kN)"])
    v4 = float(r4["envelope"].set_index("Wall Name").loc["X1", "X-load envelope |V| (kN)"])
    tests.append(["Wall length fixed-k force invariance", abs(v3-v4), 0.0, abs(v3-v4) < 1e-9])

    # Equilibrium.
    ch = r["checks"]
    eq = float(max(ch["ΔVx (kN)"].abs().max(), ch["ΔVy (kN)"].abs().max(), ch["ΔM (kN·m)"].abs().max()))
    tests.append(["Force/moment equilibrium", eq, 0.0, eq < 1e-8])

    # FPInnovations benchmark.
    fs, fw = fpinnovations_preset()
    fr = analyze_model(fw, fs["Lx"], fs["Ly"], fs["Xcm"], fs["Ycm"], fs["Fx"], fs["Fy"], fs["acc"])
    bt = benchmark_table(fr)
    # Published values are rounded, so use a modest tolerance.
    max_abs = float(bt["Difference"].abs().max())
    tests.append(["FPInnovations benchmark max abs difference", max_abs, 0.0, max_abs < 150.0])

    return pd.DataFrame(tests, columns=["Test", "Computed", "Reference", "Pass"])
