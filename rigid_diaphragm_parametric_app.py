from __future__ import annotations

"""
Rigid Diaphragm Parametric Analysis — Streamlit

Purpose
-------
A transparent rigid-diaphragm force-distribution tool based on the stiffness/
torsion formulation used in the FPInnovations/CWC example:

    V_i = F * k_i / sum(k) + T * k_i * d_i / J
    J   = sum(k_x * y_bar^2) + sum(k_y * x_bar^2)

The program analyzes X and Y loading as separate directional load cases,
explicitly evaluates both + and - accidental eccentricity, and envelopes the
wall force for each direction.

Important modelling note
------------------------
Wall stiffness k is an INPUT in this version. This is deliberate: the referenced
FPInnovations example determines wood-wall stiffness from the full wall
deflection expression (including framing, sheathing, fastener slip and anchorage)
and obtains the steel-frame stiffness from a separate frame analysis. A simple
height/length proxy is not equivalent to that method.
"""

from io import BytesIO
from typing import Dict, Tuple

import math
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Core mechanics
# -----------------------------------------------------------------------------
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
    """Clean and validate the wall table."""
    df = walls.copy()
    missing = [c for c in REQUIRED_WALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing wall columns: {missing}")

    df["Wall Name"] = df["Wall Name"].astype(str).str.strip()
    df = df[df["Wall Name"] != ""].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("Add at least one wall.")

    df["Direction"] = df["Direction"].astype(str).str.upper().str.strip()
    # Friendly aliases
    df["Direction"] = df["Direction"].replace({"EW": "X", "E-W": "X", "NS": "Y", "N-S": "Y"})
    bad_dir = df.loc[~df["Direction"].isin(["X", "Y"]), "Wall Name"].tolist()
    if bad_dir:
        raise ValueError(f"Direction must be X or Y for: {bad_dir}")

    for c in ["x (m)", "y (m)", "k (kN/m)", "Wall Length (m)", "Local X Force (kN)", "Local Y Force (kN)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["x (m)", "y (m)", "k (kN/m)", "Wall Length (m)", "Local X Force (kN)", "Local Y Force (kN)"]].isna().any().any():
        raise ValueError("Wall coordinates, stiffness, wall length and local wall forces must be numeric.")
    if (df["k (kN/m)"] <= 0).any():
        bad = df.loc[df["k (kN/m)"] <= 0, "Wall Name"].tolist()
        raise ValueError(f"Wall stiffness must be > 0 for: {bad}")
    if (df["Wall Length (m)"] <= 0).any():
        bad = df.loc[df["Wall Length (m)"] <= 0, "Wall Name"].tolist()
        raise ValueError(f"Wall length must be > 0 for: {bad}")
    return df


def diaphragm_properties(
    walls: pd.DataFrame,
    x_cm: float,
    y_cm: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute center of rigidity, relative coordinates and torsional rigidity J."""
    df = normalize_walls(walls)
    df["kx (kN/m)"] = np.where(df["Direction"].eq("X"), df["k (kN/m)"], 0.0)
    df["ky (kN/m)"] = np.where(df["Direction"].eq("Y"), df["k (kN/m)"], 0.0)

    sum_kx = float(df["kx (kN/m)"].sum())
    sum_ky = float(df["ky (kN/m)"].sum())
    if sum_kx <= 0:
        raise ValueError("At least one X-direction (EW) wall is required for a 2D rigid-diaphragm analysis.")
    if sum_ky <= 0:
        raise ValueError("At least one Y-direction (NS) wall is required for a 2D rigid-diaphragm analysis.")

    # X-coordinate of CoR is governed by Y-resisting elements; Y-coordinate by X-resisting elements.
    x_cr = float((df["ky (kN/m)"] * df["x (m)"]).sum() / sum_ky)
    y_cr = float((df["kx (kN/m)"] * df["y (m)"]).sum() / sum_kx)

    df["xbar (m)"] = df["x (m)"] - x_cr
    df["ybar (m)"] = df["y (m)"] - y_cr
    df["ky*xbar^2 (kN·m)"] = df["ky (kN/m)"] * df["xbar (m)"] ** 2
    df["kx*ybar^2 (kN·m)"] = df["kx (kN/m)"] * df["ybar (m)"] ** 2
    J = float(df["ky*xbar^2 (kN·m)"].sum() + df["kx*ybar^2 (kN·m)"].sum())
    if math.isclose(J, 0.0, abs_tol=1e-12):
        raise ValueError("Torsional rigidity J is zero. Check the wall layout; walls need lever arms about the CoR.")

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
    """
    Analyze one applied-force direction for +/- accidental eccentricity.

    Sign convention:
      +X right, +Y up, +Mz counter-clockwise.
      Mz = x*Fy - y*Fx.

    A positive diaphragm rotation produces point displacement
      ux_tor = -theta*ybar, uy_tor = +theta*xbar,
    hence torsional wall-force terms are
      Vx_tor = -Mz*kx*ybar/J
      Vy_tor = +Mz*ky*xbar/J.
    """
    d = direction.upper()
    if d not in {"X", "Y"}:
        raise ValueError("direction must be X or Y")

    df = walls_with_props.copy()
    F = float(force_kN)
    J = props["J"]

    if d == "X":
        e_real = props["Ycm"] - props["Ycr"]
        e_acc = accidental_ratio * Ly
        # Mz from +Fx applied at y = Ycm relative to Ycr: Mz = -ey*Fx
        def moment(e_total: float) -> float:
            return -F * e_total
        direct_x = F * df["kx (kN/m)"] / props["sum_kx"]
        direct_y = pd.Series(0.0, index=df.index)
        local_parallel = np.where(df["Direction"].eq("X"), df["Local X Force (kN)"], 0.0)
    else:
        e_real = props["Xcm"] - props["Xcr"]
        e_acc = accidental_ratio * Lx
        # Mz from +Fy applied at x = Xcm relative to Xcr: Mz = +ex*Fy
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

        # Equilibrium about the center of rigidity.
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

    # Envelope the actual wall-parallel force across +/- accidental eccentricity.
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
    x_cases, x_env, x_checks, x_meta = analyze_direction(
        df, props, "X", Fx, Lx, Ly, accidental_ratio
    )
    y_cases, y_env, y_checks, y_meta = analyze_direction(
        df, props, "Y", Fy, Lx, Ly, accidental_ratio
    )

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
    env["Max directional envelope |V| (kN)"] = env[
        ["X-load envelope |V| (kN)", "Y-load envelope |V| (kN)"]
    ].max(axis=1)

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
# Reference presets / validation
# -----------------------------------------------------------------------------
def fpinnovations_preset() -> Tuple[Dict[str, float], pd.DataFrame]:
    """Geometry/stiffness benchmark from pages 9–11 of the uploaded example."""
    settings = {
        "Lx": 30.5,
        "Ly": 12.2,
        "Xcm": 15.25,
        "Ycm": 6.10,
        # Diaphragm-transferred N-S force used in the source torsion calculation:
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
    # Iteration-1 wall-force checks from the published example.
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
# UI helpers
# -----------------------------------------------------------------------------

def inject_professional_css() -> None:
    st.markdown(
        """
        <style>
        html, body, [class*="css"], .stApp, .stMarkdown, .stText, .stDataFrame,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .stTabs [data-baseweb="tab"], label, input, textarea, select, button {
            font-family: Arial, Helvetica, sans-serif !important;
        }

        .block-container {
            padding-top: 1.1rem;
            padding-bottom: 2.0rem;
        }

        h1, h2, h3 {
            font-family: Arial, Helvetica, sans-serif !important;
            color: #1f2937;
            letter-spacing: -0.01em;
        }

        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            padding: 0.70rem 0.90rem;
        }

        [data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #4b5563;
        }

        [data-testid="stMetricValue"] {
            color: #111827;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        div[data-testid="stDataEditor"], div[data-testid="stDataFrame"] {
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stExpander"] details {
            border: 1px solid #d9e2ec;
            border-radius: 12px;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def default_walls() -> pd.DataFrame:
    return pd.DataFrame([
        {"Wall Name": "X1", "Direction": "X", "x (m)": 5.0,  "y (m)": 0.0,  "k (kN/m)": 10000.0, "Wall Length (m)": 8.0, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
        {"Wall Name": "X2", "Direction": "X", "x (m)": 15.0, "y (m)": 12.0, "k (kN/m)": 10000.0, "Wall Length (m)": 8.0, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
        {"Wall Name": "Y1", "Direction": "Y", "x (m)": 0.0,  "y (m)": 6.0,  "k (kN/m)": 10000.0, "Wall Length (m)": 8.0, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
        {"Wall Name": "Y2", "Direction": "Y", "x (m)": 20.0, "y (m)": 6.0,  "k (kN/m)": 10000.0, "Wall Length (m)": 8.0, "Local X Force (kN)": 0.0, "Local Y Force (kN)": 0.0},
    ])


def draw_plan(walls: pd.DataFrame, Lx: float, Ly: float, props: Dict[str, float]):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfdff")

    # Plan outline
    outline = Rectangle((0.0, 0.0), Lx, Ly, fill=False, edgecolor="#334155", linewidth=1.8)
    ax.add_patch(outline)

    wall_colors = {"X": "#1d4ed8", "Y": "#b45309"}
    text_offset_x = 0.012 * max(Lx, 1.0)
    text_offset_y = 0.018 * max(Ly, 1.0)

    for _, r in walls.iterrows():
        x = float(r["x (m)"])
        y = float(r["y (m)"])
        length = float(r["Wall Length (m)"])
        direction = str(r["Direction"])
        color = wall_colors.get(direction, "#374151")

        if direction == "X":
            x0 = max(0.0, x - length / 2.0)
            x1 = min(Lx, x + length / 2.0)
            ax.plot([x0, x1], [y, y], color=color, linewidth=5.0, solid_capstyle="butt", zorder=3)
            ax.plot(x, y, marker="o", color=color, markersize=3.5, zorder=4)
            label_y = y + text_offset_y
            ha, va = "center", "bottom"
        else:
            y0 = max(0.0, y - length / 2.0)
            y1 = min(Ly, y + length / 2.0)
            ax.plot([x, x], [y0, y1], color=color, linewidth=5.0, solid_capstyle="butt", zorder=3)
            ax.plot(x, y, marker="o", color=color, markersize=3.5, zorder=4)
            label_y = y + text_offset_y
            ha, va = "left", "bottom"

        label = f"{r['Wall Name']} ({direction})\nL={length:.2f} m"
        ax.text(
            x + (text_offset_x if direction == "Y" else 0.0),
            label_y,
            label,
            fontsize=9,
            ha=ha,
            va=va,
            color="#111827",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=5,
        )

    # CM and CR
    ax.scatter([props["Xcm"]], [props["Ycm"]], marker="o", s=70, color="#16a34a", zorder=6)
    ax.scatter([props["Xcr"]], [props["Ycr"]], marker="X", s=95, color="#dc2626", zorder=6)
    ax.text(props["Xcm"] + text_offset_x, props["Ycm"] + text_offset_y, "CM", color="#166534", fontsize=10, weight="bold")
    ax.text(props["Xcr"] + text_offset_x, props["Ycr"] - text_offset_y, "CR", color="#991b1b", fontsize=10, weight="bold")

    # Eccentricity guide from CR to CM
    ax.annotate(
        "",
        xy=(props["Xcm"], props["Ycm"]),
        xytext=(props["Xcr"], props["Ycr"]),
        arrowprops=dict(arrowstyle="->", linestyle="--", linewidth=1.4, color="#64748b"),
        zorder=2,
    )

    ax.set_xlim(-0.06 * Lx, 1.06 * Lx)
    ax.set_ylim(-0.10 * Ly, 1.12 * Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / E-W (m)", fontsize=10)
    ax.set_ylabel("Y / N-S (m)", fontsize=10)
    ax.set_title("Plan layout, wall locations, wall lengths, CM and CR", fontsize=12, weight="bold", pad=10)
    ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.6)

    legend_handles = [
        Line2D([0], [0], color=wall_colors["X"], linewidth=5.0, label="X-direction wall"),
        Line2D([0], [0], color=wall_colors["Y"], linewidth=5.0, label="Y-direction wall"),
        Line2D([0], [0], marker="o", linestyle="", color="#16a34a", markersize=8, label="Center of mass (CM)"),
        Line2D([0], [0], marker="X", linestyle="", color="#dc2626", markersize=8, label="Center of rigidity (CR)"),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def to_excel_bytes(result: Dict[str, object]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([result["properties"]]).to_excel(writer, sheet_name="Key Properties", index=False)
        result["wall_properties"].to_excel(writer, sheet_name="Wall Properties", index=False)
        result["envelope"].to_excel(writer, sheet_name="Envelope", index=False)
        result["x_cases"].to_excel(writer, sheet_name="X Cases", index=False)
        result["y_cases"].to_excel(writer, sheet_name="Y Cases", index=False)
        result["checks"].to_excel(writer, sheet_name="Equilibrium Checks", index=False)
    return bio.getvalue()


def run_sweep(
    base_walls: pd.DataFrame,
    settings: Dict[str, float],
    wall_name: str,
    parameter: str,
    start: float,
    stop: float,
    steps: int,
) -> pd.DataFrame:
    values = np.linspace(start, stop, steps)
    rows = []
    for val in values:
        walls = base_walls.copy()
        mask = walls["Wall Name"].eq(wall_name)
        if parameter == "x coordinate (m)":
            walls.loc[mask, "x (m)"] = val
        elif parameter == "y coordinate (m)":
            walls.loc[mask, "y (m)"] = val
        elif parameter == "stiffness multiplier":
            walls.loc[mask, "k (kN/m)"] = base_walls.loc[mask, "k (kN/m)"].iloc[0] * val
        else:
            raise ValueError(parameter)

        try:
            r = analyze_model(
                walls,
                settings["Lx"], settings["Ly"], settings["Xcm"], settings["Ycm"],
                settings["Fx"], settings["Fy"], settings["acc"],
            )
            env = r["envelope"]
            row = env.loc[env["Wall Name"].eq(wall_name)].iloc[0]
            rows.append({
                "Parameter": val,
                "X-load envelope |V| (kN)": row["X-load envelope |V| (kN)"],
                "Y-load envelope |V| (kN)": row["Y-load envelope |V| (kN)"],
                "Max wall force in model (kN)": env["Max directional envelope |V| (kN)"].max(),
                "Xcr (m)": r["properties"]["Xcr"],
                "Ycr (m)": r["properties"]["Ycr"],
            })
        except Exception:
            rows.append({"Parameter": val})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Streamlit app
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Rigid Diaphragm Parametric Analysis", layout="wide")
    inject_professional_css()
    st.title("Rigid Diaphragm Parametric Analysis")
    st.caption("Stiffness-based force distribution with signed natural eccentricity, ± accidental eccentricity, torsion to all resisting lines, and equilibrium checks.")

    with st.expander("Method and scope", expanded=False):
        st.markdown(
            """
This app treats the diaphragm as rigid in-plane and the vertical resisting elements as translational springs.
For each X or Y load direction it computes the center of rigidity, the torsional rigidity `J`, direct stiffness-proportional shear, and torsional shear. Both signs of accidental eccentricity are solved explicitly and each wall is enveloped.

**Use actual wall/frame stiffness `k` whenever possible.** The FPInnovations example derives wood shear-wall stiffness from the wall deformation equation and obtains the steel-frame stiffness from a 2D frame analysis. The older script's geometry-only `1/Di` proxy is therefore not used here.
            """
        )

    # Persistent defaults
    if "walls" not in st.session_state:
        st.session_state.walls = default_walls()
    if "preset" not in st.session_state:
        st.session_state.preset = "Custom"

    cA, cB, cC = st.columns([1.2, 1.2, 2.0])
    with cA:
        if st.button("Load FPInnovations benchmark", use_container_width=True):
            s, w = fpinnovations_preset()
            st.session_state.walls = w
            st.session_state.fp_settings = s
            st.session_state.preset = "FP"
            st.rerun()
    with cB:
        if st.button("Reset simple example", use_container_width=True):
            st.session_state.walls = default_walls()
            st.session_state.pop("fp_settings", None)
            st.session_state.preset = "Custom"
            st.rerun()
    with cC:
        st.info("Recommended workflow: input the plan, CM, diaphragm forces, and actual wall/frame stiffnesses; then move walls or vary stiffness in the sweep tab.")

    fp = st.session_state.get("fp_settings", {})
    d_Lx = float(fp.get("Lx", 20.0))
    d_Ly = float(fp.get("Ly", 12.0))
    d_Xcm = float(fp.get("Xcm", d_Lx / 2))
    d_Ycm = float(fp.get("Ycm", d_Ly / 2))
    d_Fx = float(fp.get("Fx", 100.0))
    d_Fy = float(fp.get("Fy", 100.0))
    d_acc = float(fp.get("acc", 0.10))

    st.subheader("1. Diaphragm and load inputs")
    r1 = st.columns(4)
    Lx = r1[0].number_input("Lx — X/E-W plan dimension (m)", min_value=0.01, value=d_Lx, step=0.1)
    Ly = r1[1].number_input("Ly — Y/N-S plan dimension (m)", min_value=0.01, value=d_Ly, step=0.1)
    Xcm = r1[2].number_input("Xcm (m)", value=d_Xcm, step=0.05)
    Ycm = r1[3].number_input("Ycm (m)", value=d_Ycm, step=0.05)

    r2 = st.columns(4)
    Fx = r2[0].number_input("Fx — diaphragm force in X (kN)", value=d_Fx, step=1.0)
    Fy = r2[1].number_input("Fy — diaphragm force in Y (kN)", value=d_Fy, step=1.0)
    acc = r2[2].number_input("Accidental eccentricity ratio", min_value=0.0, value=d_acc, step=0.01, format="%.3f")
    with r2[3]:
        st.markdown("**Accidental offsets**")
        st.caption(f"X-load offset = ±{acc*Ly:.3f} m")
        st.caption(f"Y-load offset = ±{acc*Lx:.3f} m")

    st.subheader("2. Wall / frame table")
    st.caption("Direction X = E-W wall resisting X force; Direction Y = N-S wall resisting Y force. Coordinates locate the resisting line. k is lateral stiffness in kN/m. Optional local wall forces are loads resisted directly by that wall (e.g., its own inertial wall weight) and are added after diaphragm distribution.")
    edited = st.data_editor(
        st.session_state.walls,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Wall Name": st.column_config.TextColumn(required=True),
            "Direction": st.column_config.SelectboxColumn(options=["X", "Y"], required=True),
            "x (m)": st.column_config.NumberColumn(format="%.3f"),
            "y (m)": st.column_config.NumberColumn(format="%.3f"),
            "k (kN/m)": st.column_config.NumberColumn(format="%.2f", min_value=0.0001),
            "Wall Length (m)": st.column_config.NumberColumn(format="%.3f", min_value=0.0001),
            "Local X Force (kN)": st.column_config.NumberColumn(format="%.3f"),
            "Local Y Force (kN)": st.column_config.NumberColumn(format="%.3f"),
        },
        key="wall_editor",
    )
    st.session_state.walls = edited

    settings = {"Lx": Lx, "Ly": Ly, "Xcm": Xcm, "Ycm": Ycm, "Fx": Fx, "Fy": Fy, "acc": acc}

    try:
        result = analyze_model(edited, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    p = result["properties"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Xcr", f"{p['Xcr']:.3f} m")
    k2.metric("Ycr", f"{p['Ycr']:.3f} m")
    k3.metric("ex = Xcm−Xcr", f"{p['ex_signed']:.3f} m")
    k4.metric("ey = Ycm−Ycr", f"{p['ey_signed']:.3f} m")
    k5.metric("J", f"{p['J']:,.1f} kN·m")
    k6.metric("Σk", f"X {p['sum_kx']:,.0f} | Y {p['sum_ky']:,.0f} kN/m")

    tabs = st.tabs([
        "Wall force envelope",
        "Plan / CoR",
        "Load cases",
        "Equilibrium checks",
        "Parametric sweep",
        "Reference benchmark",
        "Downloads",
    ])

    with tabs[0]:
        env = result["envelope"].copy()
        num = env.select_dtypes(include=[np.number]).columns
        env[num] = env[num].round(3)
        st.dataframe(env, hide_index=True, use_container_width=True)
        st.caption("The X-load and Y-load envelopes are separate directional results. 'Max directional envelope' is a comparison aid, not an orthogonal-load combination rule.")

    with tabs[1]:
        props_df = result["wall_properties"].copy()
        try:
            fig = draw_plan(props_df, Lx, Ly, p)
            st.pyplot(fig, clear_figure=True)
        except ModuleNotFoundError as exc:
            if exc.name == "matplotlib":
                st.warning(
                    "Plan graphic requires matplotlib. On Streamlit Cloud, make sure the repository contains "
                    "a file named exactly `requirements.txt` with `matplotlib>=3.7`, then reboot the app. "
                    "The engineering calculations and tables below are still available."
                )
            else:
                raise
        st.dataframe(
            props_df[["Wall Name", "Direction", "x (m)", "y (m)", "k (kN/m)", "xbar (m)", "ybar (m)", "ky*xbar^2 (kN·m)", "kx*ybar^2 (kN·m)"]].round(3),
            hide_index=True,
            use_container_width=True,
        )

    with tabs[2]:
        dsel = st.radio("Load direction", ["X", "Y"], horizontal=True)
        cases = result["x_cases"] if dsel == "X" else result["y_cases"]
        st.dataframe(cases.round(4), hide_index=True, use_container_width=True, height=500)

    with tabs[3]:
        checks = result["checks"].copy()
        st.dataframe(checks.round(8), hide_index=True, use_container_width=True)
        max_err = max(
            checks["ΔVx (kN)"].abs().max(),
            checks["ΔVy (kN)"].abs().max(),
            checks["ΔM (kN·m)"].abs().max(),
        )
        if max_err < 1e-6:
            st.success("All force and moment equilibrium checks close to numerical precision.")
        else:
            st.warning(f"Equilibrium residual detected: max residual = {max_err:.6g}")

    with tabs[4]:
        names = normalize_walls(edited)["Wall Name"].tolist()
        c1, c2 = st.columns(2)
        wall_name = c1.selectbox("Wall to vary", names)
        parameter = c2.selectbox("Parameter", ["x coordinate (m)", "y coordinate (m)", "stiffness multiplier"])

        current_row = normalize_walls(edited).loc[lambda d: d["Wall Name"].eq(wall_name)].iloc[0]
        if parameter == "x coordinate (m)":
            default_start, default_stop = 0.0, Lx
        elif parameter == "y coordinate (m)":
            default_start, default_stop = 0.0, Ly
        else:
            default_start, default_stop = 0.5, 2.0

        c3, c4, c5 = st.columns(3)
        start = c3.number_input("Sweep start", value=float(default_start))
        stop = c4.number_input("Sweep stop", value=float(default_stop))
        steps = c5.number_input("Number of steps", min_value=3, max_value=201, value=31, step=1)
        sweep = run_sweep(normalize_walls(edited), settings, wall_name, parameter, start, stop, int(steps))
        st.line_chart(sweep.set_index("Parameter")[["X-load envelope |V| (kN)", "Y-load envelope |V| (kN)", "Max wall force in model (kN)"]])
        st.dataframe(sweep.round(4), hide_index=True, use_container_width=True, height=350)

    with tabs[5]:
        if st.session_state.preset == "FP":
            bench = benchmark_table(result)
            st.dataframe(bench.round(3), hide_index=True, use_container_width=True)
            st.markdown(
                """
The benchmark checks the published CoR, J, torsion and the first-pass wall forces. The optional local wall-force columns reproduce the example's treatment of wall inertial forces that are resisted directly by parallel walls rather than transferred through the diaphragm. The paper then changes wall construction and repeats the stiffness/force calculation to convergence, so its later iterations require updated k values.
                """
            )
        else:
            st.info("Click 'Load FPInnovations benchmark' above to populate the published geometry and stiffness values and see the validation checks.")

    with tabs[6]:
        xlsx = to_excel_bytes(result)
        st.download_button(
            "Download full analysis workbook",
            xlsx,
            file_name="rigid_diaphragm_parametric_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.download_button(
            "Download wall envelope CSV",
            result["envelope"].to_csv(index=False).encode("utf-8"),
            file_name="rigid_diaphragm_wall_envelope.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption("Engineering-use note: verify stiffness assumptions, applicable code eccentricity/load-combination requirements, and final design independently before using results for construction documents.")


if __name__ == "__main__":
    main()
