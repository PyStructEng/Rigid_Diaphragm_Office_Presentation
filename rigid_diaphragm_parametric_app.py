from __future__ import annotations

"""Rigid Diaphragm Research Lab - Streamlit front end.

Presentation + parametric-research interface backed by rigid_diaphragm_core.py.
"""

from io import BytesIO
import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from diaphragm_calculation_sheet import render_calculation_tab

from rigid_diaphragm_core import (
    analyze_model,
    benchmark_table,
    coupled_single_wood_wall,
    default_walls,
    fpinnovations_preset,
    linearized_holdown_da_mm,
    length_proportional_stiffness_model,
    normalize_walls,
    run_geometry_study,
    run_interaction_study,
    run_length_calculated_k_building_study,
    run_length_fixed_k_study,
    run_self_tests,
    run_stiffness_study,
    run_wood_length_study,
    run_wood_parameter_study,
    wood_wall_response,
)

from wood_shearwall_mechanics import (
    DATABASES,
    SHEATHING_BV,
    LUMBER_E,
    TAKEUP_DEVICES,
    ROD_GEOMETRY,
    STUD_DEPTH_MM,
    DEFAULT_CAVITY_MM,
    DEFAULT_COMPRESSION_BEARING_LENGTH_MM,
    default_storeys,
    normalize_storeys,
    analyze_stacked_wall,
    transformed_section,
    get_bv,
    get_lumber_e,
    get_rod_properties,
    get_takeup,
    calculate_lc_mm,
    run_mechanics_self_tests,
)


# -----------------------------------------------------------------------------
# Styling / plotting
# -----------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        html, body, .stApp, [class*="css"], .stMarkdown, .stText,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .stTabs [data-baseweb="tab"], label, input, textarea, select, button {
            font-family: Arial, Helvetica, sans-serif !important;
        }
        .block-container { padding-top: 1.0rem; padding-bottom: 2rem; max-width: 1700px; }
        h1, h2, h3 { font-family: Arial, Helvetica, sans-serif !important; color:#1f2937; letter-spacing:-0.015em; }
        div[data-testid="stMetric"] { background:#f8fafc; border:1px solid #dbe3ea; border-radius:12px; padding:0.72rem 0.9rem; }
        [data-testid="stMetricLabel"] { font-weight:600; color:#475569; }
        [data-testid="stMetricValue"] { color:#111827; }
        div[data-testid="stDataEditor"], div[data-testid="stDataFrame"] { border:1px solid #dbe3ea; border-radius:12px; overflow:hidden; }
        div[data-testid="stExpander"] details { border:1px solid #dbe3ea; border-radius:12px; background:#ffffff; }
        .small-note { color:#64748b; font-size:0.88rem; }
        .hero-note { background:#f8fafc; border-left:4px solid #334155; padding:0.7rem 0.9rem; border-radius:7px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _mpl():
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
    return plt


def draw_plan_with_forces(
    walls: pd.DataFrame,
    Lx: float,
    Ly: float,
    props: Dict[str, float],
    envelope: pd.DataFrame,
    load_direction: str,
    show_forces: bool = True,
    selected_wall: str | None = None,
    case_label: str = "governing envelope",
    figsize: tuple[float, float] = (10.6, 6.4),
):
    plt = _mpl()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    walls = normalize_walls(walls)
    env = envelope.set_index("Wall Name")
    d = load_direction.upper()
    signed_col = f"{d}-load governing signed V (kN)"
    abs_col = f"{d}-load envelope |V| (kN)"
    max_force = max(float(env[abs_col].max()), 1e-9)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfdff")
    ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, edgecolor="#334155", linewidth=1.8))

    wall_colors = {"X": "#1d4ed8", "Y": "#b45309"}
    ox = 0.012 * max(Lx, 1.0)
    oy = 0.020 * max(Ly, 1.0)

    for _, r in walls.iterrows():
        name = str(r["Wall Name"])
        direction = str(r["Direction"])
        x, y = float(r["x (m)"]), float(r["y (m)"])
        length = float(r["Wall Length (m)"])
        stiffness = float(r["k (kN/m)"])
        color = wall_colors[direction]
        is_selected = selected_wall is not None and name == selected_wall
        wall_lw = 8.0 if is_selected else 5.0

        # Actual wall length, centered on the resisting-line coordinate and clipped to plan.
        if direction == "X":
            x0, x1 = max(0.0, x - length / 2), min(Lx, x + length / 2)
            ax.plot([x0, x1], [y, y], linewidth=wall_lw, color=color, solid_capstyle="butt", zorder=3)
        else:
            y0, y1 = max(0.0, y - length / 2), min(Ly, y + length / 2)
            ax.plot([x, x], [y0, y1], linewidth=wall_lw, color=color, solid_capstyle="butt", zorder=3)
        ax.plot(x, y, marker="o", markersize=3.5, color=color, zorder=4)

        V = float(env.loc[name, signed_col]) if name in env.index else 0.0
        label = f"{name}  L={length:.2f} m\nk={stiffness:,.0f} kN/m"
        if show_forces:
            label += f"  V={V:+.1f} kN"
        ax.text(
            x + (ox if direction == "Y" else 0), y + oy, label,
            fontsize=8.6, ha="left" if direction == "Y" else "center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.88),
            zorder=7,
        )

        if show_forces and abs(V) > 1e-12:
            frac = 0.18 + 0.82 * abs(V) / max_force
            if direction == "X":
                arrow_len = frac * 0.13 * Lx
                sgn = 1 if V >= 0 else -1
                ax.annotate(
                    "", xy=(x + sgn * arrow_len, y - 0.025 * Ly), xytext=(x, y - 0.025 * Ly),
                    arrowprops=dict(arrowstyle="-|>", linewidth=1.8, color=color), zorder=6,
                )
            else:
                arrow_len = frac * 0.13 * Ly
                sgn = 1 if V >= 0 else -1
                ax.annotate(
                    "", xy=(x - 0.018 * Lx, y + sgn * arrow_len), xytext=(x - 0.018 * Lx, y),
                    arrowprops=dict(arrowstyle="-|>", linewidth=1.8, color=color), zorder=6,
                )

    ax.scatter([props["Xcm"]], [props["Ycm"]], s=75, marker="o", color="#16a34a", zorder=9)
    ax.scatter([props["Xcr"]], [props["Ycr"]], s=95, marker="X", color="#dc2626", zorder=9)
    ax.text(props["Xcm"] + ox, props["Ycm"] + oy, "CM", color="#166534", weight="bold", fontsize=10)
    ax.text(props["Xcr"] + ox, props["Ycr"] - oy, "CR", color="#991b1b", weight="bold", fontsize=10)
    ax.annotate(
        "", xy=(props["Xcm"], props["Ycm"]), xytext=(props["Xcr"], props["Ycr"]),
        arrowprops=dict(arrowstyle="->", linestyle="--", linewidth=1.2, color="#64748b"), zorder=2,
    )

    ax.set_xlim(-0.06 * Lx, 1.06 * Lx)
    ax.set_ylim(-0.10 * Ly, 1.14 * Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / E-W (m)")
    ax.set_ylabel("Y / N-S (m)")
    ax.set_title(f"Rigid diaphragm - {d}-load {case_label}", fontsize=12, weight="bold", pad=10)
    ax.grid(True, linewidth=0.6, color="#cbd5e1", alpha=0.55)
    handles = [
        Line2D([0], [0], color=wall_colors["X"], linewidth=5, label="X-direction wall"),
        Line2D([0], [0], color=wall_colors["Y"], linewidth=5, label="Y-direction wall"),
        Line2D([0], [0], marker="o", linestyle="", color="#16a34a", label="CM"),
        Line2D([0], [0], marker="X", linestyle="", color="#dc2626", label="CR"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)
    fig.tight_layout()
    return fig


def draw_force_bars(envelope: pd.DataFrame, load_direction: str):
    plt = _mpl()
    d = load_direction.upper()
    col = f"{d}-load governing signed V (kN)"
    df = envelope[["Wall Name", col]].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.bar(df["Wall Name"], df[col], edgecolor="#334155", linewidth=0.7)
    ax.axhline(0, linewidth=0.9, color="#64748b")
    ax.set_ylabel("Governing signed wall force (kN)")
    ax.set_xlabel("Wall")
    ax.set_title(f"{d}-load wall-force envelope", weight="bold")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(df[col]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    return fig


def force_table_for_view(result: Dict[str, object], load_direction: str, case_mode: str) -> tuple[pd.DataFrame, str]:
    """Return a force table shaped like the envelope table for visualization.

    Envelope mode is a wall-by-wall design envelope and is not necessarily one
    simultaneous physical load case.  The +/- modes show one actual accidental-
    eccentricity case at a time.
    """
    d = load_direction.upper()
    env = result["envelope"].copy()
    signed_col = f"{d}-load governing signed V (kN)"
    abs_col = f"{d}-load envelope |V| (kN)"
    acc_col = f"{d}-load acc. sign"

    if case_mode.startswith("Envelope"):
        return env, "wall-by-wall design envelope"

    sign = "+" if case_mode.startswith("+") else "−"
    cases = result["x_cases"] if d == "X" else result["y_cases"]
    c = cases.loc[cases["Accidental Sign"].eq(sign), ["Wall Name", "Wall-parallel V (kN)"]].copy()
    c = c.drop_duplicates(subset=["Wall Name"]).set_index("Wall Name")
    for idx, row in env.iterrows():
        name = row["Wall Name"]
        if name in c.index:
            v = float(c.loc[name, "Wall-parallel V (kN)"])
            env.at[idx, signed_col] = v
            env.at[idx, abs_col] = abs(v)
            env.at[idx, acc_col] = sign
    return env, f"{sign} accidental-eccentricity case"


def render_model_snapshot(
    walls: pd.DataFrame,
    settings: Dict[str, float],
    result: Dict[str, object],
    load_direction: str,
    case_mode: str = "Envelope (wall-by-wall)",
    selected_wall: str | None = None,
    heading: str | None = None,
    show_force_bars: bool = False,
    compact: bool = True,
) -> None:
    """Reusable visual anchor used globally and inside study tabs."""
    if heading:
        st.markdown(f"#### {heading}")
    view_table, case_label = force_table_for_view(result, load_direction, case_mode)
    props = result["properties"]

    if show_force_bars:
        left, right = st.columns([1.55, 1.0])
        with left:
            st.pyplot(
                draw_plan_with_forces(
                    walls, settings["Lx"], settings["Ly"], props, view_table, load_direction,
                    selected_wall=selected_wall, case_label=case_label,
                    figsize=(9.4, 5.5) if compact else (10.6, 6.4),
                ),
                clear_figure=True, use_container_width=True,
            )
        with right:
            st.pyplot(draw_force_bars(view_table, load_direction), clear_figure=True, use_container_width=True)
            a, b = st.columns(2)
            a.metric("CR", f"({props['Xcr']:.2f}, {props['Ycr']:.2f}) m")
            b.metric("J", f"{props['J']:,.0f} kN·m")
            c, d = st.columns(2)
            c.metric("ex", f"{props['ex_signed']:.3f} m")
            d.metric("ey", f"{props['ey_signed']:.3f} m")
    else:
        left, right = st.columns([1.75, 0.85])
        with left:
            st.pyplot(
                draw_plan_with_forces(
                    walls, settings["Lx"], settings["Ly"], props, view_table, load_direction,
                    selected_wall=selected_wall, case_label=case_label, figsize=(8.5, 4.8),
                ),
                clear_figure=True, use_container_width=True,
            )
        with right:
            st.metric("Xcr", f"{props['Xcr']:.3f} m")
            st.metric("Ycr", f"{props['Ycr']:.3f} m")
            st.metric("J", f"{props['J']:,.0f} kN·m")
            if selected_wall:
                row = view_table.loc[view_table["Wall Name"].eq(selected_wall)]
                if not row.empty:
                    col = f"{load_direction.upper()}-load governing signed V (kN)"
                    st.metric(f"{selected_wall} force", f"{float(row.iloc[0][col]):+.2f} kN")

    if case_mode.startswith("Envelope"):
        st.caption("Wall-by-wall envelope: individual maxima can come from different ± accidental-eccentricity cases and are not necessarily simultaneous.")


def draw_line_chart(df: pd.DataFrame, x: str, ys: list[str], title: str, xlabel: str | None = None, ylabel: str | None = None, log_x: bool = False):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    for y in ys:
        if y in df.columns:
            ax.plot(df[x], df[y], linewidth=2.0, label=y)
    if log_x:
        ax.set_xscale("log")
    ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or "Value")
    ax.grid(True, alpha=0.25)
    if len(ys) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def draw_cr_trajectory(
    walls: pd.DataFrame,
    Lx: float,
    Ly: float,
    states: pd.DataFrame,
    x_cm: float,
    y_cm: float,
):
    """Plan view of the center-of-rigidity trajectory during coupled iteration."""
    plt = _mpl()
    from matplotlib.patches import Rectangle

    walls = normalize_walls(walls)
    s = states.sort_values("State").copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfdff")
    ax.add_patch(Rectangle((0, 0), Lx, Ly, fill=False, edgecolor="#475569", linewidth=1.7))

    # Faint wall layout for geometric context.
    for _, r in walls.iterrows():
        x, y = float(r["x (m)"]), float(r["y (m)"])
        length = float(r["Wall Length (m)"])
        if str(r["Direction"]) == "X":
            ax.plot([max(0.0, x-length/2), min(Lx, x+length/2)], [y, y], linewidth=3.0, alpha=0.28)
        else:
            ax.plot([x, x], [max(0.0, y-length/2), min(Ly, y+length/2)], linewidth=3.0, alpha=0.28)

    ax.plot(s["Xcr (m)"], s["Ycr (m)"], marker="o", linewidth=2.0, zorder=5, label="CR trajectory")
    ax.scatter([x_cm], [y_cm], marker="o", s=70, zorder=6, label="CM")
    ax.scatter([float(s.iloc[0]["Xcr (m)"])], [float(s.iloc[0]["Ycr (m)"])], marker="X", s=95, zorder=7, label="Initial CR")
    ax.scatter([float(s.iloc[-1]["Xcr (m)"])], [float(s.iloc[-1]["Ycr (m)"])], marker="X", s=125, zorder=8, label="Final CR")

    # Label only a modest number of states to keep the plot readable.
    every = max(1, int(math.ceil(len(s) / 8)))
    for j, (_, r) in enumerate(s.iterrows()):
        if j == 0 or j == len(s)-1 or j % every == 0:
            ax.annotate(f"{int(r['State'])}", (float(r["Xcr (m)"]), float(r["Ycr (m)"])), xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlim(-0.06*Lx, 1.06*Lx)
    ax.set_ylim(-0.08*Ly, 1.10*Ly)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X / E-W (m)")
    ax.set_ylabel("Y / N-S (m)")
    ax.set_title("Center-of-rigidity trajectory during coupled iteration", weight="bold")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def draw_before_after_force_bars(wall_states: pd.DataFrame):
    """Grouped signed wall-force comparison for state 0 and the final state."""
    plt = _mpl()
    ws = wall_states.copy()
    s0 = int(ws["State"].min())
    sf = int(ws["State"].max())
    b = ws.loc[ws["State"].eq(s0), ["Wall Name", "Signed V (kN)"]].rename(columns={"Signed V (kN)":"Initial"})
    f = ws.loc[ws["State"].eq(sf), ["Wall Name", "Signed V (kN)"]].rename(columns={"Signed V (kN)":"Final"})
    d = b.merge(f, on="Wall Name", how="outer").fillna(0.0)
    x = np.arange(len(d))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.9))
    ax.bar(x-width/2, d["Initial"], width, label="Initial")
    ax.bar(x+width/2, d["Final"], width, label="Final")
    ax.axhline(0.0, linewidth=0.9)
    ax.set_xticks(x, d["Wall Name"].astype(str))
    ax.set_xlabel("Wall")
    ax.set_ylabel("Signed wall force (kN)")
    ax.set_title("System force redistribution: initial vs converged", weight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def coupled_force_change_table(wall_states: pd.DataFrame) -> pd.DataFrame:
    """Create a compact initial/final wall-force and stiffness comparison table."""
    ws = wall_states.copy()
    s0 = int(ws["State"].min())
    sf = int(ws["State"].max())
    cols = ["Wall Name", "Direction", "k (kN/m)", "Signed V (kN)", "|V| (kN)"]
    b = ws.loc[ws["State"].eq(s0), cols].copy()
    f = ws.loc[ws["State"].eq(sf), cols].copy()
    b = b.rename(columns={
        "k (kN/m)":"Initial k (kN/m)", "Signed V (kN)":"Initial V (kN)", "|V| (kN)":"Initial |V| (kN)"
    })
    f = f.rename(columns={
        "Direction":"Direction final", "k (kN/m)":"Final k (kN/m)", "Signed V (kN)":"Final V (kN)", "|V| (kN)":"Final |V| (kN)"
    })
    out = b.merge(f, on="Wall Name", how="outer")
    out["Direction"] = out["Direction"].fillna(out.get("Direction final"))
    if "Direction final" in out.columns:
        out = out.drop(columns=["Direction final"])
    out["ΔV (kN)"] = out["Final V (kN)"] - out["Initial V (kN)"]
    denom = out["Initial |V| (kN)"].replace(0.0, np.nan)
    out["Δ|V| (%)"] = 100.0 * (out["Final |V| (kN)"] - out["Initial |V| (kN)"]) / denom
    out["Δk (%)"] = 100.0 * (out["Final k (kN/m)"] - out["Initial k (kN/m)"]) / out["Initial k (kN/m)"].replace(0.0, np.nan)
    return out


def draw_heatmap(df: pd.DataFrame, value_col: str, title: str):
    plt = _mpl()
    p = df.dropna(subset=["Normalized position", "Stiffness multiplier", value_col]).pivot(
        index="Stiffness multiplier", columns="Normalized position", values=value_col
    )
    fig, ax = plt.subplots(figsize=(9.3, 5.8))
    im = ax.imshow(
        p.values, origin="lower", aspect="auto",
        extent=[p.columns.min(), p.columns.max(), p.index.min(), p.index.max()],
        interpolation="nearest",
    )
    ax.set_xlabel("Normalized wall position")
    ax.set_ylabel("Stiffness multiplier k/k₀")
    ax.set_title(title, weight="bold")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    return fig


def draw_deformation_breakdown(response: Dict[str, float]):
    plt = _mpl()
    labels = ["Bending", "Sheathing", "Fastener slip", "Anchorage"]
    vals = [response["Δ bending (mm)"], response["Δ sheathing (mm)"], response["Δ fastener (mm)"], response["Δ anchorage (mm)"]]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.barh(labels, vals, edgecolor="#334155", linewidth=0.6)
    ax.set_xlabel("Deflection contribution (mm)")
    ax.set_title("Wood wall deformation breakdown", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    total = sum(vals)
    for i, v in enumerate(vals):
        pct = 100 * v / total if total else 0
        ax.text(v, i, f" {v:.3f} mm ({pct:.1f}%)", va="center", fontsize=9)
    fig.tight_layout()
    return fig




def draw_mechanics_breakdown(row: pd.Series | Dict[str, float]):
    plt = _mpl()
    labels = ["Bending", "Panel shear", "Nail slip", "Anchorage", "Lower-storey rotation"]
    vals = [
        float(row["Δ bending (mm)"]), float(row["Δ panel shear (mm)"]),
        float(row["Δ nail slip (mm)"]), float(row["Δ anchorage (mm)"]),
        float(row["Δ rotation from below (mm)"]),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.barh(labels, vals, edgecolor="#334155", linewidth=0.6)
    ax.set_xlabel("Inter-storey deflection contribution (mm)")
    ax.set_title("Mechanics-based deflection breakdown", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    total = sum(vals)
    for i, v in enumerate(vals):
        pct = 100.0 * v / total if total else 0.0
        ax.text(v, i, f" {v:.3f} mm ({pct:.1f}%)", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def draw_wood_wall_schematic(row: pd.Series, analyzed: pd.Series):
    """Simple engineering schematic of wall length, rod, chord pack, Lc and ytr."""
    plt = _mpl()
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(10.5, 2.7))
    L = float(row["Wall length (m)"]) * 1000.0
    Lc = float(analyzed["Lc (mm)"])
    nstud = int(row["Chord studs / end"])
    stud_w = 38.0
    cavity = max(L - Lc - nstud * stud_w, 0.0)
    x_rod = cavity / 2.0
    x_comp = x_rod + Lc
    ax.plot([0, L], [0, 0], linewidth=3, color="#64748b")
    # Tension rod
    ax.axvline(x_rod, ymin=0.25, ymax=0.78, linewidth=4, color="#dc2626")
    # Compression chord pack, represented by individual studs centered on compression chord region
    pack_start = min(L - nstud * stud_w, x_comp - (nstud * stud_w) / 2.0)
    pack_start = max(pack_start, 0.0)
    for j in range(nstud):
        x0 = pack_start + j * stud_w
        ax.add_patch(Rectangle((x0, -70), stud_w, 140, fill=False, edgecolor="#1d4ed8", linewidth=1.5))
    ax.scatter([x_comp], [0], color="#1d4ed8", s=35, zorder=5)
    ax.annotate("", xy=(x_comp, 110), xytext=(x_rod, 110), arrowprops=dict(arrowstyle="<->", color="#111827", linewidth=1.3))
    ax.text((x_rod+x_comp)/2, 135, f"Lc = {Lc/1000:.3f} m", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.annotate("", xy=(L, -135), xytext=(0, -135), arrowprops=dict(arrowstyle="<->", color="#475569", linewidth=1.2))
    ax.text(L/2, -165, f"Ls = {L/1000:.3f} m", ha="center", va="top", fontsize=10)
    ax.text(x_rod, -95, "Tension rod", ha="center", va="top", color="#991b1b", fontsize=9)
    ax.text(x_comp, -95, f"Compression chord\n{nstud} studs/end", ha="center", va="top", color="#1e3a8a", fontsize=9)
    ax.text(0.01*L, 175, f"Ac = {float(analyzed['Ac (mm²)']):,.0f} mm²   |   At = {float(analyzed['At (mm²)']):,.1f} mm²   |   ytr = {float(analyzed['ytr (mm)']):,.0f} mm", fontsize=9, va="top")
    ax.set_xlim(-0.03*L, 1.03*L)
    ax.set_ylim(-210, 210)
    ax.axis("off")
    fig.tight_layout()
    return fig


def resize_storey_table(df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = int(n)
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return default_storeys(n)
    old = df.copy().reset_index(drop=True)
    if len(old) >= n:
        out = old.iloc[:n].copy()
    else:
        add = default_storeys(n - len(old))
        # New rows should continue storey numbering; preserve neutral default loads.
        add["Floor lateral force (kN)"] = 0.0
        out = pd.concat([old, add], ignore_index=True)
    out["Storey"] = np.arange(1, n + 1)
    return out


def single_storey_mechanics_k(storey_row: pd.Series | Dict[str, object], V_kN: float, wall_length_m: float | None = None,
                               live_fraction: float = 0.5, cavity_mm: float = DEFAULT_CAVITY_MM,
                               bearing_length_mm: float = DEFAULT_COMPRESSION_BEARING_LENGTH_MM) -> Dict[str, float]:
    r = dict(storey_row)
    r["Storey"] = 1
    r["Floor lateral force (kN)"] = abs(float(V_kN))
    if wall_length_m is not None:
        r["Wall length (m)"] = float(wall_length_m)
    out = analyze_stacked_wall(
        pd.DataFrame([r]), include_lower_storey_rotation=True,
        live_load_fraction_in_compression=float(live_fraction), cavity_mm=float(cavity_mm),
        compression_bearing_length_mm=float(bearing_length_mm),
    )["storeys"].iloc[0]
    return out.to_dict()



def build_all_wall_design_table(
    walls: pd.DataFrame,
    existing: pd.DataFrame | None,
    base_design: pd.Series | Dict[str, object],
) -> pd.DataFrame:
    """Return one single-storey wood mechanics design row per building wall.

    Wall length and direction are synchronized from the diaphragm model. Existing
    wall-specific design choices are preserved by Wall Name. New walls inherit
    the current Wood Wall Lab single-storey design.
    """
    walls_n = normalize_walls(walls)
    base = dict(base_design)
    old = None
    if isinstance(existing, pd.DataFrame) and not existing.empty and "Wall Name" in existing.columns:
        old = existing.set_index("Wall Name", drop=False)

    rows = []
    for _, wr in walls_n.iterrows():
        name = str(wr["Wall Name"])
        if old is not None and name in old.index:
            row = old.loc[name].to_dict()
        else:
            row = {
                "Couple mechanics": True,
                "Wall Name": name,
                "Direction": str(wr["Direction"]),
                "Height (m)": float(base.get("Height (m)", 3.0)),
                "Wall length (m)": float(wr["Wall Length (m)"]),
                "Panel Type": str(base.get("Panel Type", "CSP")),
                "Panel thickness (mm)": float(base.get("Panel thickness (mm)", 12.5)),
                "Panel sides": str(base.get("Panel sides", "S.S")),
                "Nail diameter (mm)": float(base.get("Nail diameter (mm)", 3.25)),
                "Nail spacing (mm)": float(base.get("Nail spacing (mm)", 150.0)),
                "Species": str(base.get("Species", "S-P-F")),
                "Grade": str(base.get("Grade", "No.1/No.2")),
                "Stud size": str(base.get("Stud size", "2x6")),
                "Chord studs / end": int(base.get("Chord studs / end", 4)),
                "Rod model": str(base.get("Rod model", "SR8H")),
                "Take-up device": str(base.get("Take-up device", "CTUD87")),
                "Service dead line load (kN/m)": float(base.get("Service dead line load (kN/m)", 0.0)),
                "Service live line load (kN/m)": float(base.get("Service live line load (kN/m)", 0.0)),
            }
        row["Wall Name"] = name
        row["Direction"] = str(wr["Direction"])
        row["Wall length (m)"] = float(wr["Wall Length (m)"])
        rows.append(row)
    return pd.DataFrame(rows)


def design_row_to_storey_row(design_row: pd.Series | Dict[str, object], V_kN: float) -> Dict[str, object]:
    """Convert an all-wall design row to the mechanics engine's one-storey schema."""
    d = dict(design_row)
    return {
        "Storey": 1,
        "Floor lateral force (kN)": abs(float(V_kN)),
        "Height (m)": float(d["Height (m)"]),
        "Wall length (m)": float(d["Wall length (m)"]),
        "Panel Type": str(d["Panel Type"]),
        "Panel thickness (mm)": float(d["Panel thickness (mm)"]),
        "Panel sides": str(d["Panel sides"]),
        "Nail diameter (mm)": float(d["Nail diameter (mm)"]),
        "Nail spacing (mm)": float(d["Nail spacing (mm)"]),
        "Species": str(d["Species"]),
        "Grade": str(d["Grade"]),
        "Stud size": str(d["Stud size"]),
        "Chord studs / end": int(d["Chord studs / end"]),
        "Rod model": str(d["Rod model"]),
        "Take-up device": str(d["Take-up device"]),
        "Service dead line load (kN/m)": float(d["Service dead line load (kN/m)"]),
        "Service live line load (kN/m)": float(d["Service live line load (kN/m)"]),
    }


def to_excel_bytes(current_result: Dict[str, object], studies: Dict[str, pd.DataFrame] | None = None) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        pd.DataFrame([current_result["properties"]]).to_excel(writer, sheet_name="Key Properties", index=False)
        current_result["wall_properties"].to_excel(writer, sheet_name="Wall Properties", index=False)
        current_result["envelope"].to_excel(writer, sheet_name="Envelope", index=False)
        current_result["x_cases"].to_excel(writer, sheet_name="X Cases", index=False)
        current_result["y_cases"].to_excel(writer, sheet_name="Y Cases", index=False)
        current_result["checks"].to_excel(writer, sheet_name="Equilibrium", index=False)
        if studies:
            used = set()
            for key, df in studies.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    base = key[:31] or "Study"
                    sheet = base
                    j = 1
                    while sheet in used:
                        suffix = f"_{j}"
                        sheet = (base[:31-len(suffix)] + suffix)
                        j += 1
                    used.add(sheet)
                    df.to_excel(writer, sheet_name=sheet, index=False)
    return bio.getvalue()


# -----------------------------------------------------------------------------
# Session defaults
# -----------------------------------------------------------------------------
def initialize_state() -> None:
    if "walls" not in st.session_state:
        st.session_state.walls = default_walls()
    if "preset" not in st.session_state:
        st.session_state.preset = "Custom"
    defaults = {
        "wood_V": 46.0,
        "wood_H": 5.0,
        "wood_L": 12.2,
        "wood_E": 9500.0,
        "wood_A": 10640.0,
        "wood_Bv": 5700.0,
        "wood_en": 0.40,
        "wood_da": 1.89,
        "wood_spacing": 150.0,
        "wood_hd_capacity": 55.25,
        "wood_hd_deflection": 5.54,
        "wood_anchor_mode": "Manual da",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "wood_n_storeys" not in st.session_state:
        st.session_state.wood_n_storeys = 1
    if "wood_storeys" not in st.session_state:
        st.session_state.wood_storeys = default_storeys(1)
    if "wood_include_rotation" not in st.session_state:
        st.session_state.wood_include_rotation = True
    if "wood_live_fraction" not in st.session_state:
        st.session_state.wood_live_fraction = 0.5
    if "wood_cavity_mm" not in st.session_state:
        st.session_state.wood_cavity_mm = DEFAULT_CAVITY_MM
    if "wood_bearing_length_mm" not in st.session_state:
        st.session_state.wood_bearing_length_mm = DEFAULT_COMPRESSION_BEARING_LENGTH_MM


def current_wood_params(V_override: float | None = None, L_override: float | None = None, da_override: float | None = None) -> Dict[str, float]:
    return {
        "V_kN": float(st.session_state.wood_V if V_override is None else V_override),
        "H_m": float(st.session_state.wood_H),
        "L_m": float(st.session_state.wood_L if L_override is None else L_override),
        "E_N_per_mm2": float(st.session_state.wood_E),
        "A_mm2": float(st.session_state.wood_A),
        "Bv_N_per_mm": float(st.session_state.wood_Bv),
        "en_mm": float(st.session_state.wood_en),
        "da_mm": float(st.session_state.wood_da if da_override is None else da_override),
    }


def wood_input_panel(prefix: str = "wood") -> Tuple[Dict[str, float], bool, float, float]:
    c1, c2, c3, c4 = st.columns(4)
    st.session_state.wood_V = c1.number_input("Wall force V (kN)", min_value=0.001, value=float(st.session_state.wood_V), step=1.0, key=f"{prefix}_V_ui")
    st.session_state.wood_H = c2.number_input("Wall height H (m)", min_value=0.10, value=float(st.session_state.wood_H), step=0.1, key=f"{prefix}_H_ui")
    st.session_state.wood_L = c3.number_input("Wall length L (m)", min_value=0.10, value=float(st.session_state.wood_L), step=0.1, key=f"{prefix}_L_ui")
    st.session_state.wood_E = c4.number_input("Boundary element E (N/mm²)", min_value=1.0, value=float(st.session_state.wood_E), step=100.0, key=f"{prefix}_E_ui")

    c5, c6, c7, c8 = st.columns(4)
    st.session_state.wood_A = c5.number_input("Boundary element A (mm²)", min_value=1.0, value=float(st.session_state.wood_A), step=100.0, key=f"{prefix}_A_ui")
    st.session_state.wood_Bv = c6.number_input("Sheathing Bv (N/mm)", min_value=1.0, value=float(st.session_state.wood_Bv), step=100.0, key=f"{prefix}_Bv_ui")
    st.session_state.wood_en = c7.number_input("Nail deformation en (mm)", min_value=0.0, value=float(st.session_state.wood_en), step=0.01, format="%.3f", key=f"{prefix}_en_ui")
    st.session_state.wood_spacing = c8.number_input("Nail spacing used for force/nail (mm)", min_value=1.0, value=float(st.session_state.wood_spacing), step=10.0, key=f"{prefix}_spacing_ui")

    anchor_mode = st.radio("Anchorage deformation model", ["Manual da", "Linearized hold-down"], horizontal=True, key=f"{prefix}_anchor_mode")
    auto_hd = anchor_mode == "Linearized hold-down"
    if not auto_hd:
        st.session_state.wood_da = st.number_input("Total anchorage elongation da (mm)", min_value=0.0, value=float(st.session_state.wood_da), step=0.05, format="%.3f", key=f"{prefix}_da_ui")
        da = float(st.session_state.wood_da)
        hd_capacity = float(st.session_state.wood_hd_capacity)
        hd_deflection = float(st.session_state.wood_hd_deflection)
    else:
        h1, h2 = st.columns(2)
        st.session_state.wood_hd_capacity = h1.number_input("Hold-down capacity used for linearization (kN)", min_value=0.001, value=float(st.session_state.wood_hd_capacity), step=1.0, key=f"{prefix}_hd_cap_ui")
        st.session_state.wood_hd_deflection = h2.number_input("Deflection at that capacity (mm)", min_value=0.0, value=float(st.session_state.wood_hd_deflection), step=0.05, key=f"{prefix}_hd_def_ui")
        hd_capacity = float(st.session_state.wood_hd_capacity)
        hd_deflection = float(st.session_state.wood_hd_deflection)
        hd = linearized_holdown_da_mm(float(st.session_state.wood_V), float(st.session_state.wood_H), float(st.session_state.wood_L), hd_capacity, hd_deflection)
        da = float(hd["da (mm)"])
        st.info(f"Linearized hold-down tension estimate = {hd['Hold-down tension estimate (kN)']:.2f} kN; calculated da = {da:.3f} mm.")

    params = current_wood_params(da_override=da)
    return params, auto_hd, hd_capacity, hd_deflection


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Rigid Diaphragm Research Lab", page_icon="📐", layout="wide")
    inject_css()
    initialize_state()

    st.title("Rigid Diaphragm Research Lab")
    st.caption("Interactive rigid-diaphragm research plus a mechanics-based stacked wood shear-wall deflection and stiffness laboratory.")

    top1, top2, top3 = st.columns([1.15, 1.15, 2.3])
    with top1:
        if st.button("Load FPInnovations benchmark", use_container_width=True):
            s, w = fpinnovations_preset()
            st.session_state.walls = w
            st.session_state.fp_settings = s
            st.session_state.preset = "FPInnovations"
            st.rerun()
    with top2:
        if st.button("Reset symmetric example", use_container_width=True):
            st.session_state.walls = default_walls()
            st.session_state.pop("fp_settings", None)
            st.session_state.preset = "Custom"
            st.rerun()
    with top3:
        st.info("Research strategy: isolate geometry first, stiffness second, wall length with fixed k third, then calculate wood-wall k and reconnect it to the building model.")

    fp = st.session_state.get("fp_settings", {})
    d_Lx = float(fp.get("Lx", 20.0)); d_Ly = float(fp.get("Ly", 12.0))
    d_Xcm = float(fp.get("Xcm", d_Lx/2)); d_Ycm = float(fp.get("Ycm", d_Ly/2))
    d_Fx = float(fp.get("Fx", 100.0)); d_Fy = float(fp.get("Fy", 100.0)); d_acc = float(fp.get("acc", 0.10))

    with st.expander("System model inputs", expanded=False):
        st.subheader("1. Diaphragm and load inputs")
        r1 = st.columns(4)
        Lx = r1[0].number_input("Lx - X/E-W plan dimension (m)", min_value=0.01, value=d_Lx, step=0.1)
        Ly = r1[1].number_input("Ly - Y/N-S plan dimension (m)", min_value=0.01, value=d_Ly, step=0.1)
        Xcm = r1[2].number_input("Xcm (m)", value=d_Xcm, step=0.05)
        Ycm = r1[3].number_input("Ycm (m)", value=d_Ycm, step=0.05)
        r2 = st.columns(4)
        Fx = r2[0].number_input("Fx - diaphragm force in X (kN)", value=d_Fx, step=1.0)
        Fy = r2[1].number_input("Fy - diaphragm force in Y (kN)", value=d_Fy, step=1.0)
        acc = r2[2].number_input("Accidental eccentricity ratio", min_value=0.0, value=d_acc, step=0.01, format="%.3f")
        with r2[3]:
            st.markdown("**Accidental offsets**")
            st.caption(f"X-load offset = ±{acc*Ly:.3f} m")
            st.caption(f"Y-load offset = ±{acc*Lx:.3f} m")

        st.subheader("2. Wall / frame table")
        st.caption("Direction X = E-W wall resisting X force; Direction Y = N-S wall resisting Y force. The x/y coordinates locate the wall centre/resisting line. Wall Length controls the drawing and unit shear; k controls rigid-diaphragm force distribution.")
        edited = st.data_editor(
            st.session_state.walls, num_rows="dynamic", hide_index=True, use_container_width=True,
            column_config={
                "Wall Name": st.column_config.TextColumn(required=True),
                "Direction": st.column_config.SelectboxColumn(options=["X", "Y"], required=True),
                "x (m)": st.column_config.NumberColumn(format="%.3f"),
                "y (m)": st.column_config.NumberColumn(format="%.3f"),
                "k (kN/m)": st.column_config.NumberColumn(format="%.2f", min_value=0.0001),
                "Wall Length (m)": st.column_config.NumberColumn(format="%.3f", min_value=0.0001),
                "Local X Force (kN)": st.column_config.NumberColumn(format="%.3f"),
                "Local Y Force (kN)": st.column_config.NumberColumn(format="%.3f"),
            }, key="wall_editor_research",
        )
        st.session_state.walls = edited

    settings = {"Lx": Lx, "Ly": Ly, "Xcm": Xcm, "Ycm": Ycm, "Fx": Fx, "Fy": Fy, "acc": acc}
    try:
        result = analyze_model(edited, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
        clean_walls = normalize_walls(edited)
    except Exception as exc:
        st.error(f"Model error: {exc}")
        st.stop()

    p = result["properties"]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Xcr", f"{p['Xcr']:.3f} m")
    m2.metric("Ycr", f"{p['Ycr']:.3f} m")
    m3.metric("ex = Xcm-Xcr", f"{p['ex_signed']:.3f} m")
    m4.metric("ey = Ycm-Ycr", f"{p['ey_signed']:.3f} m")
    m5.metric("J", f"{p['J']:,.0f} kN·m")
    m6.metric("Σk", f"X {p['sum_kx']:,.0f} | Y {p['sum_ky']:,.0f}")

    # ------------------------------------------------------------------
    # Persistent model visualization - stays above every study tab
    # ------------------------------------------------------------------
    st.markdown("## Current system visualization")
    vc1, vc2, vc3 = st.columns([1.0, 1.35, 2.65])
    global_load_dir = vc1.radio("Load direction", ["X", "Y"], horizontal=True, key="global_load_dir")
    global_case_mode = vc2.selectbox(
        "Force view",
        ["Envelope (wall-by-wall)", "+ accidental eccentricity", "− accidental eccentricity"],
        key="global_case_mode",
    )
    with vc3:
        if global_case_mode.startswith("Envelope"):
            st.info("Envelope view is for design comparison. Wall maxima may come from different ± accidental-eccentricity cases and are not necessarily simultaneous.")
        else:
            st.success("Single load-case view: all wall forces shown are simultaneous for the selected accidental-eccentricity sign.")

    render_model_snapshot(
        clean_walls, settings, result, global_load_dir, global_case_mode,
        heading=None, show_force_bars=True, compact=False,
    )
    st.divider()

    tabs = st.tabs([
        "Presentation",
        "Geometry study",
        "Stiffness study",
        "Wall-length study",
        "Interaction heatmap",
        "Batch suite",
        "Wood wall lab",
        "Coupled iteration",
        "Validation & export",
        "Step-by-step calculations",
    ])

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    with tabs[0]:
        st.subheader("Model overview")
        st.caption("The same current-system visualization is kept at the top of the app so the structural context never disappears while moving between studies.")
        render_model_snapshot(
            clean_walls, settings, result, global_load_dir, global_case_mode,
            heading="Current wall layout", show_force_bars=False, compact=True,
        )
        st.markdown(
            '<div class="hero-note"><b>Presentation workflow:</b> change one variable at a time. Use a single ± accidental-eccentricity case when you want to show a simultaneous force state; use the envelope only for wall-by-wall design comparison.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("### Current wall-force results")
        env = result["envelope"].copy()
        st.dataframe(env.round(3), hide_index=True, use_container_width=True)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    with tabs[1]:
        st.subheader("Study A - wall location / geometry")
        st.caption("Moves one wall perpendicular to its resisting direction while holding its stiffness and all other model properties fixed.")
        g1, g2, g3 = st.columns(3)
        g_wall = g1.selectbox("Wall to move", clean_walls["Wall Name"].tolist(), key="geo_wall")
        g_load = g2.selectbox("Load direction to evaluate", ["X", "Y"], key="geo_load")
        g_points = g3.number_input("Study points", min_value=11, max_value=501, value=101, step=10, key="geo_points")
        g4, g5 = st.columns(2)
        g_start = g4.number_input("Start normalized position", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key="geo_start")
        g_stop = g5.number_input("Stop normalized position", min_value=0.0, max_value=1.0, value=1.0, step=0.05, key="geo_stop")

        # Live geometry preview - this is intentionally separate from the batch sweep.
        grow = clean_walls.loc[clean_walls["Wall Name"].eq(g_wall)].iloc[0]
        if grow["Direction"] == "X":
            current_gpos = float(grow["y (m)"]) / Ly if Ly else 0.0
            move_dim = Ly
        else:
            current_gpos = float(grow["x (m)"]) / Lx if Lx else 0.0
            move_dim = Lx
        current_gpos = float(np.clip(current_gpos, 0.0, 1.0))
        g_preview_pos = st.slider(
            "Live preview - normalized wall position", 0.0, 1.0, current_gpos, 0.01, key="geo_preview_pos"
        )
        geo_preview_walls = clean_walls.copy()
        gmask = geo_preview_walls["Wall Name"].eq(g_wall)
        if grow["Direction"] == "X":
            geo_preview_walls.loc[gmask, "y (m)"] = g_preview_pos * move_dim
        else:
            geo_preview_walls.loc[gmask, "x (m)"] = g_preview_pos * move_dim
        try:
            geo_preview_result = analyze_model(geo_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
            render_model_snapshot(
                geo_preview_walls, settings, geo_preview_result, g_load, global_case_mode,
                selected_wall=g_wall, heading="Live geometry preview", show_force_bars=False, compact=True,
            )
        except Exception as exc:
            st.warning(f"Preview unavailable at this position: {exc}")

        if st.button("Run geometry study", type="primary", key="run_geo"):
            st.session_state.geo_df = run_geometry_study(clean_walls, settings, g_wall, g_load, int(g_points), g_start, g_stop)
        if "geo_df" in st.session_state:
            gd = st.session_state.geo_df.dropna(subset=["Selected wall |V| (kN)"])
            if not gd.empty:
                a, b = st.columns(2)
                with a:
                    st.pyplot(draw_line_chart(gd, "Normalized position", ["Selected wall |V| (kN)", "Max wall |V| (kN)"], "Wall position vs force", "Normalized wall position", "Force (kN)"), clear_figure=True)
                with b:
                    st.pyplot(draw_line_chart(gd, "Normalized position", ["Xcr (m)", "Ycr (m)"], "Wall position vs center of rigidity", "Normalized wall position", "Coordinate (m)"), clear_figure=True)
                st.dataframe(gd.round(5), hide_index=True, use_container_width=True, height=340)

    # ------------------------------------------------------------------
    # Stiffness
    # ------------------------------------------------------------------
    with tabs[2]:
        st.subheader("Study B - relative wall stiffness")
        st.caption("Varies only the selected wall stiffness. Geometry and all other wall stiffnesses remain fixed.")
        s1, s2, s3 = st.columns(3)
        s_wall = s1.selectbox("Wall to stiffen/soften", clean_walls["Wall Name"].tolist(), key="stiff_wall")
        s_load = s2.selectbox("Load direction to evaluate", ["X", "Y"], key="stiff_load")
        s_points = s3.number_input("Study points", min_value=11, max_value=501, value=101, step=10, key="stiff_points")
        s4, s5 = st.columns(2)
        s_start = s4.number_input("Minimum k/k₀", min_value=0.01, value=0.25, step=0.05, key="stiff_start")
        s_stop = s5.number_input("Maximum k/k₀", min_value=0.02, value=4.0, step=0.25, key="stiff_stop")

        s_preview_mult = st.slider("Live preview - stiffness multiplier k/k₀", 0.10, 5.00, 1.00, 0.05, key="stiff_preview_mult")
        stiff_preview_walls = clean_walls.copy()
        smask = stiff_preview_walls["Wall Name"].eq(s_wall)
        base_k_preview = float(clean_walls.loc[clean_walls["Wall Name"].eq(s_wall), "k (kN/m)"].iloc[0])
        stiff_preview_walls.loc[smask, "k (kN/m)"] = base_k_preview * s_preview_mult
        try:
            stiff_preview_result = analyze_model(stiff_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
            render_model_snapshot(
                stiff_preview_walls, settings, stiff_preview_result, s_load, global_case_mode,
                selected_wall=s_wall, heading="Live stiffness preview", show_force_bars=False, compact=True,
            )
        except Exception as exc:
            st.warning(f"Preview unavailable at this stiffness: {exc}")

        if st.button("Run stiffness study", type="primary", key="run_stiff"):
            st.session_state.stiff_df = run_stiffness_study(clean_walls, settings, s_wall, s_load, int(s_points), s_start, s_stop, True)
        if "stiff_df" in st.session_state:
            sd = st.session_state.stiff_df.dropna(subset=["Selected wall |V| (kN)"])
            if not sd.empty:
                a, b = st.columns(2)
                with a:
                    st.pyplot(draw_line_chart(sd, "Stiffness multiplier", ["Selected wall |V| (kN)", "Max wall |V| (kN)"], "Stiffness ratio vs wall force", "k/k₀", "Force (kN)", log_x=True), clear_figure=True)
                with b:
                    st.pyplot(draw_line_chart(sd, "Relative stiffness k_i/Σk_dir", ["Selected wall force share |V|/F"], "Relative stiffness vs force share", "kᵢ/Σk", "|Vᵢ|/F"), clear_figure=True)
                st.dataframe(sd.round(5), hide_index=True, use_container_width=True, height=340)

        st.markdown("### Preliminary CWC/FPInnovations method — stiffness proportional to wall length")
        st.caption(
            "For an initial rigid-diaphragm analysis, published CWC/FPInnovations examples use relative wall "
            "stiffness proportional to wall length: kᵢ ∝ Lᵢ. This is a preliminary distribution assumption, "
            "not a mechanics-based physical stiffness. The mapped k values below use one common scale factor only "
            "so the existing solver can display them in kN/m; the force distribution depends only on their ratios."
        )
        try:
            lp_walls, lp_table, lp_scale = length_proportional_stiffness_model(clean_walls)
            lp_result = analyze_model(lp_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)

            lpa, lpb, lpc = st.columns(3)
            lpa.metric("Common display scale C", f"{lp_scale:,.1f} kN/m²")
            lpb.metric("Length-based Xcr", f"{lp_result['properties']['Xcr']:.3f} m")
            lpc.metric("Length-based Ycr", f"{lp_result['properties']['Ycr']:.3f} m")

            render_model_snapshot(
                lp_walls, settings, lp_result, s_load, global_case_mode,
                selected_wall=s_wall, heading="Length-proportional relative-stiffness preview",
                show_force_bars=False, compact=True,
            )

            base_view, _ = force_table_for_view(result, s_load, global_case_mode)
            lp_view, _ = force_table_for_view(lp_result, s_load, global_case_mode)
            force_col = f"{s_load.upper()}-load governing signed V (kN)"
            cmp = base_view[["Wall Name", force_col]].rename(columns={force_col:"Current model V (kN)"})
            cmp = cmp.merge(
                lp_view[["Wall Name", force_col]].rename(columns={force_col:"Length-proportional V (kN)"}),
                on="Wall Name", how="outer"
            )
            cmp["ΔV (kN)"] = cmp["Length-proportional V (kN)"] - cmp["Current model V (kN)"]
            denom = cmp["Current model V (kN)"].abs()
            cmp["Δ|V| vs current (%)"] = np.where(denom > 1e-9,
                (cmp["Length-proportional V (kN)"].abs() - denom) / denom * 100.0, np.nan)
            cmp = cmp.merge(lp_table, on="Wall Name", how="left")

            st.markdown("#### Current model vs length-proportional initial RDA")
            st.dataframe(cmp.round(4), hide_index=True, use_container_width=True)
            st.caption(
                "Interpret the L/ΣL column as the relative stiffness assumption. The mapped k values are a normalized "
                "representation for the solver, not calculated wall stiffness. For final comparison, use the mechanics-based k = V/Δ solution."
            )

            if st.button("Apply length-proportional relative stiffness to current model", key="apply_length_prop_k"):
                st.session_state.walls = lp_walls
                st.rerun()
        except Exception as exc:
            st.warning(f"Length-proportional stiffness preview unavailable: {exc}")

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Study C - wall length")
        st.markdown("**Two different experiments are intentionally separated:** fixed stiffness first, then wall length with calculated wood-wall stiffness.")
        l1, l2 = st.columns(2)
        l_wall = l1.selectbox("Wall to vary", clean_walls["Wall Name"].tolist(), key="length_wall")
        l_load = l2.selectbox("Load direction to evaluate", ["X", "Y"], key="length_load")
        row = clean_walls.loc[clean_walls["Wall Name"].eq(l_wall)].iloc[0]
        plan_limit = Lx if row["Direction"] == "X" else Ly
        default_lo = max(0.25, min(float(row["Wall Length (m)"]) * 0.5, plan_limit * 0.5))
        default_hi = max(default_lo + 0.1, min(plan_limit, float(row["Wall Length (m)"]) * 1.5))
        lc1, lc2, lc3 = st.columns(3)
        L_start = lc1.number_input("Minimum wall length (m)", min_value=0.05, value=float(default_lo), step=0.1, key="L_start")
        L_stop = lc2.number_input("Maximum wall length (m)", min_value=0.06, value=float(default_hi), step=0.1, key="L_stop")
        L_points = lc3.number_input("Study points", min_value=11, max_value=401, value=101, step=10, key="L_points")

        fixed_tab, calc_tab = st.tabs(["A - Length only, k fixed", "B - Length changes calculated k"])
        with fixed_tab:
            st.caption("This deliberately isolates wall length from stiffness. It is a mechanics check and a useful presentation demonstration.")
            fixed_preview_default = float(np.clip(float(row["Wall Length (m)"]), 0.05, max(0.05, plan_limit)))
            fixed_preview_L = st.slider(
                "Live preview - wall length with k held fixed (m)",
                0.05, 100.0, fixed_preview_default, 0.05, key="length_fixed_preview_L"
            )
            fixed_preview_walls = clean_walls.copy()
            lmask = fixed_preview_walls["Wall Name"].eq(l_wall)
            fixed_preview_walls.loc[lmask, "Wall Length (m)"] = fixed_preview_L
            try:
                fixed_preview_result = analyze_model(fixed_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                render_model_snapshot(
                    fixed_preview_walls, settings, fixed_preview_result, l_load, global_case_mode,
                    selected_wall=l_wall, heading="Live length preview - fixed stiffness", show_force_bars=False, compact=True,
                )
            except Exception as exc:
                st.warning(f"Preview unavailable at this wall length: {exc}")

            if st.button("Run fixed-k length study", type="primary", key="run_length_fixed"):
                st.session_state.length_fixed_df = run_length_fixed_k_study(clean_walls, settings, l_wall, l_load, L_start, L_stop, int(L_points))
            if "length_fixed_df" in st.session_state:
                ld = st.session_state.length_fixed_df.dropna(subset=["Selected wall |V| (kN)"])
                a, b = st.columns(2)
                with a:
                    st.pyplot(draw_line_chart(ld, "Wall Length (m)", ["Selected wall |V| (kN)"], "Wall length vs distributed force - k fixed", "Wall length (m)", "Wall force (kN)"), clear_figure=True)
                with b:
                    st.pyplot(draw_line_chart(ld, "Wall Length (m)", ["Selected wall v (kN/m)"], "Wall length vs unit shear - k fixed", "Wall length (m)", "Unit shear (kN/m)"), clear_figure=True)
                st.dataframe(ld.round(5), hide_index=True, use_container_width=True, height=300)

        with calc_tab:
            st.caption("Uses the mechanics-based Wood Wall Lab design to recalculate secant stiffness as wall length changes, then re-runs the rigid-diaphragm model. This controlled study is enabled for a one-storey wood-wall model so the current single-level diaphragm force maps directly to wall shear.")
            if int(st.session_state.wood_n_storeys) != 1:
                st.warning("Set the Wood Wall Lab to 1 storey for the automatic length → mechanics-based k → building redistribution study.")
            else:
                base_design = st.session_state.wood_storeys.iloc[0].to_dict()
                calc_preview_default = float(np.clip(float(row["Wall Length (m)"]), 0.05, max(0.05, plan_limit)))
                calc_preview_L = st.slider(
                    "Live preview - wall length with mechanics-based k (m)",
                    0.05, 100.0, calc_preview_default, 0.05, key="length_calc_preview_L_v3"
                )
                calc_preview_walls = clean_walls.copy()
                lmask2 = calc_preview_walls["Wall Name"].eq(l_wall)
                calc_preview_walls.loc[lmask2, "Wall Length (m)"] = calc_preview_L
                try:
                    seed_result = analyze_model(calc_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                    seed_env = seed_result["envelope"].set_index("Wall Name")
                    seed_col = f"{l_load.upper()}-load envelope |V| (kN)"
                    seed_V = float(seed_env.loc[l_wall, seed_col])
                    wr_preview = single_storey_mechanics_k(
                        base_design, seed_V, wall_length_m=calc_preview_L,
                        live_fraction=float(st.session_state.wood_live_fraction),
                        cavity_mm=float(st.session_state.wood_cavity_mm),
                        bearing_length_mm=float(st.session_state.wood_bearing_length_mm),
                    )
                    calc_preview_walls.loc[lmask2, "k (kN/m)"] = float(wr_preview["k secant (kN/m)"])
                    calc_preview_result = analyze_model(calc_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                    st.caption(
                        f"Calculated preview stiffness for {l_wall}: {float(wr_preview['k secant (kN/m)']):,.0f} kN/m "
                        f"from seed wall force {seed_V:.2f} kN; Δ = {float(wr_preview['Δ total inter-storey (mm)']):.3f} mm."
                    )
                    render_model_snapshot(
                        calc_preview_walls, settings, calc_preview_result, l_load, global_case_mode,
                        selected_wall=l_wall, heading="Live length preview - mechanics-based stiffness", show_force_bars=False, compact=True,
                    )
                except Exception as exc:
                    st.warning(f"Mechanics-based calculated-k preview unavailable: {exc}")

                if st.button("Run length + mechanics-based-k study", type="primary", key="run_length_calc_v3"):
                    rows_calc = []
                    for Lv in np.linspace(float(L_start), float(L_stop), int(L_points)):
                        walls_case = clean_walls.copy()
                        m = walls_case["Wall Name"].eq(l_wall)
                        walls_case.loc[m, "Wall Length (m)"] = float(Lv)
                        try:
                            seed = analyze_model(walls_case, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                            seed_env = seed["envelope"].set_index("Wall Name")
                            Vseed = float(seed_env.loc[l_wall, f"{l_load.upper()}-load envelope |V| (kN)"])
                            wresp = single_storey_mechanics_k(
                                base_design, Vseed, wall_length_m=float(Lv),
                                live_fraction=float(st.session_state.wood_live_fraction),
                                cavity_mm=float(st.session_state.wood_cavity_mm),
                                bearing_length_mm=float(st.session_state.wood_bearing_length_mm),
                            )
                            kval = float(wresp["k secant (kN/m)"])
                            walls_case.loc[m, "k (kN/m)"] = kval
                            rr = analyze_model(walls_case, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                            ee = rr["envelope"].set_index("Wall Name")
                            wallV = float(ee.loc[l_wall, f"{l_load.upper()}-load envelope |V| (kN)"])
                            rows_calc.append({
                                "Wall Length (m)": float(Lv), "Seed wall force for k (kN)": Vseed,
                                "Calculated k (kN/m)": kval, "Δ total (mm)": float(wresp["Δ total inter-storey (mm)"]),
                                "Selected wall |V| (kN)": wallV,
                                "Max wall |V| (kN)": float(ee[f"{l_load.upper()}-load envelope |V| (kN)"].max()),
                                "Xcr (m)": float(rr["properties"]["Xcr"]), "Ycr (m)": float(rr["properties"]["Ycr"]),
                                "Lc (m)": float(wresp["Lc (mm)"])/1000.0,
                            })
                        except Exception as exc:
                            rows_calc.append({"Wall Length (m)": float(Lv), "Error": str(exc)})
                    st.session_state.length_calc_df = pd.DataFrame(rows_calc)
                if "length_calc_df" in st.session_state:
                    lcd = st.session_state.length_calc_df.dropna(subset=["Calculated k (kN/m)"])
                    if not lcd.empty:
                        a, b = st.columns(2)
                        with a:
                            st.pyplot(draw_line_chart(lcd, "Wall Length (m)", ["Calculated k (kN/m)"], "Wall length vs mechanics-based wall stiffness", "Wall length (m)", "k (kN/m)"), clear_figure=True)
                        with b:
                            st.pyplot(draw_line_chart(lcd, "Wall Length (m)", ["Selected wall |V| (kN)", "Max wall |V| (kN)"], "Wall length vs building force distribution", "Wall length (m)", "Force (kN)"), clear_figure=True)
                        st.dataframe(lcd.round(5), hide_index=True, use_container_width=True, height=300)

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Study D - geometry × stiffness interaction")
        h1, h2, h3 = st.columns(3)
        h_wall = h1.selectbox("Wall", clean_walls["Wall Name"].tolist(), key="heat_wall")
        h_load = h2.selectbox("Load direction", ["X", "Y"], key="heat_load")
        grid_n = h3.selectbox("Grid resolution", [21, 31, 41, 61, 81, 101], index=2, key="heat_grid")
        h4, h5 = st.columns(2)
        h_kmin = h4.number_input("Minimum k/k₀", min_value=0.01, value=0.25, step=0.05, key="heat_kmin")
        h_kmax = h5.number_input("Maximum k/k₀", min_value=0.02, value=4.0, step=0.25, key="heat_kmax")

        hp1, hp2 = st.columns(2)
        h_preview_pos = hp1.slider("Live preview - normalized position", 0.0, 1.0, 0.50, 0.01, key="heat_preview_pos")
        h_preview_mult = hp2.slider("Live preview - stiffness multiplier k/k₀", 0.10, 5.00, 1.00, 0.05, key="heat_preview_mult")
        heat_preview_walls = clean_walls.copy()
        hmask = heat_preview_walls["Wall Name"].eq(h_wall)
        hrow = clean_walls.loc[clean_walls["Wall Name"].eq(h_wall)].iloc[0]
        if hrow["Direction"] == "X":
            heat_preview_walls.loc[hmask, "y (m)"] = h_preview_pos * Ly
        else:
            heat_preview_walls.loc[hmask, "x (m)"] = h_preview_pos * Lx
        heat_preview_walls.loc[hmask, "k (kN/m)"] = float(hrow["k (kN/m)"]) * h_preview_mult
        try:
            heat_preview_result = analyze_model(heat_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
            render_model_snapshot(
                heat_preview_walls, settings, heat_preview_result, h_load, global_case_mode,
                selected_wall=h_wall, heading="Live interaction preview", show_force_bars=False, compact=True,
            )
        except Exception as exc:
            st.warning(f"Interaction preview unavailable: {exc}")

        st.caption(f"This grid will solve approximately {int(grid_n)**2:,} rigid-diaphragm models.")
        if st.button("Run interaction study", type="primary", key="run_heat"):
            with st.spinner(f"Running {int(grid_n)**2:,} cases..."):
                st.session_state.heat_df = run_interaction_study(clean_walls, settings, h_wall, h_load, int(grid_n), int(grid_n), 0.0, 1.0, h_kmin, h_kmax)
        if "heat_df" in st.session_state:
            hd = st.session_state.heat_df.dropna(subset=["Selected wall |V| (kN)"])
            if not hd.empty:
                c1, c2 = st.columns(2)
                with c1:
                    st.pyplot(draw_heatmap(hd, "Selected wall |V| (kN)", "Selected-wall force: position × stiffness"), clear_figure=True)
                with c2:
                    st.pyplot(draw_heatmap(hd, "Max wall |V| (kN)", "Maximum system wall force: position × stiffness"), clear_figure=True)
                st.dataframe(hd.round(5), hide_index=True, use_container_width=True, height=320)

    # ------------------------------------------------------------------
    # Batch suite
    # ------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Core batch research suite")
        st.caption("Runs the standard controlled studies together and stores every case for export. This is the quickest way to create a repeatable study dataset.")
        b1, b2, b3 = st.columns(3)
        b_wall = b1.selectbox("Research wall", clean_walls["Wall Name"].tolist(), key="batch_wall")
        b_load = b2.selectbox("Research load direction", ["X", "Y"], key="batch_load")
        resolution = b3.selectbox("Resolution", ["Fast (~1,100 cases)", "Standard (~4,000 cases)", "Deep (~10,600 cases)"], index=1)
        if resolution.startswith("Fast"):
            n_curve, n_grid = 61, 31
        elif resolution.startswith("Standard"):
            n_curve, n_grid = 101, 61
        else:
            n_curve, n_grid = 151, 101
        brow = clean_walls.loc[clean_walls["Wall Name"].eq(b_wall)].iloc[0]
        limit = Lx if brow["Direction"] == "X" else Ly
        b_lmin = max(0.25, min(float(brow["Wall Length (m)"]) * 0.5, limit * 0.5))
        b_lmax = max(b_lmin + 0.1, min(limit, float(brow["Wall Length (m)"]) * 1.5))
        total_est = n_curve * 3 + n_grid**2
        st.info(f"Planned run count: approximately {total_est:,} analyses.")
        render_model_snapshot(
            clean_walls, settings, result, b_load, global_case_mode,
            selected_wall=b_wall, heading="Research model being sampled", show_force_bars=False, compact=True,
        )
        if st.button("Run core research suite", type="primary", key="run_batch"):
            with st.spinner(f"Running ~{total_est:,} models..."):
                suite = {}
                suite["Geometry"] = run_geometry_study(clean_walls, settings, b_wall, b_load, n_curve, 0.0, 1.0)
                suite["Stiffness"] = run_stiffness_study(clean_walls, settings, b_wall, b_load, n_curve, 0.25, 4.0, True)
                suite["Length_Fixed_k"] = run_length_fixed_k_study(clean_walls, settings, b_wall, b_load, b_lmin, b_lmax, n_curve)
                suite["Interaction"] = run_interaction_study(clean_walls, settings, b_wall, b_load, n_grid, n_grid, 0.0, 1.0, 0.25, 4.0)
                st.session_state.batch_suite = suite
        if "batch_suite" in st.session_state:
            suite = st.session_state.batch_suite
            total_rows = sum(len(v) for v in suite.values())
            st.success(f"Batch suite complete: {total_rows:,} stored result rows across {len(suite)} studies.")
            summary = pd.DataFrame([{"Study": k, "Rows": len(v), "Valid rows": int(v.dropna(how="all").shape[0])} for k, v in suite.items()])
            st.dataframe(summary, hide_index=True, use_container_width=True)
            batch_xlsx = to_excel_bytes(result, suite)
            st.download_button("Download batch research workbook", batch_xlsx, file_name="rigid_diaphragm_research_suite.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # ------------------------------------------------------------------
    # Wood wall lab - mechanics based
    # ------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Mechanics-Based Wood Shear Wall Laboratory")
        st.caption(
            "FPInnovations stacked-wall mechanics with service-level app inputs. "
            "The engine uses transformed continuous-rod bending stiffness, panel shear, the validated nail-slip relationship, "
            "your validated anchorage deformation model, and rotation from storeys below only. No strength load factors are applied internally."
        )

        wood_main, wood_db, wood_method = st.tabs(["Wall analysis", "Embedded databases", "Method & validation"])

        with wood_main:
            cfg1, cfg2, cfg3, cfg4 = st.columns(4)
            wood_preview_wall = cfg1.selectbox("Building wall linked to this lab", clean_walls["Wall Name"].tolist(), key="wood_preview_wall_v3")
            wood_preview_load = cfg2.selectbox("Building load direction", ["X", "Y"], key="wood_preview_load_v3")
            n_storeys = int(cfg3.number_input("Number of stacked storeys", min_value=1, max_value=12, value=int(st.session_state.wood_n_storeys), step=1, key="wood_n_storeys_ui_v3"))
            st.session_state.wood_n_storeys = n_storeys
            include_rot = cfg4.toggle("Include lower-storey rotation", value=bool(st.session_state.wood_include_rotation), key="wood_include_rot_ui_v3")
            st.session_state.wood_include_rotation = bool(include_rot)

            st.session_state.wood_storeys = resize_storey_table(st.session_state.wood_storeys, n_storeys)

            with st.expander("Serviceability / geometry assumptions", expanded=False):
                aa1, aa2, aa3 = st.columns(3)
                live_fraction = aa1.number_input(
                    "Service live-load fraction used in compression chord force",
                    min_value=0.0, max_value=1.0, value=float(st.session_state.wood_live_fraction), step=0.05,
                    key="wood_live_fraction_ui_v3",
                    help="Explicit input retained from the validated spreadsheet logic; no hidden load factor is applied."
                )
                cavity_mm = aa2.number_input(
                    "Symmetric rod/chord cavity allowance (mm)", min_value=0.0,
                    value=float(st.session_state.wood_cavity_mm), step=1.0, key="wood_cavity_ui_v3",
                    help="Used in Lc = Ls - (n_chord × 38 + cavity). Default is 9 in = 228.6 mm."
                )
                bearing_len = aa3.number_input(
                    "Compression-perpendicular bearing length (mm)", min_value=1.0,
                    value=float(st.session_state.wood_bearing_length_mm), step=1.0, key="wood_bearing_len_ui_v3",
                    help="Existing validated script uses 3 × 38 = 114 mm."
                )
                st.session_state.wood_live_fraction = float(live_fraction)
                st.session_state.wood_cavity_mm = float(cavity_mm)
                st.session_state.wood_bearing_length_mm = float(bearing_len)

            # Convenient synchronization for the common one-storey presentation / iteration case.
            if n_storeys == 1:
                try:
                    current_view, current_label = force_table_for_view(result, wood_preview_load, global_case_mode)
                    current_force = abs(float(current_view.loc[current_view["Wall Name"].eq(wood_preview_wall), f"{wood_preview_load}-load governing signed V (kN)"].iloc[0]))
                    current_wall_length = float(clean_walls.loc[clean_walls["Wall Name"].eq(wood_preview_wall), "Wall Length (m)"].iloc[0])
                    sync1, sync2, sync3 = st.columns([1.3, 1.0, 1.0])
                    sync1.info(f"Current building case: {wood_preview_wall} | {current_label} | |V| = {current_force:.2f} kN | L = {current_wall_length:.2f} m")
                    if sync2.button("Sync V from building", use_container_width=True, key="sync_wood_V_v3"):
                        st.session_state.wood_storeys.loc[0, "Floor lateral force (kN)"] = current_force
                        st.session_state.pop("wood_storey_editor_v3", None)
                        st.rerun()
                    if sync3.button("Sync V + wall length", use_container_width=True, key="sync_wood_VL_v3"):
                        st.session_state.wood_storeys.loc[0, "Floor lateral force (kN)"] = current_force
                        st.session_state.wood_storeys.loc[0, "Wall length (m)"] = current_wall_length
                        st.session_state.pop("wood_storey_editor_v3", None)
                        st.rerun()
                except Exception as exc:
                    st.warning(f"Could not read current building wall force for synchronization: {exc}")

            panel_types = sorted(SHEATHING_BV["Panel Type"].unique().tolist())
            panel_thicks = sorted(SHEATHING_BV["Thickness (mm)"].unique().tolist())
            species_opts = LUMBER_E["Species"].unique().tolist()
            grade_opts = LUMBER_E["Grade"].unique().tolist()
            rod_opts = sorted(set(ROD_GEOMETRY["Strong Rod Standard"].tolist() + ROD_GEOMETRY["Strong Rod High Strength"].tolist()))
            tud_opts = TAKEUP_DEVICES["Model No."].tolist()

            st.markdown("### Storey inputs")
            st.caption("Rows are ordered bottom → top. Storey 1 is the lowest storey. 'Floor lateral force' is the service-level force applied at that level; storey shear is calculated automatically as the sum of that level and all levels above.")
            edited_storeys = st.data_editor(
                st.session_state.wood_storeys,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "Storey": st.column_config.NumberColumn("Storey", disabled=True, format="%d"),
                    "Floor lateral force (kN)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
                    "Height (m)": st.column_config.NumberColumn(format="%.3f", min_value=0.1),
                    "Wall length (m)": st.column_config.NumberColumn(format="%.3f", min_value=0.1),
                    "Panel Type": st.column_config.SelectboxColumn(options=panel_types, required=True),
                    "Panel thickness (mm)": st.column_config.SelectboxColumn(options=panel_thicks, required=True),
                    "Panel sides": st.column_config.SelectboxColumn(options=["S.S", "B.S"], required=True),
                    "Nail diameter (mm)": st.column_config.NumberColumn(format="%.3f", min_value=0.1),
                    "Nail spacing (mm)": st.column_config.NumberColumn(format="%.1f", min_value=1.0),
                    "Species": st.column_config.SelectboxColumn(options=species_opts, required=True),
                    "Grade": st.column_config.SelectboxColumn(options=grade_opts, required=True),
                    "Stud size": st.column_config.SelectboxColumn(options=list(STUD_DEPTH_MM.keys()), required=True),
                    "Chord studs / end": st.column_config.NumberColumn(format="%d", min_value=1, step=1),
                    "Rod model": st.column_config.SelectboxColumn(options=rod_opts, required=True),
                    "Take-up device": st.column_config.SelectboxColumn(options=tud_opts, required=True),
                    "Service dead line load (kN/m)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
                    "Service live line load (kN/m)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
                },
                key="wood_storey_editor_v3",
            )
            edited_storeys["Storey"] = np.arange(1, len(edited_storeys) + 1)
            st.session_state.wood_storeys = edited_storeys.copy()

            try:
                wood_result = analyze_stacked_wall(
                    edited_storeys,
                    include_lower_storey_rotation=bool(include_rot),
                    live_load_fraction_in_compression=float(live_fraction),
                    cavity_mm=float(cavity_mm),
                    compression_bearing_length_mm=float(bearing_len),
                )
                wood_out = wood_result["storeys"]
                st.session_state.wood_mechanics_result = wood_out.copy()
            except Exception as exc:
                st.error(f"Wood-wall mechanics calculation stopped: {exc}")
                wood_out = None

            if wood_out is not None:
                sel_storey = st.selectbox("Storey to inspect / link to current diaphragm model", wood_out["Storey"].astype(int).tolist(), index=0, key="wood_selected_storey_v3")
                srow = wood_out.loc[wood_out["Storey"].eq(sel_storey)].iloc[0]
                input_row = edited_storeys.loc[edited_storeys["Storey"].eq(sel_storey)].iloc[0]

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Storey shear V", f"{float(srow['Storey shear V (kN)']):,.2f} kN")
                m2.metric("Inter-storey Δ", f"{float(srow['Δ total inter-storey (mm)']):,.3f} mm")
                m3.metric("Secant stiffness k", f"{float(srow['k secant (kN/m)']):,.0f} kN/m")
                m4.metric("Lc", f"{float(srow['Lc (mm)'])/1000:.3f} m")
                m5.metric("Rod T/Tr", f"{100*float(srow['Rod utilization T/Tr']):.1f}%")
                from engineering_calculations import render_wood_secant_panel
                render_wood_secant_panel(
                    float(srow["Storey shear V (kN)"]),
                    float(srow["Δ total inter-storey (mm)"]),
                    float(srow["k secant (kN/m)"]),
                )

                sc1, sc2 = st.columns([1.15, 1.0])
                with sc1:
                    st.pyplot(draw_wood_wall_schematic(input_row, srow), clear_figure=True, use_container_width=True)
                with sc2:
                    st.pyplot(draw_mechanics_breakdown(srow), clear_figure=True, use_container_width=True)

                # Whole-stack visualizations
                if len(wood_out) > 1:
                    prof1, prof2 = st.columns(2)
                    with prof1:
                        st.pyplot(draw_line_chart(wood_out, "Storey", ["Δ total inter-storey (mm)"], "Inter-storey deflection by storey", "Storey (bottom → top)", "Δ (mm)"), clear_figure=True)
                    with prof2:
                        st.pyplot(draw_line_chart(wood_out, "Storey", ["k secant (kN/m)"], "Secant wall stiffness by storey", "Storey (bottom → top)", "k (kN/m)"), clear_figure=True)

                st.markdown("### Mechanics calculation table")
                show_cols = [
                    "Storey", "Floor lateral force (kN)", "Storey shear V (kN)", "M top (kN·m)", "M base (kN·m)",
                    "Lc (mm)", "Ac (mm²)", "At (mm²)", "EItr (N·mm²)", "Effective Bv (N/mm)",
                    "Force per nail (N)", "en (mm)", "Tension chord force (kN)", "Compression chord force (kN)",
                    "da total (mm)", "Δ bending (mm)", "Δ panel shear (mm)", "Δ nail slip (mm)",
                    "Δ anchorage (mm)", "Δ rotation from below (mm)", "Δ total inter-storey (mm)",
                    "k secant (kN/m)", "Inter-storey drift ratio", "Cumulative lateral displacement (mm)",
                ]
                st.dataframe(wood_out[show_cols].round(6), hide_index=True, use_container_width=True, height=360)

                st.markdown("### Building preview using selected storey stiffness")
                wood_preview_walls = clean_walls.copy()
                wmask = wood_preview_walls["Wall Name"].eq(wood_preview_wall)
                wood_preview_walls.loc[wmask, "k (kN/m)"] = float(srow["k secant (kN/m)"])
                try:
                    wood_preview_result = analyze_model(wood_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                    render_model_snapshot(
                        wood_preview_walls, settings, wood_preview_result, wood_preview_load, global_case_mode,
                        selected_wall=wood_preview_wall, heading=f"Building preview with Storey {int(sel_storey)} wood-wall k",
                        show_force_bars=False, compact=True,
                    )
                    b1, b2 = st.columns([1.25, 0.75])
                    b1.info(f"{wood_preview_wall}: current system k = {float(clean_walls.loc[clean_walls['Wall Name'].eq(wood_preview_wall), 'k (kN/m)'].iloc[0]):,.0f} kN/m → mechanics-based Storey {int(sel_storey)} k = {float(srow['k secant (kN/m)']):,.0f} kN/m")
                    if b2.button("Apply selected storey k to building wall", type="primary", use_container_width=True, key="apply_mech_k_v3"):
                        new_walls = st.session_state.walls.copy()
                        new_walls.loc[new_walls["Wall Name"].eq(wood_preview_wall), "k (kN/m)"] = float(srow["k secant (kN/m)"])
                        st.session_state.walls = new_walls
                        st.rerun()
                except Exception as exc:
                    st.warning(f"Building preview unavailable: {exc}")

                st.markdown("### Quick parametric sensitivity of the selected storey")
                sens1, sens2, sens3, sens4 = st.columns(4)
                sens_param = sens1.selectbox("Parameter", ["Wall length (m)", "Nail spacing (mm)", "Chord studs / end", "Floor lateral force (kN)"], key="wood_sens_param_v3")
                base_val = float(input_row[sens_param])
                if sens_param == "Chord studs / end":
                    ss = sens2.number_input("Start", min_value=1, value=max(1, int(base_val)-2), step=1, key="wood_sens_start_int_v3")
                    ee = sens3.number_input("Stop", min_value=1, value=max(2, int(base_val)+4), step=1, key="wood_sens_stop_int_v3")
                    nn = sens4.number_input("Points", min_value=2, max_value=20, value=max(2, int(ee-ss+1)), step=1, key="wood_sens_n_int_v3")
                    vals = np.unique(np.rint(np.linspace(ss, ee, int(nn))).astype(int))
                else:
                    ss = sens2.number_input("Start", min_value=0.001, value=max(0.001, base_val*0.5), key="wood_sens_start_v3")
                    ee = sens3.number_input("Stop", min_value=0.002, value=max(0.002, base_val*1.5), key="wood_sens_stop_v3")
                    nn = sens4.number_input("Points", min_value=11, max_value=501, value=101, step=10, key="wood_sens_n_v3")
                    vals = np.linspace(float(ss), float(ee), int(nn))
                sens_rows = []
                for vv in vals:
                    trial = edited_storeys.copy()
                    trial.loc[trial["Storey"].eq(sel_storey), sens_param] = int(vv) if sens_param == "Chord studs / end" else float(vv)
                    try:
                        rr = analyze_stacked_wall(trial, bool(include_rot), float(live_fraction), float(cavity_mm), float(bearing_len))["storeys"]
                        rrsel = rr.loc[rr["Storey"].eq(sel_storey)].iloc[0]
                        sens_rows.append({"Parameter": float(vv), "k secant (kN/m)": float(rrsel["k secant (kN/m)"]), "Δ total (mm)": float(rrsel["Δ total inter-storey (mm)"])})
                    except Exception:
                        sens_rows.append({"Parameter": float(vv), "k secant (kN/m)": np.nan, "Δ total (mm)": np.nan})
                sens_df = pd.DataFrame(sens_rows)
                sp1, sp2 = st.columns(2)
                with sp1:
                    st.pyplot(draw_line_chart(sens_df, "Parameter", ["k secant (kN/m)"], f"{sens_param} vs stiffness", sens_param, "k (kN/m)"), clear_figure=True)
                with sp2:
                    st.pyplot(draw_line_chart(sens_df, "Parameter", ["Δ total (mm)"], f"{sens_param} vs deflection", sens_param, "Δ (mm)"), clear_figure=True)

        with wood_db:
            st.markdown("### Embedded engineering database explorer")
            st.caption("These tables are transcribed from `shearwallanalysis_R2_faster.py`. They are shown transparently so users can inspect exactly which values the analysis is using.")
            db_name = st.selectbox("Database", list(DATABASES.keys()), key="wood_db_name_v3")
            db = DATABASES[db_name].copy()
            st.dataframe(db, hide_index=True, use_container_width=True, height=min(520, 90 + 35 * len(db)))
            st.download_button(
                f"Download {db_name} CSV", db.to_csv(index=False).encode("utf-8"),
                file_name=f"{db_name.lower().replace(' ', '_').replace('/', '_')}.csv", mime="text/csv", key="wood_db_dl_v3"
            )
            st.markdown("#### Database inventory")
            inv = pd.DataFrame([{"Database": name, "Rows": len(df), "Columns": len(df.columns)} for name, df in DATABASES.items()])
            st.dataframe(inv, hide_index=True, use_container_width=True)

        with wood_method:
            st.markdown("### Implemented mechanics")
            st.latex(r"\Delta_i=\Delta_{b,i}+\Delta_{s,i}+\Delta_{n,i}+\Delta_{a,i}+\Delta_{r,i}")
            st.latex(r"\Delta_{b,i}=\frac{V_iH_i^3}{3(EI)_i}+\frac{M_iH_i^2}{2(EI)_i}")
            st.latex(r"\Delta_{s,i}=\frac{V_iH_i}{L_iB_{v,i}}")
            st.latex(r"e_{n,i}=\left(\frac{0.013\,v_{s,i}}{d_f^2}\right)^2,\qquad \Delta_{n,i}=0.0025H_ie_{n,i}")
            st.latex(r"\Delta_{a,i}=\frac{H_i}{L_i}d_{a,i}")
            st.latex(r"\Delta_{r,i}=H_i\left(\sum_{j=1}^{i-1}\theta_j+\sum_{j=1}^{i-1}\alpha_j\right)")
            st.latex(r"k_i=\frac{V_i}{\Delta_i}")
            st.markdown("**Continuous-rod transformed section**")
            st.latex(r"n=E_t/E_c,\quad A_{t,tr}=nA_t,\quad y_{tr}=\frac{A_cL_c}{A_{t,tr}+A_c},\quad I_{tr}=A_{t,tr}y_{tr}^2+A_c(L_c-y_{tr})^2")
            st.markdown("**Symmetric chord/rod geometry used in this app**")
            st.latex(r"L_c=L_s-(n_{chord}\times38+228.6)\ \mathrm{mm}")
            st.caption("`n_chord` is entered explicitly as the number of chord studs at each end; the same number is assumed on both ends.")
            st.markdown("### Mechanics regression checks")
            mech_tests = run_mechanics_self_tests()
            st.dataframe(mech_tests, hide_index=True, use_container_width=True)
            if bool(mech_tests["Pass"].all()):
                st.success("All mechanics regression checks passed.")
            else:
                st.error("At least one mechanics regression check failed.")

    # ------------------------------------------------------------------
    # Method 4 - fully coupled all-wall mechanics iteration
    # ------------------------------------------------------------------
    with tabs[7]:
        st.subheader("Method 4 - fully coupled converged mechanics stiffness")
        st.caption(
            "Every participating wood wall is updated in the same global iteration. The diaphragm is solved once, "
            "the simultaneous service force in every coupled wall is sent to its own mechanics model, all new wall "
            "stiffnesses are calculated, all stiffnesses are updated together, and the diaphragm is solved again."
        )
        st.info(
            "This is a single-diaphragm-level coupled solution. Each participating wall therefore uses a one-storey "
            "wood mechanics model at this level. Walls that are not wood, or that you intentionally want to keep fixed, "
            "can be unchecked in the design table."
        )

        if int(st.session_state.wood_n_storeys) != 1:
            st.warning("For the fully coupled single-diaphragm solution, set the Wood Wall Lab to 1 storey. Multi-storey wall mechanics remain available in the Wood Wall Lab, but a true multi-level coupled building model is a separate analysis problem.")
            render_model_snapshot(clean_walls, settings, result, global_load_dir, global_case_mode, heading="Current single-level diaphragm model", show_force_bars=False, compact=True)
        else:
            base_design = st.session_state.wood_storeys.iloc[0].to_dict()
            current_designs = st.session_state.get("all_wall_designs_v6")
            design_table = build_all_wall_design_table(clean_walls, current_designs, base_design)

            st.markdown("### 1. Assign a mechanics design to every wall")
            st.caption(
                "Wall length is synchronized from the diaphragm model. Edit the physical design of each wall here. "
                "The lateral force is not an input: it is taken from the current diaphragm solution at every global iteration."
            )

            panel_types = sorted(SHEATHING_BV["Panel Type"].unique().tolist())
            panel_thicks = sorted(SHEATHING_BV["Thickness (mm)"].unique().tolist())
            species_opts = LUMBER_E["Species"].unique().tolist()
            grade_opts = LUMBER_E["Grade"].unique().tolist()
            rod_opts = sorted(set(ROD_GEOMETRY["Strong Rod Standard"].tolist() + ROD_GEOMETRY["Strong Rod High Strength"].tolist()))
            tud_opts = TAKEUP_DEVICES["Model No."].tolist()

            copy1, copy2 = st.columns([1.0, 1.0])
            if copy1.button("Copy current Wood Wall Lab design to all walls", use_container_width=True, key="copy_wood_design_all_v6"):
                reset_designs = build_all_wall_design_table(clean_walls, None, base_design)
                st.session_state.all_wall_designs_v6 = reset_designs
                st.session_state.pop("all_wall_design_editor_v6", None)
                st.rerun()
            if copy2.button("Resync wall lengths from current diaphragm model", use_container_width=True, key="sync_all_wall_lengths_v6"):
                st.session_state.all_wall_designs_v6 = build_all_wall_design_table(clean_walls, design_table, base_design)
                st.session_state.pop("all_wall_design_editor_v6", None)
                st.rerun()

            edited_designs = st.data_editor(
                design_table,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "Couple mechanics": st.column_config.CheckboxColumn("Couple mechanics", help="Checked walls update their stiffness from the wood mechanics model. Unchecked walls retain their current diaphragm stiffness."),
                    "Wall Name": st.column_config.TextColumn(disabled=True),
                    "Direction": st.column_config.TextColumn(disabled=True),
                    "Height (m)": st.column_config.NumberColumn(format="%.3f", min_value=0.1),
                    "Wall length (m)": st.column_config.NumberColumn(format="%.3f", disabled=True),
                    "Panel Type": st.column_config.SelectboxColumn(options=panel_types, required=True),
                    "Panel thickness (mm)": st.column_config.SelectboxColumn(options=panel_thicks, required=True),
                    "Panel sides": st.column_config.SelectboxColumn(options=["S.S", "B.S"], required=True),
                    "Nail diameter (mm)": st.column_config.NumberColumn(format="%.3f", min_value=0.1),
                    "Nail spacing (mm)": st.column_config.NumberColumn(format="%.1f", min_value=1.0),
                    "Species": st.column_config.SelectboxColumn(options=species_opts, required=True),
                    "Grade": st.column_config.SelectboxColumn(options=grade_opts, required=True),
                    "Stud size": st.column_config.SelectboxColumn(options=list(STUD_DEPTH_MM.keys()), required=True),
                    "Chord studs / end": st.column_config.NumberColumn(format="%d", min_value=1, step=1),
                    "Rod model": st.column_config.SelectboxColumn(options=rod_opts, required=True),
                    "Take-up device": st.column_config.SelectboxColumn(options=tud_opts, required=True),
                    "Service dead line load (kN/m)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
                    "Service live line load (kN/m)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
                },
                key="all_wall_design_editor_v6",
            )
            # Always enforce current building identity / geometry after the editor returns.
            edited_designs = build_all_wall_design_table(clean_walls, edited_designs, base_design)
            st.session_state.all_wall_designs_v6 = edited_designs.copy()

            coupled_names = edited_designs.loc[edited_designs["Couple mechanics"].fillna(False), "Wall Name"].astype(str).tolist()
            fixed_names = edited_designs.loc[~edited_designs["Couple mechanics"].fillna(False), "Wall Name"].astype(str).tolist()
            dsum1, dsum2 = st.columns(2)
            dsum1.success(f"Mechanics-coupled walls: {', '.join(coupled_names) if coupled_names else 'None'}")
            dsum2.info(f"Fixed-stiffness walls: {', '.join(fixed_names) if fixed_names else 'None'}")
            a1, a2, a3 = st.columns(3)
            a1.metric("Live-load fraction in compression", f"{float(st.session_state.wood_live_fraction):.2f}")
            a2.metric("Symmetric cavity allowance", f"{float(st.session_state.wood_cavity_mm):.1f} mm")
            a3.metric("Compression bearing length", f"{float(st.session_state.wood_bearing_length_mm):.1f} mm")
            st.caption("These three mechanics assumptions are controlled in Wood Wall Lab → Serviceability / geometry assumptions and are applied consistently to every mechanics-coupled wall.")

            st.markdown("### 2. Global iteration settings")
            ci1, ci2, ci3, ci4, ci5 = st.columns(5)
            c_load = ci1.selectbox("Load direction", ["X", "Y"], key="all_coupled_load_v6")
            c_case = ci2.selectbox("Simultaneous accidental-eccentricity case", ["+ accidental eccentricity", "− accidental eccentricity"], key="all_coupled_case_v6")
            c_tol_pct = ci3.number_input("Convergence tolerance (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.1, key="all_coupled_tol_v6")
            c_relax = ci4.number_input("Relaxation factor", min_value=0.05, max_value=1.0, value=0.7, step=0.05, key="all_coupled_relax_v6")
            cmax = ci5.number_input("Maximum iterations", min_value=2, max_value=100, value=40, step=1, key="all_coupled_max_v6")

            min_force = st.number_input(
                "Minimum |wall force| required to update mechanics stiffness (kN)",
                min_value=0.0, value=0.10, step=0.05, key="all_coupled_min_force_v6",
                help="If a coupled wall has essentially zero force in the selected simultaneous case, a service-load secant stiffness V/Δ cannot be meaningfully updated from that case. The app retains that wall's previous stiffness and flags it."
            )

            render_model_snapshot(
                clean_walls, settings, result, c_load, c_case,
                heading="Current model before Method 4 iteration", show_force_bars=False, compact=True,
            )

            if not coupled_names:
                st.error("Select at least one wall under 'Couple mechanics'.")
            elif st.button("Run Method 4 - fully coupled all-wall iteration", type="primary", key="run_all_coupled_v6"):
                walls_it = clean_walls.copy()
                baseline_walls = clean_walls.copy()
                designs_lookup = edited_designs.set_index("Wall Name", drop=False)
                hist_rows: list[dict] = []
                system_rows: list[dict] = []
                wall_state_rows: list[dict] = []
                converged = False
                total_F = float(Fx if c_load == "X" else Fy)
                tol = float(c_tol_pct) / 100.0

                def capture_all_state(state_no: int, phase: str, walls_snapshot: pd.DataFrame, rr_snapshot: Dict[str, object]):
                    view_tbl, case_label = force_table_for_view(rr_snapshot, c_load, c_case)
                    props = rr_snapshot["properties"]
                    system_rows.append({
                        "State": state_no, "Phase": phase, "Load Direction": c_load, "Force View": case_label,
                        "Xcr (m)": float(props["Xcr"]), "Ycr (m)": float(props["Ycr"]),
                        "ex (m)": float(props["ex_signed"]), "ey (m)": float(props["ey_signed"]),
                        "J (kN·m)": float(props["J"]),
                    })
                    signed_col = f"{c_load}-load governing signed V (kN)"
                    wall_lookup = walls_snapshot.set_index("Wall Name")
                    for _, vr in view_tbl.iterrows():
                        name = str(vr["Wall Name"])
                        wr = wall_lookup.loc[name]
                        vv = float(vr[signed_col])
                        wall_state_rows.append({
                            "State": state_no, "Phase": phase, "Load Direction": c_load, "Force View": case_label,
                            "Wall Name": name, "Direction": str(wr["Direction"]),
                            "x (m)": float(wr["x (m)"]), "y (m)": float(wr["y (m)"]),
                            "Wall Length (m)": float(wr["Wall Length (m)"]), "k (kN/m)": float(wr["k (kN/m)"]),
                            "Signed V (kN)": vv, "|V| (kN)": abs(vv),
                            "Force share V/F": vv / total_F if abs(total_F) > 1e-12 else np.nan,
                            "Mechanics coupled": bool(name in coupled_names),
                        })
                    return view_tbl

                rr_current = analyze_model(walls_it, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                capture_all_state(0, "Baseline", walls_it.copy(), rr_current)

                for it in range(1, int(cmax) + 1):
                    view_tbl, case_label = force_table_for_view(rr_current, c_load, c_case)
                    signed_col = f"{c_load}-load governing signed V (kN)"
                    force_lookup = {str(r["Wall Name"]): abs(float(r[signed_col])) for _, r in view_tbl.iterrows()}
                    current_k_lookup = walls_it.set_index("Wall Name")["k (kN/m)"].astype(float).to_dict()

                    proposed_k: dict[str, float] = dict(current_k_lookup)
                    mechanics_rows_this_it: list[dict] = []
                    residuals = []
                    force_changes = []

                    # IMPORTANT: every mechanics wall is evaluated from the SAME current diaphragm state.
                    # No wall stiffness is applied until all wall mechanics calculations are complete.
                    for name in coupled_names:
                        Vwall = float(force_lookup[name])
                        k_used = float(current_k_lookup[name])
                        design = designs_lookup.loc[name]

                        if Vwall < float(min_force):
                            mechanics_rows_this_it.append({
                                "Iteration": it, "Wall Name": name, "|V| (kN)": Vwall,
                                "k used (kN/m)": k_used, "k mechanics (kN/m)": np.nan,
                                "k next (kN/m)": k_used, "Mechanics residual |k_model-k|/k": np.nan,
                                "Relaxed step |Δk|/k": 0.0, "Relative ΔV": np.nan,
                                "Status": f"Retained k: |V| < {float(min_force):.3f} kN",
                            })
                            continue

                        storey_row = design_row_to_storey_row(design, Vwall)
                        wr = single_storey_mechanics_k(
                            storey_row, Vwall, wall_length_m=float(design["Wall length (m)"]),
                            live_fraction=float(st.session_state.wood_live_fraction),
                            cavity_mm=float(st.session_state.wood_cavity_mm),
                            bearing_length_mm=float(st.session_state.wood_bearing_length_mm),
                        )
                        k_model = float(wr["k secant (kN/m)"])
                        if not np.isfinite(k_model) or k_model <= 0:
                            raise ValueError(f"{name}: mechanics model returned invalid stiffness {k_model} kN/m at |V|={Vwall:.4f} kN.")
                        k_next = float(c_relax) * k_model + (1.0 - float(c_relax)) * k_used
                        residual_k = abs(k_model - k_used) / max(abs(k_used), 1e-9)
                        step_k = abs(k_next - k_used) / max(abs(k_used), 1e-9)
                        proposed_k[name] = k_next
                        residuals.append(residual_k)
                        mechanics_rows_this_it.append({
                            "Iteration": it, "Wall Name": name, "|V| (kN)": Vwall,
                            "k used (kN/m)": k_used, "k mechanics (kN/m)": k_model,
                            "k next (kN/m)": k_next, "Mechanics residual |k_model-k|/k": residual_k,
                            "Relaxed step |Δk|/k": step_k, "Relative ΔV": np.nan,
                            "Wall deflection (mm)": float(wr["Δ total inter-storey (mm)"]),
                            "Δ bending (mm)": float(wr["Δ bending (mm)"]),
                            "Δ panel shear (mm)": float(wr["Δ panel shear (mm)"]),
                            "Δ nail slip (mm)": float(wr["Δ nail slip (mm)"]),
                            "Δ anchorage (mm)": float(wr["Δ anchorage (mm)"]),
                            "Δ rotation from below (mm)": float(wr["Δ rotation from below (mm)"]),
                            "Lc (m)": float(wr["Lc (mm)"]) / 1000.0,
                            "Status": "Updated",
                        })

                    # All relaxed stiffnesses are committed SIMULTANEOUSLY here.
                    for name, kval in proposed_k.items():
                        walls_it.loc[walls_it["Wall Name"].eq(name), "k (kN/m)"] = float(kval)

                    rr_next = analyze_model(walls_it, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                    capture_all_state(it, f"After iteration {it}", walls_it.copy(), rr_next)

                    # Compare the newly solved diaphragm force with the force that drove this mechanics update.
                    next_view, _ = force_table_for_view(rr_next, c_load, c_case)
                    next_force_lookup = {str(r["Wall Name"]): abs(float(r[signed_col])) for _, r in next_view.iterrows()}
                    for mr in mechanics_rows_this_it:
                        name = str(mr["Wall Name"])
                        v0 = float(force_lookup[name])
                        v1 = float(next_force_lookup[name])
                        dv_step = abs(v1 - v0) / max(abs(v0), 1e-9)
                        mr["Next |V| (kN)"] = v1
                        mr["Relative ΔV"] = dv_step
                        force_changes.append(dv_step)
                    hist_rows.extend(mechanics_rows_this_it)

                    max_resid = max(residuals) if residuals else np.nan
                    max_dv = max(force_changes) if force_changes else np.nan
                    system_rows[-1]["Max mechanics stiffness residual"] = max_resid
                    system_rows[-1]["Max relative force change"] = max_dv

                    is_converged = (
                        bool(residuals) and np.isfinite(max_resid) and max_resid < tol and
                        np.isfinite(max_dv) and max_dv < tol
                    )
                    rr_current = rr_next
                    if is_converged:
                        converged = True
                        break

                hist_df = pd.DataFrame(hist_rows)
                system_df = pd.DataFrame(system_rows)
                wall_states_df = pd.DataFrame(wall_state_rows)
                st.session_state.all_coupled_hist_v6 = hist_df
                st.session_state.all_coupled_system_states_v6 = system_df
                st.session_state.all_coupled_wall_states_v6 = wall_states_df
                st.session_state.all_coupled_initial_walls_v6 = baseline_walls.copy()
                st.session_state.all_coupled_final_walls_v6 = walls_it.copy()
                st.session_state.all_coupled_designs_v6 = edited_designs.copy()
                st.session_state.all_coupled_converged_v6 = converged
                st.session_state.all_coupled_load_v6_saved = c_load
                st.session_state.all_coupled_case_v6_saved = c_case

            if "all_coupled_hist_v6" in st.session_state:
                hist = st.session_state.all_coupled_hist_v6
                system_states = st.session_state.all_coupled_system_states_v6
                wall_states = st.session_state.all_coupled_wall_states_v6
                saved_load = st.session_state.get("all_coupled_load_v6_saved", c_load)
                saved_case = st.session_state.get("all_coupled_case_v6_saved", c_case)
                final_walls = st.session_state.all_coupled_final_walls_v6
                initial_walls = st.session_state.all_coupled_initial_walls_v6

                iterations_done = int(hist["Iteration"].max()) if not hist.empty else 0
                if st.session_state.get("all_coupled_converged_v6", False):
                    st.success(f"Method 4 converged globally in {iterations_done} iterations. All updated wall stiffnesses and wall forces satisfied the selected tolerance.")
                else:
                    st.warning(f"Method 4 stopped after {iterations_done} iterations without satisfying the global convergence tolerance for every participating wall.")

                st.markdown("### 3. Global convergence")
                updated_hist = hist.loc[hist["Status"].eq("Updated")].copy() if "Status" in hist.columns else hist.copy()
                if not updated_hist.empty:
                    k_used_wide = updated_hist.pivot(index="Iteration", columns="Wall Name", values="k used (kN/m)").reset_index()
                    k_model_wide = updated_hist.pivot(index="Iteration", columns="Wall Name", values="k mechanics (kN/m)").reset_index()
                    force_wide_hist = updated_hist.pivot(index="Iteration", columns="Wall Name", values="|V| (kN)").reset_index()
                    g1, g2 = st.columns(2)
                    with g1:
                        k_cols = [c for c in k_used_wide.columns if c != "Iteration"]
                        st.pyplot(draw_line_chart(k_used_wide, "Iteration", k_cols, "All coupled wall stiffnesses used by diaphragm", "Iteration", "k (kN/m)"), clear_figure=True, use_container_width=True)
                    with g2:
                        v_cols = [c for c in force_wide_hist.columns if c != "Iteration"]
                        st.pyplot(draw_line_chart(force_wide_hist, "Iteration", v_cols, "All coupled wall forces", "Iteration", "|V| (kN)"), clear_figure=True, use_container_width=True)

                    # Compatibility residuals make it obvious whether k_model and k_used have truly converged.
                    compat = updated_hist.pivot(index="Iteration", columns="Wall Name", values="Mechanics residual |k_model-k|/k").reset_index()
                    compat_cols = [c for c in compat.columns if c != "Iteration"]
                    st.pyplot(draw_line_chart(compat, "Iteration", compat_cols, "Mechanics compatibility residual by wall", "Iteration", "|k_model-k_used| / k_used"), clear_figure=True, use_container_width=True)

                st.markdown("### 4. Whole-building response")
                rr_final = analyze_model(final_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                whole1, whole2 = st.columns(2)
                with whole1:
                    st.pyplot(draw_cr_trajectory(final_walls, Lx, Ly, system_states, Xcm, Ycm), clear_figure=True, use_container_width=True)
                with whole2:
                    full_force_wide = wall_states.pivot(index="State", columns="Wall Name", values="Signed V (kN)").reset_index()
                    cols = [c for c in full_force_wide.columns if c != "State"]
                    st.pyplot(draw_line_chart(full_force_wide, "State", cols, "Every wall force through Method 4", "Coupled state", "Signed V (kN)"), clear_figure=True, use_container_width=True)

                before_after = coupled_force_change_table(wall_states)
                b1, b2 = st.columns([1.0, 1.0])
                with b1:
                    st.pyplot(draw_before_after_force_bars(wall_states), clear_figure=True, use_container_width=True)
                with b2:
                    initial_state = system_states.iloc[0]
                    final_state = system_states.iloc[-1]
                    cr_shift = math.hypot(float(final_state["Xcr (m)"] - initial_state["Xcr (m)"]), float(final_state["Ycr (m)"] - initial_state["Ycr (m)"]))
                    st.metric("CR movement", f"{cr_shift:.3f} m")
                    st.metric("Initial Xcr", f"{float(initial_state['Xcr (m)']):.3f} m")
                    st.metric("Final Xcr", f"{float(final_state['Xcr (m)']):.3f} m")
                    st.metric("Initial J", f"{float(initial_state['J (kN·m)']):,.0f} kN·m")
                    st.metric("Final J", f"{float(final_state['J (kN·m)']):,.0f} kN·m")
                st.dataframe(before_after.round(4), hide_index=True, use_container_width=True)

                st.markdown("### 5. Initial vs converged diaphragm")
                initial_result = analyze_model(initial_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                final_result = analyze_model(final_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                iview, ilabel = force_table_for_view(initial_result, saved_load, saved_case)
                fview, flabel = force_table_for_view(final_result, saved_load, saved_case)
                p1, p2 = st.columns(2)
                with p1:
                    st.pyplot(draw_plan_with_forces(initial_walls, Lx, Ly, initial_result["properties"], iview, saved_load, case_label="initial " + ilabel, figsize=(8.2, 4.9)), clear_figure=True, use_container_width=True)
                with p2:
                    st.pyplot(draw_plan_with_forces(final_walls, Lx, Ly, final_result["properties"], fview, saved_load, case_label="Method 4 converged " + flabel, figsize=(8.2, 4.9)), clear_figure=True, use_container_width=True)

                st.markdown("### 6. Final mechanics compatibility by wall")
                if not updated_hist.empty:
                    final_it = int(updated_hist["Iteration"].max())
                    final_compat = updated_hist.loc[updated_hist["Iteration"].eq(final_it)].copy()
                    compat_cols_show = [
                        "Wall Name", "|V| (kN)", "k used (kN/m)", "k mechanics (kN/m)", "k next (kN/m)",
                        "Mechanics residual |k_model-k|/k", "Relative ΔV", "Wall deflection (mm)",
                        "Δ bending (mm)", "Δ panel shear (mm)", "Δ nail slip (mm)", "Δ anchorage (mm)", "Lc (m)", "Status",
                    ]
                    st.dataframe(final_compat[[c for c in compat_cols_show if c in final_compat.columns]].round(6), hide_index=True, use_container_width=True)

                with st.expander("Full wall-by-wall mechanics iteration table", expanded=False):
                    st.dataframe(hist.round(6), hide_index=True, use_container_width=True, height=460)
                with st.expander("Full system-state table", expanded=False):
                    st.dataframe(system_states.round(6), hide_index=True, use_container_width=True)
                with st.expander("Every wall at every coupled state", expanded=False):
                    st.dataframe(wall_states.round(6), hide_index=True, use_container_width=True, height=460)
                with st.expander("Wall design assignments used", expanded=False):
                    st.dataframe(st.session_state.all_coupled_designs_v6, hide_index=True, use_container_width=True)

                st.markdown("### 7. Export / apply Method 4 solution")
                ex1, ex2, ex3, ex4 = st.columns(4)
                ex1.download_button("Mechanics history CSV", hist.to_csv(index=False).encode("utf-8"), file_name="method4_all_wall_mechanics_history.csv", mime="text/csv", use_container_width=True)
                ex2.download_button("System states CSV", system_states.to_csv(index=False).encode("utf-8"), file_name="method4_system_states.csv", mime="text/csv", use_container_width=True)
                ex3.download_button("Wall states CSV", wall_states.to_csv(index=False).encode("utf-8"), file_name="method4_all_wall_states.csv", mime="text/csv", use_container_width=True)
                ex4.download_button("Wall designs CSV", st.session_state.all_coupled_designs_v6.to_csv(index=False).encode("utf-8"), file_name="method4_wall_designs.csv", mime="text/csv", use_container_width=True)

                if st.button("Apply Method 4 converged stiffnesses to main system", type="primary", key="apply_all_coupled_v6"):
                    st.session_state.walls = final_walls.copy()
                    st.rerun()

    # ------------------------------------------------------------------
    # Validation and export
    # ------------------------------------------------------------------
    with tabs[8]:
        st.subheader("Verification, regression checks and exports")
        render_model_snapshot(
            clean_walls, settings, result, global_load_dir, global_case_mode,
            heading="Current model under verification", show_force_bars=False, compact=True,
        )
        v1, v2 = st.columns(2)
        with v1:
            st.markdown("### Automated self-tests")
            tests = run_self_tests()
            mech_tests = run_mechanics_self_tests()
            tests_all = pd.concat([
                tests.assign(Engine="Rigid diaphragm"),
                mech_tests.assign(Engine="Wood-wall mechanics"),
            ], ignore_index=True, sort=False)
            st.dataframe(tests_all, hide_index=True, use_container_width=True)
            if bool(tests_all["Pass"].fillna(False).all()):
                st.success("All rigid-diaphragm and wood-wall mechanics checks passed.")
            else:
                st.error("At least one automated check failed. Do not rely on research outputs until resolved.")
        with v2:
            st.markdown("### Current-model equilibrium")
            checks = result["checks"].copy()
            st.dataframe(checks.round(9), hide_index=True, use_container_width=True)
            err = max(checks["ΔVx (kN)"].abs().max(), checks["ΔVy (kN)"].abs().max(), checks["ΔM (kN·m)"].abs().max())
            if err < 1e-7:
                st.success(f"Force and moment equilibrium close to numerical precision (max residual {err:.2e}).")
            else:
                st.warning(f"Equilibrium residual = {err:.4g}")

        st.markdown("### FPInnovations benchmark")
        fs, fw = fpinnovations_preset()
        fr = analyze_model(fw, fs["Lx"], fs["Ly"], fs["Xcm"], fs["Ycm"], fs["Fx"], fs["Fy"], fs["acc"])
        st.dataframe(benchmark_table(fr).round(3), hide_index=True, use_container_width=True)

        st.markdown("### Download")
        studies = {}
        for key, state_key in [
            ("Geometry", "geo_df"), ("Stiffness", "stiff_df"), ("Length_Fixed_k", "length_fixed_df"),
            ("Length_Calc_k", "length_calc_df"), ("Interaction", "heat_df"),
            ("Method4_Mechanics_History", "all_coupled_hist_v6"),
            ("Method4_System_States", "all_coupled_system_states_v6"),
            ("Method4_Wall_States", "all_coupled_wall_states_v6"),
            ("Method4_Wall_Designs", "all_coupled_designs_v6"),
            ("Legacy_Coupled_History", "coupled_hist_v4"),
            ("Legacy_Coupled_System", "coupled_system_states_v4"),
            ("Legacy_Coupled_Walls", "coupled_wall_states_v4")
        ]:
            if state_key in st.session_state:
                studies[key] = st.session_state[state_key]
        if "wood_mechanics_result" in st.session_state:
            studies["Wood_Mechanics"] = st.session_state.wood_mechanics_result
        if "batch_suite" in st.session_state:
            studies.update({f"Batch_{k}": v for k, v in st.session_state.batch_suite.items()})
        xlsx = to_excel_bytes(result, studies)
        d1, d2 = st.columns(2)
        d1.download_button("Download current analysis + completed studies", xlsx, file_name="rigid_diaphragm_research_lab_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        d2.download_button("Download current wall-force envelope CSV", result["envelope"].to_csv(index=False).encode("utf-8"), file_name="rigid_diaphragm_wall_envelope.csv", mime="text/csv", use_container_width=True)

        st.markdown(
            """
**Engineering-use boundary.** The rigid-diaphragm engine is equilibrium-checked and benchmarked against the uploaded FPInnovations example. The Wood Wall Laboratory uses the attached FPInnovations mechanics-based stacked-wall equations, the user-confirmed symmetric `Lc` rule, the validated nail-slip relationship, and the validated anchorage formulation from the supplied Python scripts. Embedded property tables are transcribed from `shearwallanalysis_R2_faster.py` and are exposed in the Database Explorer for review. Final engineering use still requires confirmation that the selected database values and source editions are appropriate for the project.
            """
        )

    # ------------------------------------------------------------------
    # Transparent calculation sheet / printable PDF
    # ------------------------------------------------------------------
    with tabs[9]:
        render_calculation_tab(clean_walls, settings, result)

    st.divider()
    st.caption("Research Lab Mechanics V6 - Method 4 updates all participating wood-wall stiffnesses simultaneously and retains full mechanics, CR and wall-force history for validation and reporting.")


if __name__ == "__main__":
    main()
