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
from multi_storey_ui import render_multi_storey_tab
from wall_design_ui import render_wall_design_studio

from rigid_diaphragm_core import (
    analyze_model,
    benchmark_table,
    coupled_single_wood_wall,
    default_walls,
    fpinnovations_preset,
    linearized_holdown_da_mm,
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
    st.caption("Interactive presentation, verified rigid-diaphragm mechanics, batch parametric studies, and a transparent wood shear-wall stiffness laboratory.")

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
        "Multi-storey periods & drift",
        "Wall design studio",
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
            st.caption("Uses the FPInnovations-example wall-stiffness equation with your current Wood Wall Lab inputs. Bv, en and anchorage data must be verified for the intended design basis.")
            st.info("The current Wood Wall Lab property values are used here. Adjust them in the Wood Wall Lab tab, then return and rerun this study.")
            calc_preview_default = float(np.clip(float(row["Wall Length (m)"]), 0.05, max(0.05, plan_limit)))
            calc_preview_L = st.slider(
                "Live preview - wall length with calculated wood-wall k (m)",
                0.05, 100.0, calc_preview_default, 0.05, key="length_calc_preview_L"
            )
            calc_preview_walls = clean_walls.copy()
            lmask2 = calc_preview_walls["Wall Name"].eq(l_wall)
            calc_preview_walls.loc[lmask2, "Wall Length (m)"] = calc_preview_L
            try:
                seed_result = analyze_model(calc_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                seed_env = seed_result["envelope"].set_index("Wall Name")
                seed_col = f"{l_load.upper()}-load envelope |V| (kN)"
                seed_V = float(seed_env.loc[l_wall, seed_col])
                wp_preview = current_wood_params(V_override=seed_V, L_override=calc_preview_L)
                wr_preview = wood_wall_response(**wp_preview)
                calc_preview_walls.loc[lmask2, "k (kN/m)"] = wr_preview["k secant (kN/m)"]
                calc_preview_result = analyze_model(calc_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                st.caption(f"Calculated preview stiffness for {l_wall}: {wr_preview['k secant (kN/m)']:,.0f} kN/m using seed wall force {seed_V:.2f} kN.")
                render_model_snapshot(
                    calc_preview_walls, settings, calc_preview_result, l_load, global_case_mode,
                    selected_wall=l_wall, heading="Live length preview - calculated stiffness", show_force_bars=False, compact=True,
                )
            except Exception as exc:
                st.warning(f"Calculated-k preview unavailable: {exc}")

            if st.button("Run length + calculated-k study", type="primary", key="run_length_calc"):
                wp = current_wood_params()
                st.session_state.length_calc_df = run_length_calculated_k_building_study(clean_walls, settings, l_wall, l_load, L_start, L_stop, int(L_points), wp, True)
            if "length_calc_df" in st.session_state:
                lcd = st.session_state.length_calc_df.dropna(subset=["Calculated k (kN/m)"])
                a, b = st.columns(2)
                with a:
                    st.pyplot(draw_line_chart(lcd, "Wall Length (m)", ["Calculated k (kN/m)"], "Wall length vs calculated wall stiffness", "Wall length (m)", "k (kN/m)"), clear_figure=True)
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
    # Wood wall lab
    # ------------------------------------------------------------------
    with tabs[6]:
        st.subheader("Wood Wall Laboratory")
        st.caption("Implements the wall-stiffness equation printed in the uploaded FPInnovations example. This V1 intentionally does not embed CSA O86 tables: enter verified Bv, en and anchorage properties for your chosen design basis.")
        wv1, wv2 = st.columns(2)
        wood_preview_wall = wv1.selectbox("Building wall linked to the laboratory", clean_walls["Wall Name"].tolist(), key="wood_preview_wall")
        wood_preview_load = wv2.selectbox("Load direction for building preview", ["X", "Y"], key="wood_preview_load")
        wp, auto_hd, hd_cap, hd_def = wood_input_panel("lab")
        if auto_hd:
            hd = linearized_holdown_da_mm(wp["V_kN"], wp["H_m"], wp["L_m"], hd_cap, hd_def)
            wp["da_mm"] = hd["da (mm)"]
        wr = wood_wall_response(**wp)
        force_per_nail = wr["v (kN/m = N/mm)"] * float(st.session_state.wood_spacing)

        wood_preview_walls = clean_walls.copy()
        wmask = wood_preview_walls["Wall Name"].eq(wood_preview_wall)
        wood_preview_walls.loc[wmask, "k (kN/m)"] = wr["k secant (kN/m)"]
        try:
            wood_preview_result = analyze_model(wood_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
            render_model_snapshot(
                wood_preview_walls, settings, wood_preview_result, wood_preview_load, global_case_mode,
                selected_wall=wood_preview_wall, heading="Building preview using calculated wood-wall stiffness",
                show_force_bars=False, compact=True,
            )
        except Exception as exc:
            st.warning(f"Building preview unavailable: {exc}")

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Secant stiffness k", f"{wr['k secant (kN/m)']:,.0f} kN/m")
        q2.metric("Total deflection Δ", f"{wr['Δ total (mm)']:.3f} mm")
        q3.metric("Unit shear v", f"{wr['v (kN/m = N/mm)']:.3f} kN/m")
        q4.metric("Force / nail", f"{force_per_nail:.0f} N")
        from engineering_calculations import render_wood_secant_panel
        render_wood_secant_panel(
            float(wp["V_kN"]),
            float(wr["Δ total (mm)"]),
            float(wr["k secant (kN/m)"]),
        )

        c1, c2 = st.columns([1.0, 1.15])
        with c1:
            st.pyplot(draw_deformation_breakdown(wr), clear_figure=True)
        with c2:
            comp = pd.DataFrame({
                "Component": ["Bending", "Sheathing", "Fastener slip", "Anchorage", "Total"],
                "Deflection (mm)": [wr["Δ bending (mm)"], wr["Δ sheathing (mm)"], wr["Δ fastener (mm)"], wr["Δ anchorage (mm)"], wr["Δ total (mm)"]],
            })
            st.dataframe(comp.round(5), hide_index=True, use_container_width=True)
            st.warning("en is a user-entered deformation parameter in this V1. The FPInnovations example obtains en from CSA O86 based on force per nail. Do not treat the current value as universal.")

        st.markdown("### Wood-wall sensitivity")
        wtab1, wtab2 = st.tabs(["Length sweep", "Single-parameter sweep"])
        with wtab1:
            a1, a2, a3 = st.columns(3)
            wl_start = a1.number_input("Length sweep start (m)", min_value=0.1, value=max(0.5, wp["L_m"]*0.4), step=0.1, key="wood_L_start")
            wl_stop = a2.number_input("Length sweep stop (m)", min_value=0.2, value=max(1.0, wp["L_m"]*1.2), step=0.1, key="wood_L_stop")
            wl_n = a3.number_input("Points", min_value=11, max_value=501, value=101, step=10, key="wood_L_n")
            wld = run_wood_length_study(wp["V_kN"], wp["H_m"], wl_start, wl_stop, int(wl_n), wp["E_N_per_mm2"], wp["A_mm2"], wp["Bv_N_per_mm"], wp["en_mm"], wp["da_mm"], auto_hd, hd_cap, hd_def)
            a, b = st.columns(2)
            with a:
                st.pyplot(draw_line_chart(wld, "Wall Length (m)", ["k secant (kN/m)"], "Wall length vs wood-wall stiffness", "Wall length (m)", "k (kN/m)"), clear_figure=True)
            with b:
                st.pyplot(draw_line_chart(wld, "Wall Length (m)", ["Δ sheathing (mm)", "Δ fastener (mm)", "Δ anchorage (mm)"], "Wall length vs deformation components", "Wall length (m)", "Deflection (mm)"), clear_figure=True)
        with wtab2:
            param_map = {
                "Wall force V": "V_kN",
                "Wall height H": "H_m",
                "Wall length L": "L_m",
                "Boundary E": "E_N_per_mm2",
                "Boundary A": "A_mm2",
                "Sheathing Bv": "Bv_N_per_mm",
                "Nail deformation en": "en_mm",
                "Anchorage da": "da_mm",
            }
            ps1, ps2, ps3, ps4 = st.columns(4)
            p_label = ps1.selectbox("Parameter", list(param_map.keys()), key="wood_param")
            p_key = param_map[p_label]
            base_val = float(wp[p_key])
            p_start = ps2.number_input("Start", value=max(0.0001, base_val*0.5), key="wood_p_start")
            p_stop = ps3.number_input("Stop", value=max(0.0002, base_val*1.5), key="wood_p_stop")
            p_n = ps4.number_input("Points", min_value=11, max_value=501, value=101, step=10, key="wood_p_n")
            psd = run_wood_parameter_study(wp, p_key, p_start, p_stop, int(p_n))
            st.pyplot(draw_line_chart(psd, "Parameter", ["k secant (kN/m)"], f"{p_label} vs wall stiffness", p_label, "k (kN/m)"), clear_figure=True)

        st.markdown("### Send this wall stiffness to the building")
        ap1, ap2 = st.columns([1.2, 1.0])
        ap1.info(f"Target wall: {wood_preview_wall} | calculated k = {wr['k secant (kN/m)']:,.0f} kN/m")
        if ap2.button("Apply calculated k to linked wall", type="primary", use_container_width=True):
            new_walls = st.session_state.walls.copy()
            new_walls.loc[new_walls["Wall Name"].eq(wood_preview_wall), "k (kN/m)"] = wr["k secant (kN/m)"]
            st.session_state.walls = new_walls
            st.success(f"Assigned k = {wr['k secant (kN/m)']:,.0f} kN/m to {wood_preview_wall}.")
            st.rerun()

    # ------------------------------------------------------------------
    # Coupled iteration
    # ------------------------------------------------------------------
    with tabs[7]:
        st.subheader("Coupled single-wood-wall iteration")
        st.caption("Re-distributes building force, calculates the selected wood wall's secant stiffness, updates k, and repeats. Other walls remain fixed. en remains the user-entered value in this V1.")
        ci1, ci2, ci3, ci4 = st.columns(4)
        c_wall = ci1.selectbox("Wood wall to iterate", clean_walls["Wall Name"].tolist(), key="coupled_wall")
        c_load = ci2.selectbox("Load direction used for iteration", ["X", "Y"], key="coupled_load")
        c_tol_pct = ci3.number_input("Convergence tolerance (%)", min_value=0.01, max_value=10.0, value=0.5, step=0.1, key="coupled_tol")
        c_relax = ci4.number_input("Relaxation factor", min_value=0.05, max_value=1.0, value=0.7, step=0.05, key="coupled_relax")
        cmax = st.number_input("Maximum iterations", min_value=2, max_value=100, value=30, step=1, key="coupled_max")
        st.info("The selected wall uses the current Wood Wall Lab properties. Its wall length is taken from the building wall table; all other wood properties come from the lab.")
        render_model_snapshot(
            clean_walls, settings, result, c_load, global_case_mode,
            selected_wall=c_wall, heading="Current model before coupled iteration", show_force_bars=False, compact=True,
        )
        if st.button("Run coupled iteration", type="primary", key="run_coupled"):
            cw = current_wood_params()
            cw["L_m"] = float(clean_walls.loc[clean_walls["Wall Name"].eq(c_wall), "Wall Length (m)"].iloc[0])
            auto = st.session_state.get("lab_anchor_mode", "Manual da") == "Linearized hold-down"
            hist, final_walls, coupled = coupled_single_wood_wall(clean_walls, settings, c_wall, c_load, cw, c_tol_pct/100.0, int(cmax), c_relax, auto, float(st.session_state.wood_hd_capacity), float(st.session_state.wood_hd_deflection))
            st.session_state.coupled_hist = hist
            st.session_state.coupled_final_walls = final_walls
            st.session_state.coupled_result = coupled
        if "coupled_hist" in st.session_state:
            hist = st.session_state.coupled_hist
            meta = st.session_state.coupled_result["meta"]
            if meta["converged"]:
                st.success(f"Converged in {meta['iterations']} iterations. Final k = {meta['final_k']:,.0f} kN/m.")
            else:
                st.warning(f"Did not meet the selected tolerance within {meta['iterations']} iterations. Final k = {meta['final_k']:,.0f} kN/m.")
            a, b = st.columns(2)
            with a:
                st.pyplot(draw_line_chart(hist, "Iteration", ["k used (kN/m)", "k from wall model (kN/m)"], "Stiffness convergence", "Iteration", "k (kN/m)"), clear_figure=True)
            with b:
                st.pyplot(draw_line_chart(hist, "Iteration", ["Wall force |V| (kN)"], "Wall-force convergence", "Iteration", "Force (kN)"), clear_figure=True)
            st.dataframe(hist.round(6), hide_index=True, use_container_width=True)
            try:
                final_preview_walls = st.session_state.coupled_final_walls.copy()
                final_preview_result = analyze_model(final_preview_walls, Lx, Ly, Xcm, Ycm, Fx, Fy, acc)
                render_model_snapshot(
                    final_preview_walls, settings, final_preview_result, c_load, global_case_mode,
                    selected_wall=c_wall, heading="Converged coupled model preview", show_force_bars=False, compact=True,
                )
            except Exception as exc:
                st.warning(f"Converged-model preview unavailable: {exc}")
            if st.button("Use converged wall model in main system", key="apply_coupled"):
                st.session_state.walls = st.session_state.coupled_final_walls.copy()
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
            st.dataframe(tests, hide_index=True, use_container_width=True)
            if bool(tests["Pass"].all()):
                st.success("All automated checks passed.")
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
            ("Length_Calc_k", "length_calc_df"), ("Interaction", "heat_df"), ("Coupled_History", "coupled_hist")
        ]:
            if state_key in st.session_state:
                studies[key] = st.session_state[state_key]
        if "batch_suite" in st.session_state:
            studies.update({f"Batch_{k}": v for k, v in st.session_state.batch_suite.items()})
        xlsx = to_excel_bytes(result, studies)
        d1, d2 = st.columns(2)
        d1.download_button("Download current analysis + completed studies", xlsx, file_name="rigid_diaphragm_research_lab_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
        d2.download_button("Download current wall-force envelope CSV", result["envelope"].to_csv(index=False).encode("utf-8"), file_name="rigid_diaphragm_wall_envelope.csv", mime="text/csv", use_container_width=True)

        st.markdown(
            """
**Engineering-use boundary.** The rigid-diaphragm engine is equilibrium-checked and benchmarked against the uploaded FPInnovations example. The Wood Wall Laboratory implements the stiffness equation printed in that example, but this V1 deliberately requires the user to supply verified `Bv`, `en`, and anchorage properties. It does not reproduce CSA O86 tables or manufacturer databases.
            """
        )

    # ------------------------------------------------------------------
    # Transparent calculation sheet / printable PDF
    # ------------------------------------------------------------------
    with tabs[9]:
        render_calculation_tab(clean_walls, settings, result)

    # ------------------------------------------------------------------
    # NEW - independent multi-storey diaphragm/mode/drift research solver
    # ------------------------------------------------------------------
    with tabs[10]:
        render_multi_storey_tab(clean_walls, settings)

    with tabs[11]:
        render_wall_design_studio(clean_walls, settings)

    st.divider()
    st.caption("Research Lab V1 - keep geometry effects, stiffness effects, wall-length effects, and wood-wall property assumptions separated when interpreting results.")


if __name__ == "__main__":
    main()
