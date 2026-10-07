from __future__ import annotations

"""Batch parametric atlas for the mechanics-based wood shear-wall model.

This module intentionally keeps the study engine separate from Streamlit so the
same dataset can be regenerated in a notebook or test harness.  It does not
perform a wall-strength/code-capacity check.  It varies mechanics inputs and
reports service-level deflection/stiffness response using the validated
wood_shearwall_mechanics engine.
"""

from io import BytesIO
from typing import Dict, Iterable, List, Sequence
import math
import uuid

import numpy as np
import pandas as pd

from wood_shearwall_mechanics import (
    SHEATHING_BV,
    analyze_stacked_wall,
    get_bv,
    normalize_storeys,
)

MODEL_VERSION = "Wood Shearwall Parametric Atlas V1"


def parse_positive_float_list(text: str, *, name: str = "values") -> List[float]:
    """Parse comma/semicolon/newline-separated positive numbers, preserving order."""
    raw = str(text).replace(";", ",").replace("\n", ",")
    vals: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"{name}: '{token}' is not a number.") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name}: all values must be finite and > 0.")
        if not any(abs(value - old) <= 1e-12 for old in vals):
            vals.append(value)
    if not vals:
        raise ValueError(f"{name}: enter at least one value.")
    return vals


def available_sheathing_pairs(
    panel_types: Sequence[str] | None = None,
    thicknesses_mm: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Return only panel-type/thickness pairs that exist in the embedded Bv table."""
    q = SHEATHING_BV.copy()
    if panel_types:
        wanted = {str(x) for x in panel_types}
        q = q[q["Panel Type"].astype(str).isin(wanted)]
    if thicknesses_mm:
        arr = np.asarray([float(x) for x in thicknesses_mm], dtype=float)
        q = q[q["Thickness (mm)"].astype(float).apply(lambda x: bool(np.isclose(arr, x).any()))]
    return q.sort_values(["Panel Type", "Thickness (mm)"]).reset_index(drop=True)


def build_assembly_catalog(
    panel_types: Sequence[str],
    thicknesses_mm: Sequence[float],
    panel_sides: Sequence[str],
    nail_spacings_mm: Sequence[float],
    nail_diameter_mm: float,
) -> pd.DataFrame:
    """Build traceable mechanics-study assemblies from valid Bv panel pairs.

    'Valid' here means only that the panel type/thickness pair exists in the
    embedded Bv database. Nail spacing is a research input and is NOT screened
    against a CSA wall-resistance table in this module.
    """
    pairs = available_sheathing_pairs(panel_types, thicknesses_mm)
    if pairs.empty:
        raise ValueError("No selected panel type/thickness combination exists in the embedded sheathing Bv table.")
    sides = [str(x).upper() for x in panel_sides]
    bad_sides = [x for x in sides if x not in {"S.S", "B.S"}]
    if bad_sides:
        raise ValueError(f"Panel sides must be S.S or B.S; received {bad_sides}.")
    if not sides:
        raise ValueError("Select at least one panel-side condition.")
    spacings = [float(x) for x in nail_spacings_mm]
    if not spacings or any(x <= 0 for x in spacings):
        raise ValueError("Nail spacings must be > 0.")
    if float(nail_diameter_mm) <= 0:
        raise ValueError("Nail diameter must be > 0.")

    rows = []
    idx = 0
    for _, p in pairs.iterrows():
        for side in sides:
            for spacing in spacings:
                idx += 1
                panel = str(p["Panel Type"])
                thick = float(p["Thickness (mm)"])
                rows.append({
                    "Assembly ID": f"A{idx:04d}",
                    "Panel Type": panel,
                    "Panel thickness (mm)": thick,
                    "Panel sides": side,
                    "Nail diameter (mm)": float(nail_diameter_mm),
                    "Nail spacing (mm)": float(spacing),
                    "Effective Bv (N/mm)": float(get_bv(panel, thick, side)),
                    "Database pair exists": True,
                    "Strength-table screen": "NOT PERFORMED - mechanics study only",
                })
    return pd.DataFrame(rows)


def _analyze_single_storey(
    row: Dict[str, object],
    *,
    live_load_fraction_in_compression: float,
    cavity_mm: float,
    compression_bearing_length_mm: float,
) -> Dict[str, object]:
    df = pd.DataFrame([row])
    result = analyze_stacked_wall(
        df,
        include_lower_storey_rotation=False,
        live_load_fraction_in_compression=float(live_load_fraction_in_compression),
        cavity_mm=float(cavity_mm),
        compression_bearing_length_mm=float(compression_bearing_length_mm),
    )
    return result["storeys"].iloc[0].to_dict()


def _force_from_demand(mode: str, level: float, base_force_kN: float, wall_length_m: float) -> float:
    if mode == "Baseline force multiplier":
        return float(base_force_kN) * float(level)
    if mode == "Constant unit shear (kN/m)":
        return float(level) * float(wall_length_m)
    if mode == "Constant total force (kN)":
        return float(level)
    raise ValueError(f"Unsupported demand mode: {mode}")


def _curated_result_fields(r: Dict[str, object]) -> Dict[str, object]:
    keys = [
        "Storey shear V (kN)", "M top (kN·m)", "M base (kN·m)",
        "Lc (mm)", "Ac (mm²)", "At (mm²)", "EItr (N·mm²)",
        "Effective Bv (N/mm)", "v (N/mm)", "Force per nail (N)", "en (mm)",
        "Tension chord force (kN)", "Compression chord force (kN)",
        "Rod resistance Tr (kN)", "Rod utilization T/Tr", "da total (mm)",
        "Δ bending (mm)", "Δ panel shear (mm)", "Δ nail slip (mm)",
        "Δ anchorage (mm)", "Δ rotation from below (mm)",
        "Δ total inter-storey (mm)", "k secant (kN/m)",
        "Inter-storey drift ratio", "Cumulative lateral displacement (mm)",
        "Bending fraction", "Panel shear fraction", "Nail slip fraction",
        "Anchorage fraction", "Lower-storey rotation fraction",
    ]
    return {k: r.get(k, np.nan) for k in keys}


def run_wood_parametric_atlas(
    base_row: Dict[str, object] | pd.Series,
    *,
    panel_types: Sequence[str],
    thicknesses_mm: Sequence[float],
    panel_sides: Sequence[str],
    nail_spacings_mm: Sequence[float],
    heights_m: Sequence[float],
    aspect_ratios_h_over_l: Sequence[float],
    demand_mode: str,
    demand_levels: Sequence[float],
    live_load_fraction_in_compression: float = 0.5,
    cavity_mm: float = 228.6,
    compression_bearing_length_mm: float = 114.0,
    study_id: str | None = None,
) -> Dict[str, pd.DataFrame]:
    """Run Assembly × Height × H/L × Demand full factorial study.

    The current one-storey Wood Wall Lab row supplies all fixed framing,
    chord/rod, take-up, gravity-load and nail-diameter assumptions.  Sheathing,
    side condition and nail spacing are swept through the assembly catalog.
    """
    base = dict(base_row)
    base["Storey"] = 1
    base_df = normalize_storeys(pd.DataFrame([base]))
    base = base_df.iloc[0].to_dict()

    heights = [float(x) for x in heights_m]
    ars = [float(x) for x in aspect_ratios_h_over_l]
    levels = [float(x) for x in demand_levels]
    if not heights or any(x <= 0 for x in heights):
        raise ValueError("Heights must be > 0.")
    if not ars or any(x <= 0 for x in ars):
        raise ValueError("Aspect ratios H/L must be > 0.")
    if not levels or any(x <= 0 for x in levels):
        raise ValueError("Demand levels must be > 0.")

    assemblies = build_assembly_catalog(
        panel_types, thicknesses_mm, panel_sides, nail_spacings_mm,
        nail_diameter_mm=float(base["Nail diameter (mm)"]),
    )

    sid = study_id or f"WOOD-{uuid.uuid4().hex[:8].upper()}"
    base_force = float(base["Floor lateral force (kN)"])
    base_h = float(base["Height (m)"])
    base_l = float(base["Wall length (m)"])
    base_ar = base_h / base_l
    base_mech = _analyze_single_storey(
        base,
        live_load_fraction_in_compression=live_load_fraction_in_compression,
        cavity_mm=cavity_mm,
        compression_bearing_length_mm=compression_bearing_length_mm,
    )
    global_k_ref = float(base_mech["k secant (kN/m)"])

    reference_cache: Dict[tuple, Dict[str, object]] = {}
    rows: List[Dict[str, object]] = []
    case_no = 0

    for H in heights:
        for ar in ars:
            L = H / ar
            for level in levels:
                V = _force_from_demand(demand_mode, level, base_force, L)
                ref_key = (round(H, 12), round(L, 12), round(V, 12))
                if ref_key not in reference_cache:
                    ref_row = dict(base)
                    ref_row.update({
                        "Storey": 1,
                        "Floor lateral force (kN)": V,
                        "Height (m)": H,
                        "Wall length (m)": L,
                    })
                    try:
                        reference_cache[ref_key] = _analyze_single_storey(
                            ref_row,
                            live_load_fraction_in_compression=live_load_fraction_in_compression,
                            cavity_mm=cavity_mm,
                            compression_bearing_length_mm=compression_bearing_length_mm,
                        )
                    except Exception as exc:
                        reference_cache[ref_key] = {"Error": str(exc), "k secant (kN/m)": np.nan}
                ref = reference_cache[ref_key]
                k_ref_same = float(ref.get("k secant (kN/m)", np.nan))

                for _, a in assemblies.iterrows():
                    case_no += 1
                    trial = dict(base)
                    trial.update({
                        "Storey": 1,
                        "Floor lateral force (kN)": V,
                        "Height (m)": H,
                        "Wall length (m)": L,
                        "Panel Type": str(a["Panel Type"]),
                        "Panel thickness (mm)": float(a["Panel thickness (mm)"]),
                        "Panel sides": str(a["Panel sides"]),
                        "Nail spacing (mm)": float(a["Nail spacing (mm)"]),
                    })
                    common = {
                        "Study ID": sid,
                        "Case ID": f"C{case_no:06d}",
                        "Assembly ID": str(a["Assembly ID"]),
                        "Model version": MODEL_VERSION,
                        "Panel Type": str(a["Panel Type"]),
                        "Panel thickness (mm)": float(a["Panel thickness (mm)"]),
                        "Panel sides": str(a["Panel sides"]),
                        "Nail diameter (mm)": float(base["Nail diameter (mm)"]),
                        "Nail spacing (mm)": float(a["Nail spacing (mm)"]),
                        "Height (m)": H,
                        "Wall length (m)": L,
                        "Aspect ratio H/L": ar,
                        "Demand mode": demand_mode,
                        "Demand level": level,
                        "Wall force V (kN)": V,
                        "Unit shear V/L (kN/m)": V / L,
                        "Species": str(base["Species"]),
                        "Grade": str(base["Grade"]),
                        "Stud size": str(base["Stud size"]),
                        "Chord studs / end": int(base["Chord studs / end"]),
                        "Rod model": str(base["Rod model"]),
                        "Take-up device": str(base["Take-up device"]),
                        "Service dead line load (kN/m)": float(base["Service dead line load (kN/m)"]),
                        "Service live line load (kN/m)": float(base["Service live line load (kN/m)"]),
                        "Reference same-condition k (kN/m)": k_ref_same,
                        "Global baseline k (kN/m)": global_k_ref,
                        "Bv panel pair exists": True,
                        "Strength-table screen": "NOT PERFORMED - verify permitted wall assembly separately",
                    }
                    try:
                        r = _analyze_single_storey(
                            trial,
                            live_load_fraction_in_compression=live_load_fraction_in_compression,
                            cavity_mm=cavity_mm,
                            compression_bearing_length_mm=compression_bearing_length_mm,
                        )
                        out = _curated_result_fields(r)
                        k = float(r["k secant (kN/m)"])
                        common.update(out)
                        common.update({
                            "k/L (kN/m²)": k / L,
                            "k / reference same condition": k / k_ref_same if math.isfinite(k_ref_same) and k_ref_same > 0 else np.nan,
                            "k / global baseline": k / global_k_ref if global_k_ref > 0 else np.nan,
                            "Analysis status": "OK",
                            "Error": "",
                        })
                    except Exception as exc:
                        common.update({
                            "k/L (kN/m²)": np.nan,
                            "k / reference same condition": np.nan,
                            "k / global baseline": np.nan,
                            "Analysis status": "ERROR",
                            "Error": str(exc),
                        })
                    rows.append(common)

    data = pd.DataFrame(rows)

    baseline_input = pd.DataFrame([{
        **base,
        "Aspect ratio H/L": base_ar,
        "Unit shear V/L (kN/m)": base_force / base_l,
        "Baseline k (kN/m)": global_k_ref,
        "Baseline Δ total (mm)": float(base_mech["Δ total inter-storey (mm)"]),
        "Model version": MODEL_VERSION,
    }])

    metadata = pd.DataFrame([
        {"Field": "Study ID", "Value": sid},
        {"Field": "Model version", "Value": MODEL_VERSION},
        {"Field": "Study design", "Value": "Full factorial: Assembly × Height × Aspect ratio × Demand"},
        {"Field": "Demand mode", "Value": demand_mode},
        {"Field": "Live-load fraction in compression", "Value": float(live_load_fraction_in_compression)},
        {"Field": "Symmetric cavity allowance (mm)", "Value": float(cavity_mm)},
        {"Field": "Compression bearing length (mm)", "Value": float(compression_bearing_length_mm)},
        {"Field": "Panel validity rule", "Value": "Panel type/thickness must exist in embedded SHEATHING_BV table"},
        {"Field": "Strength check", "Value": "Not performed; nail spacing and assembly resistance must be verified separately"},
        {"Field": "Reference normalization", "Value": "k/reference same condition uses current Wood Wall Lab baseline construction at the same H, L and V"},
    ])

    ok = data[data["Analysis status"].eq("OK")].copy()
    summary_rows = [
        {"Metric": "Planned cases", "Value": int(len(data))},
        {"Metric": "Successful cases", "Value": int(len(ok))},
        {"Metric": "Failed cases", "Value": int(len(data) - len(ok))},
        {"Metric": "Assemblies", "Value": int(len(assemblies))},
        {"Metric": "Heights", "Value": int(len(heights))},
        {"Metric": "Aspect ratios", "Value": int(len(ars))},
        {"Metric": "Demand levels", "Value": int(len(levels))},
    ]
    if not ok.empty:
        for col, label in [
            ("k secant (kN/m)", "Secant k"),
            ("k / reference same condition", "Normalized k at same condition"),
            ("Δ total inter-storey (mm)", "Total deflection"),
        ]:
            summary_rows.extend([
                {"Metric": f"{label} - min", "Value": float(ok[col].min())},
                {"Metric": f"{label} - median", "Value": float(ok[col].median())},
                {"Metric": f"{label} - max", "Value": float(ok[col].max())},
            ])
    summary = pd.DataFrame(summary_rows)

    dictionary = pd.DataFrame([
        {"Column": "Assembly ID", "Meaning": "Traceable sheathing/side/nail-spacing combination used in the study."},
        {"Column": "Aspect ratio H/L", "Meaning": "Wall height divided by wall length. Wall length is generated as H/(H/L)."},
        {"Column": "Demand level", "Meaning": "The sweep value interpreted according to Demand mode."},
        {"Column": "Wall force V (kN)", "Meaning": "Service-level total lateral force used by the mechanics calculation."},
        {"Column": "Unit shear V/L (kN/m)", "Meaning": "Service wall force divided by wall length."},
        {"Column": "Δ bending (mm)", "Meaning": "Flexural contribution to inter-storey wall deformation."},
        {"Column": "Δ panel shear (mm)", "Meaning": "Panel shear contribution using effective Bv."},
        {"Column": "Δ nail slip (mm)", "Meaning": "Fastener-slip contribution using the validated nail-slip relationship."},
        {"Column": "Δ anchorage (mm)", "Meaning": "Lateral deformation contribution from rod/take-up/bearing anchorage response."},
        {"Column": "Δ total inter-storey (mm)", "Meaning": "Sum of mechanics deformation components for the one-storey case."},
        {"Column": "k secant (kN/m)", "Meaning": "Service secant stiffness V/Δ."},
        {"Column": "k/L (kN/m²)", "Meaning": "Stiffness normalized by wall length; useful for testing k ∝ L assumptions."},
        {"Column": "Reference same-condition k (kN/m)", "Meaning": "Baseline construction recalculated at the same H, L and V as the case."},
        {"Column": "k / reference same condition", "Meaning": "Construction-only stiffness ratio after controlling for H, L and V."},
        {"Column": "k / global baseline", "Meaning": "Case stiffness relative to the exact current Wood Wall Lab baseline."},
        {"Column": "Bending/Panel shear/Nail slip/Anchorage fraction", "Meaning": "Each component divided by total deformation."},
        {"Column": "Rod utilization T/Tr", "Meaning": "Service tension-chord force divided by embedded rod resistance; use as a research flag, not a complete design check."},
        {"Column": "Strength-table screen", "Meaning": "Reminder that this atlas does not establish code-permitted shearwall resistance for the selected nailing."},
        {"Column": "Analysis status", "Meaning": "OK when mechanics calculation completed; ERROR rows preserve the error text for auditability."},
    ])

    return {
        "data": data,
        "assemblies": assemblies,
        "baseline": baseline_input,
        "metadata": metadata,
        "summary": summary,
        "data_dictionary": dictionary,
    }


def parametric_workbook_bytes(study: Dict[str, pd.DataFrame]) -> bytes:
    """Create a self-contained workbook intended for office archive or ChatGPT upload."""
    bio = BytesIO()
    readme = pd.DataFrame({
        "README": [
            "WOOD SHEARWALL PARAMETRIC ATLAS - ANALYSIS-READY EXPORT",
            "Upload this workbook to ChatGPT and ask for a parametric-study analysis, presentation graphs, engineering comments and conclusions.",
            "Primary raw-data sheet: Parametric_Data.",
            "Use k / reference same condition to isolate construction effects while controlling geometry and demand.",
            "Use k/L versus H/L or L to test the common k proportional to wall-length assumption.",
            "Use deformation fractions to explain WHY stiffness changes, not only how much it changes.",
            "Important: the mechanics atlas does not perform a CSA shearwall resistance/approved-assembly screen for the selected nail spacing. Verify design validity separately before office design use.",
        ]
    })
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        study["metadata"].to_excel(writer, sheet_name="Study_Metadata", index=False)
        study["baseline"].to_excel(writer, sheet_name="Baseline_Design", index=False)
        study["assemblies"].to_excel(writer, sheet_name="Assembly_Catalog", index=False)
        study["summary"].to_excel(writer, sheet_name="Study_Summary", index=False)
        study["data_dictionary"].to_excel(writer, sheet_name="Data_Dictionary", index=False)
        study["data"].to_excel(writer, sheet_name="Parametric_Data", index=False)
    return bio.getvalue()
