"""Per-wall assembly records and *trial* service deformation checks.

Engineering scope: this module is NOT a CSA O86 capacity calculator, and not a
multi-level compatible nonlinear wall element.  Wall mechanics comes from the
existing ``wood_shearwall_mechanics`` research model, evaluated as an ISOLATED
ONE-STOREY wall at an explicitly supplied trial shear.  Values are suitable for
transparent design comparisons, not automatic seismic compliance declarations.
"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from wood_shearwall_mechanics import (
    SHEATHING_BV, LUMBER_E, get_bv, get_lumber_e,
    get_rod_properties, get_takeup, single_storey_response_from_row,
)

ASSEMBLY_FIELDS = [
    "Panel Type", "Panel thickness (mm)", "Panel sides", "Nail diameter (mm)",
    "Nail edge spacing (mm)", "Nail field spacing (mm)", "Nail length (mm)",
    "Species", "Grade", "Stud size", "Chord studs / end", "Rod model",
    "Take-up device", "Source / notes",
]
ASSEMBLY_COLS = ["Assembly ID"] + ASSEMBLY_FIELDS
WALL_FIELDS = ["Storey", "Wall Name", "Assembly ID", "Wall length (m)",
    "Trial shear (kN)", "Service dead line load (kN/m)",
    "Service live line load (kN/m)", "Factored shear demand (kN)",
    "Verified shear resistance (kN)", "Factored tension demand (kN)",
    "Verified tension resistance (kN)", "Design basis / notes"]
SCHEMA = "wood-wall-design-studio-v1"


def default_assemblies() -> pd.DataFrame:
    rows = [
        ("SW-OSB-1", "OSB", 12.0, "S.S", 3.25, 150.0, 300.0, 63.0, "S-P-F", "No.1/No.2", "2x6", 4, "SR8H", "CTUD87"),
        ("SW-CSP-1", "CSP", 12.5, "S.S", 3.25, 150.0, 300.0, 63.0, "S-P-F", "No.1/No.2", "2x6", 4, "SR8H", "CTUD87"),
        ("SW-DFP-1", "DFP", 12.5, "S.S", 3.25, 150.0, 300.0, 63.0, "D.Fir-L", "No.1/No.2", "2x6", 4, "SR8H", "CTUD87"),
    ]
    return pd.DataFrame([dict(zip(ASSEMBLY_COLS[:-1], r), **{"Source / notes": "Editable demo, verify specifications independently"}) for r in rows])


def default_wall_schedule(building_walls: pd.DataFrame, assembly_id: str = "SW-OSB-1") -> pd.DataFrame:
    """Generate one stable ID per (storey, wall line) from current global layout."""
    schedule = []
    for _, r in building_walls.iterrows():
        schedule.append({
            "Storey": int(r["Storey"]), "Wall Name": str(r["Wall Name"]),
            "Assembly ID": assembly_id, "Wall length (m)": 4.8,
            "Trial shear (kN)": 40.0,
            "Service dead line load (kN/m)": 0.0,
            "Service live line load (kN/m)": 0.0,
            "Factored shear demand (kN)": 0.0,
            "Verified shear resistance (kN)": 0.0,
            "Factored tension demand (kN)": 0.0,
            "Verified tension resistance (kN)": 0.0,
            "Design basis / notes": "",
        })
    return pd.DataFrame(schedule, columns=WALL_FIELDS)


def _finite_nonnegative(v: Any, label: str, strictly: bool = False) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be numeric") from None
    if not math.isfinite(x) or x < 0 or (strictly and x == 0):
        raise ValueError(f"{label} must be {'positive' if strictly else 'nonnegative'} and finite")
    return x


def validate_assemblies(assemblies: pd.DataFrame) -> pd.DataFrame:
    missing = set(ASSEMBLY_COLS) - set(assemblies.columns)
    if missing:
        raise ValueError(f"Missing assembly fields: {sorted(missing)}")
    if not (1 <= len(assemblies) <= 800):
        raise ValueError("Provide between 1 and 800 assemblies")
    a = assemblies[ASSEMBLY_COLS].copy().reset_index(drop=True)
    for field in ["Assembly ID", "Panel Type", "Panel sides", "Species", "Grade", "Stud size", "Rod model", "Take-up device"]:
        if a[field].isna().any() or (a[field].astype(str).str.strip() == "").any():
            raise ValueError(f"Assembly {field} must be filled for every row")
        a[field] = a[field].astype(str).str.strip()
    if a["Assembly ID"].duplicated().any():
        raise ValueError("Assembly ID must be unique")
    for i, r in a.iterrows():
        prefix = f"Assembly {r['Assembly ID']}"
        if r["Panel sides"] not in ("S.S", "B.S"):
            raise ValueError(f"{prefix}: panel sides must be S.S or B.S")
        if str(r["Stud size"]) not in ("2x4", "2x6"):
            raise ValueError(f"{prefix}: stud size must be 2x4 or 2x6")
        for f in ("Panel thickness (mm)", "Nail diameter (mm)", "Nail edge spacing (mm)", "Nail field spacing (mm)", "Nail length (mm)"):
            a.at[i, f] = _finite_nonnegative(r[f], f"{prefix}: {f}", strictly=True)
        n = _finite_nonnegative(r["Chord studs / end"], f"{prefix}: chord studs", strictly=True)
        if not float(n).is_integer():
            raise ValueError(f"{prefix}: chord studs per end must be whole")
        a.at[i, "Chord studs / end"] = int(n)
        get_bv(r["Panel Type"], float(r["Panel thickness (mm)"]), r["Panel sides"])
        get_lumber_e(r["Species"], r["Grade"])
        get_rod_properties(r["Rod model"])
        get_takeup(r["Take-up device"])
    a["Source / notes"] = a["Source / notes"].fillna("").astype(str)
    return a


def validate_schedule(schedule: pd.DataFrame, assemblies: pd.DataFrame, global_walls: pd.DataFrame, floors: pd.DataFrame) -> pd.DataFrame:
    missing = set(WALL_FIELDS) - set(schedule.columns)
    if missing:
        raise ValueError(f"Missing wall design fields: {sorted(missing)}")
    d = schedule[WALL_FIELDS].copy().reset_index(drop=True)
    if d.empty or len(d) > 600:
        raise ValueError("Provide between 1 and 600 designed segments")
    a = validate_assemblies(assemblies)
    d["Wall Name"] = d["Wall Name"].astype(str).str.strip()
    d["Assembly ID"] = d["Assembly ID"].astype(str).str.strip()
    for i, r in d.iterrows():
        s = _finite_nonnegative(r["Storey"], "Storey", strictly=True)
        if not s.is_integer():
            raise ValueError("Storey must be an integer")
        d.at[i,"Storey"] = int(s)
        if not r["Wall Name"]:
            raise ValueError("Wall Name cannot be blank")
        for f in WALL_FIELDS[3:-1]:
            d.at[i,f] = _finite_nonnegative(r[f], f"Storey {int(s)} {r['Wall Name']}: {f}", strictly=f=="Wall length (m)")
    d["Storey"] = d["Storey"].astype(int)
    if d.duplicated(["Storey", "Wall Name"]).any():
        raise ValueError("Each Storey / Wall Name pair must be unique")
    if not d["Assembly ID"].isin(a["Assembly ID"]).all():
        missing_ids = sorted(set(d["Assembly ID"]) - set(a["Assembly ID"]))
        raise ValueError(f"Undefined Assembly IDs: {missing_ids}")
    actual = set(map(tuple, global_walls[["Storey", "Wall Name"]].itertuples(index=False,name=None)))
    designed = set(map(tuple, d[["Storey", "Wall Name"]].itertuples(index=False,name=None)))
    if actual != designed:
        raise ValueError(f"Design schedule does not match global model. Missing {sorted(actual-designed)[:5]}; extra {sorted(designed-actual)[:5]}. Use Sync wall IDs.")
    if not d["Storey"].isin(floors["Storey"]).all():
        raise ValueError("Wall references an undefined floor")
    d["Design basis / notes"] = d["Design basis / notes"].fillna("").astype(str)
    return d


def sync_schedule(existing: pd.DataFrame, global_walls: pd.DataFrame) -> pd.DataFrame:
    """Retain all matched segment designs; add new segments and drop deleted ones."""
    fresh = default_wall_schedule(global_walls)
    if existing is None or existing.empty:
        return fresh
    old = existing.drop_duplicates(["Storey", "Wall Name"]).set_index(["Storey", "Wall Name"])
    for i, row in fresh.iterrows():
        key = (int(row["Storey"]), str(row["Wall Name"]))
        if key in old.index:
            for f in WALL_FIELDS[2:]:
                if f in old.columns:
                    fresh.at[i,f] = old.at[key,f]
    return fresh[WALL_FIELDS]


def strength_review(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Only compare user-verified factored resistance and factored demand.

    Zeros represent missing inputs; NEVER label a wall compliant automatically.
    """
    comparisons = {}
    for name, demand, capacity in (("Shear", "Factored shear demand (kN)", "Verified shear resistance (kN)"),
                                   ("Hold-down tension", "Factored tension demand (kN)", "Verified tension resistance (kN)")):
        d, c = float(row[demand]), float(row[capacity])
        comparisons[f"{name} utilization"] = (d/c if c>0 and d>0 else math.nan)
        comparisons[f"{name} status"] = ("NOT CHECKED" if c<=0 or d<=0 else ("Over 1.0 — REVIEW" if d>c else "Within supplied capacity"))
    return comparisons


def wall_trial_response(row: pd.Series | dict[str, Any], assembly: pd.Series | dict[str, Any], height_m: float, shear_kN: float) -> dict[str, Any]:
    """Isolated single-storey trial from existing historical mechanics equations.

    The wall is considered at |shear|; no signed hold-down side or reversed-load
    hysteresis is represented.  Nonzero preload/constant seating contribution
    remains in the model.  Zero shear does NOT yield an effective stiffness.
    """
    h = _finite_nonnegative(height_m, "Storey height", strictly=True)
    shear = abs(float(shear_kN))
    if not math.isfinite(shear):
        raise ValueError("Trial shear is invalid")
    mechanics_row = {
        "Storey": 1, "Floor lateral force (kN)": shear,
        "Height (m)": h, "Wall length (m)": float(row["Wall length (m)"]),
        "Panel Type": str(assembly["Panel Type"]),
        "Panel thickness (mm)": float(assembly["Panel thickness (mm)"]),
        "Panel sides": str(assembly["Panel sides"]),
        "Nail diameter (mm)": float(assembly["Nail diameter (mm)"]),
        "Nail spacing (mm)": float(assembly["Nail edge spacing (mm)"]),
        "Species": str(assembly["Species"]), "Grade": str(assembly["Grade"]),
        "Stud size": str(assembly["Stud size"]),
        "Chord studs / end": int(assembly["Chord studs / end"]),
        "Rod model": str(assembly["Rod model"]),
        "Take-up device": str(assembly["Take-up device"]),
        "Service dead line load (kN/m)": float(row["Service dead line load (kN/m)"]),
        "Service live line load (kN/m)": float(row["Service live line load (kN/m)"]),
    }
    if shear <= 1e-9:
        return {
            "Trial |V| (kN)": 0., "Trial deflection (mm)": math.nan,
            "Trial k secant (kN/m)": math.nan, "Trial drift (%)": math.nan,
            "Drift review flag": "No trial demand",
            "Nail force (N)": 0., "Nail slip (mm)": math.nan,
            "Rod service tension estimate (kN)": math.nan, "Rod reference Tr (kN)": math.nan,
            "Bending (mm)": math.nan, "Panel shear (mm)": math.nan,
            "Nail slip contribution (mm)": math.nan, "Anchorage contribution (mm)": math.nan,
            "Analysis scope": "No trial deformation at zero shear; secant undefined",
        }
    resp = single_storey_response_from_row(mechanics_row)
    delta = float(resp["Δ total inter-storey (mm)"])
    k = 1000 * shear/delta if shear>1e-9 and delta>1e-9 else math.nan
    if shear > 1e-9 and (not math.isfinite(k) or k<=0):
        raise ValueError("Mechanics returned no finite positive secant stiffness")
    return {
        "Trial |V| (kN)": shear,
        "Trial deflection (mm)": delta,
        "Trial k secant (kN/m)": k,
        "Trial drift (%)": 100*delta/(h*1000) if shear>1e-9 else math.nan,
        "Drift review flag": ("HIGH trial drift — investigate assumptions/overload"
                              if 100*delta/(h*1000) > 2.0 else "Trial deformation only"),
        "Nail force (N)": float(resp["Force per nail (N)"]),
        "Nail slip (mm)": float(resp["en (mm)"]),
        "Rod service tension estimate (kN)": float(resp["Tension chord force (kN)"]),
        "Rod reference Tr (kN)": float(resp["Tr (kN)"]),
        "Bending (mm)": float(resp["Δ bending (mm)"]),
        "Panel shear (mm)": float(resp["Δ panel shear (mm)"]),
        "Nail slip contribution (mm)": float(resp["Δ nail slip (mm)"]),
        "Anchorage contribution (mm)": float(resp["Δ anchorage (mm)"]),
        "Analysis scope": "Isolated single-storey mechanics snapshot — NOT stacked-wall compatible",
    }


def evaluate_schedule(schedule: pd.DataFrame, assemblies: pd.DataFrame, global_walls: pd.DataFrame,
                      floors: pd.DataFrame, case_forces: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return independent wall deformation snapshots and optional manual strength checks.

    ``case_forces`` can be a single simultaneous global load-case table with
    columns Storey, Wall Name, Wall shear (kN). It cannot be an envelope.
    """
    a = validate_assemblies(assemblies)
    d = validate_schedule(schedule, a, global_walls, floors)
    lookup = a.set_index("Assembly ID")
    heights = floors.set_index("Storey")["Height (m)"]
    forces = {}
    if case_forces is not None:
        needed = {"Storey", "Wall Name", "Wall shear (kN)"}
        if not needed.issubset(case_forces):
            raise ValueError("Case force table lacks required columns")
        if case_forces.duplicated(["Storey", "Wall Name"]).any():
            raise ValueError("Select only one simultaneous force case, not a multi-case envelope")
        for _, r in case_forces.iterrows():
            forces[(int(r["Storey"]), str(r["Wall Name"]))] = float(r["Wall shear (kN)"])
        if len(forces) != len(d):
            raise ValueError("Case force table does not cover every wall segment")
    out = []
    for _, r in d.iterrows():
        key = (int(r["Storey"]), str(r["Wall Name"]))
        v = forces[key] if case_forces is not None else float(r["Trial shear (kN)"])
        arow = lookup.loc[str(r["Assembly ID"])]
        trial = wall_trial_response(r, arow, float(heights.loc[key[0]]), v)
        out.append({"Storey": key[0], "Wall Name":key[1],"Assembly ID":r["Assembly ID"],
                    "Source shear (kN)":v, "Wall length (m)":float(r["Wall length (m)"]),
                    "Panel":f"{arow['Panel Type']} {arow['Panel thickness (mm)']} mm",
                    "Nail edge (mm)":float(arow["Nail edge spacing (mm)"]),
                    **trial, **strength_review(r)})
    return pd.DataFrame(out)


def export_design_json(assemblies:pd.DataFrame,schedule:pd.DataFrame)->bytes:
    validate_assemblies(assemblies)
    data={"schema":SCHEMA,"assemblies":assemblies[ASSEMBLY_COLS].where(pd.notna(assemblies), None).to_dict(orient="records"),
          "walls":schedule[WALL_FIELDS].where(pd.notna(schedule), None).to_dict(orient="records")}
    return json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2).encode("utf-8")


def import_design_json(content: bytes, global_walls:pd.DataFrame, floors:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    if len(content)>2_000_000:
        raise ValueError("Design JSON exceeds 2 MB")
    payload=json.loads(content.decode("utf-8"))
    if not isinstance(payload,dict) or payload.get("schema")!=SCHEMA:
        raise ValueError("Incompatible wall design JSON schema")
    a=validate_assemblies(pd.DataFrame(payload["assemblies"]))
    d=validate_schedule(pd.DataFrame(payload["walls"]),a,global_walls,floors)
    return a,d
