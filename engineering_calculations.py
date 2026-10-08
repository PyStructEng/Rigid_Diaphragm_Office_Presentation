"""Traceable equation views and *independent unit audit* for the diaphragm solver.

The solver remains the source of truth. Handcalcs only renders a transparent
re-evaluation of individual scalar arithmetic steps; discrepancies raise.
ForAllPeople is used only at the reporting/validation boundary. Its Physical
objects never enter the NumPy/pandas structural matrix solver.

Unit conventions: x,y,L [m]; F,V [kN]; k [kN/m]; J [kN.m]; M [kN.m].
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd


# Keep import inside the function factory so the existing solver and baseline
# calculation-sheet tests still run in environments before dependencies install.
def equation_functions():
    from handcalcs.decorator import handcalc

    @handcalc(precision=3)
    def stiffness_share(k_i, K):
        s_i = k_i / K
        return locals()

    @handcalc(precision=3)
    def centre_rigidity(n, K):
        x_CR = n / K
        return locals()

    @handcalc(precision=3)
    def centre_rigidity_y(n, K):
        y_CR = n / K
        return locals()

    @handcalc(precision=3)
    def eccentricity(x_CM, x_CR):
        e = x_CM - x_CR
        return locals()

    @handcalc(precision=3)
    def torsional_contribution(k_x, k_y, x_bar, y_bar):
        J_i = k_y * x_bar**2 + k_x * y_bar**2
        return locals()

    @handcalc(precision=3)
    def accidental_moment(e_real, a, D, sign, V, sense):
        e_acc = sign * a * D
        e_total = e_real + e_acc
        M_z = sense * V * e_total
        return locals()

    @handcalc(precision=3)
    def direct_shear(V, k_i, K):
        V_direct = V * k_i / K
        return locals()

    @handcalc(precision=3)
    def torsional_shear(M_z, k_x, k_y, x_bar, y_bar, J):
        V_x_t = -M_z * k_x * y_bar / J
        V_y_t = M_z * k_y * x_bar / J
        return locals()

    @handcalc(precision=3)
    def combined_x_shear(V_x_d, V_x_t, V_local, L_wall):
        V_diaphragm = V_x_d + V_x_t
        V_wall = V_diaphragm + V_local
        v = abs(V_wall) / L_wall
        return locals()

    @handcalc(precision=3)
    def combined_y_shear(V_y_d, V_y_t, V_local, L_wall):
        V_diaphragm = V_y_d + V_y_t
        V_wall = V_diaphragm + V_local
        v = abs(V_wall) / L_wall
        return locals()

    @handcalc(precision=3)
    def stiffness_from_deflection(V, delta):
        k_secant = abs(V) / delta
        return locals()

    return {
        "share": stiffness_share, "cr": centre_rigidity, "cr_y": centre_rigidity_y,
        "ecc": eccentricity, "j": torsional_contribution,
        "moment": accidental_moment, "direct": direct_shear,
        "torsion": torsional_shear, "combine_x": combined_x_shear,
        "combine_y": combined_y_shear, "secant": stiffness_from_deflection,
    }


def equation_by_sections(result: dict, settings: dict, *, tol=1e-7) -> dict[int, list[tuple[str, str]]]:
    """Render Handcalcs from scalar values; assert agreement with current solver.

    Mapping keys correspond to existing numbered calculation sections (1--10).
    Each LaTeX block is genuine Handcalcs output from the shown arithmetic.
    """
    fn = equation_functions()
    wp = result["wall_properties"].reset_index(drop=True)
    p = result["properties"]
    out: dict[int, list[tuple[str, str]]] = {i: [] for i in range(1, 11)}

    def add(section: int, caption: str, func: str, expected: dict[str, float], *args: float):
        latex, scope = fn[func](*[float(x) for x in args])
        for key, target in expected.items():
            measured = float(scope[key])
            if not math.isclose(measured, float(target), rel_tol=tol, abs_tol=tol):
                raise ValueError(f"Handcalcs mismatch for {caption}: {key} = {measured}, solver = {target}")
        # Streamlit st.latex expects the math *body*, without the surrounding display tags.
        out[section].append((caption, latex.strip()))

    add(3, "X centre of rigidity (Y-direction walls)", "cr", {"x_CR": p["Xcr"]},
        sum(float(r["ky (kN/m)"]) * float(r["x (m)"]) for _, r in wp.iterrows()), p["sum_ky"])
    add(3, "Y centre of rigidity (X-direction walls)", "cr_y", {"y_CR": p["Ycr"]},
        sum(float(r["kx (kN/m)"]) * float(r["y (m)"]) for _, r in wp.iterrows()), p["sum_kx"])
    add(3, "Natural X eccentricity", "ecc", {"e": p["ex_signed"]}, settings["Xcm"], p["Xcr"])
    add(3, "Natural Y eccentricity", "ecc", {"e": p["ey_signed"]}, settings["Ycm"], p["Ycr"])

    for _, r in wp.iterrows():
        name = str(r["Wall Name"])
        kx = float(r["kx (kN/m)"])
        ky = float(r["ky (kN/m)"])
        xbar, ybar = float(r["xbar (m)"]), float(r["ybar (m)"])
        add(2, f"{name} — directional stiffness fraction", "share",
            {"s_i": (kx/p["sum_kx"] if kx else ky/p["sum_ky"])},
            (kx if kx else ky), (p["sum_kx"] if kx else p["sum_ky"]))
        add(4, f"{name} — contribution to J [kN.m]", "j",
            {"J_i": ky*xbar*xbar + kx*ybar*ybar}, kx, ky, xbar, ybar)

    for direction in ("X", "Y"):
        F = float(settings["Fx"] if direction == "X" else settings["Fy"])
        D = float(settings["Ly"] if direction == "X" else settings["Lx"])
        e_real = float(p["ey_signed"] if direction == "X" else p["ex_signed"])
        sense = -1 if direction == "X" else +1
        for sign, signed in (("−", -1), ("+", +1)):
            frame = result["x_cases"] if direction == "X" else result["y_cases"]
            case = frame[frame["Accidental Sign"] == sign].reset_index(drop=True)
            if len(case) != len(wp):
                raise ValueError("Handcalcs: load case does not match wall count")
            mz = float(case.iloc[0]["Mz (kN·m)"])
            add(5, f"{direction} load, {sign} accidental eccentricity", "moment",
                {"M_z": mz}, e_real, settings["acc"], D, signed, F, sense)
            for i, r in wp.iterrows():
                c = case.iloc[i]
                if str(c["Wall Name"]) != str(r["Wall Name"]):
                    raise ValueError("Handcalcs: wall order does not match")
                name = str(r["Wall Name"])
                kx, ky = float(r["kx (kN/m)"]), float(r["ky (kN/m)"])
                if (direction == "X" and kx) or (direction == "Y" and ky):
                    add(6, f"{direction} load, {name}: direct wall force", "direct",
                        {"V_direct": c["Direct Vx (kN)"] if direction == "X" else c["Direct Vy (kN)"]},
                        F, kx if direction == "X" else ky, p["sum_kx"] if direction == "X" else p["sum_ky"])
                add(7, f"{direction} {sign}, {name}: torsional wall forces", "torsion",
                    {"V_x_t": c["Torsion Vx (kN)"], "V_y_t": c["Torsion Vy (kN)"]},
                    mz, kx, ky, r["xbar (m)"], r["ybar (m)"], p["J"])
                dr = str(r["Direction"])
                if dr == "X":
                    vx_d = float(c["Direct Vx (kN)"])
                    vxt = float(c["Torsion Vx (kN)"])
                    add(7, f"{direction} {sign}, {name}: resulting parallel force and unit shear", "combine_x",
                        {"V_diaphragm": c["Diaphragm wall-parallel V (kN)"],
                         "V_wall": c["Wall-parallel V (kN)"], "v": c["v = |V|/Lwall (kN/m)"]},
                        vx_d, vxt, c["Local wall force (kN)"], r["Wall Length (m)"])
                elif dr == "Y":
                    vy_d = float(c["Direct Vy (kN)"])
                    vyt = float(c["Torsion Vy (kN)"])
                    add(7, f"{direction} {sign}, {name}: resulting parallel force and unit shear", "combine_y",
                        {"V_diaphragm": c["Diaphragm wall-parallel V (kN)"],
                         "V_wall": c["Wall-parallel V (kN)"], "v": c["v = |V|/Lwall (kN/m)"]},
                        vy_d, vyt, c["Local wall force (kN)"], r["Wall Length (m)"])
    return out


def audit_units(result: dict, settings: dict, *, rel_tol=1e-8, abs_tol=1e-7) -> pd.DataFrame:
    """Check important rigid-diaphragm results using dimension-bearing quantities.

    Reads ForAllPeople Physical.value only for SI-base checks and explicitly
    divides by target units for readable output. No units are implicit.
    Does not claim to validate the structural assumptions or code compliance.
    """
    import forallpeople as si
    si.environment("default", top_level=False)
    N = si.N
    m = si.m
    kN = 1000 * N
    k = kN/m
    M = kN*m
    wp = result["wall_properties"].reset_index(drop=True)
    p = result["properties"]
    records = []

    def number(q, u):
        if not hasattr(q, "dimensions") or q.dimensions != u.dimensions:
            raise AssertionError(f"Dimensional mismatch: {getattr(q, 'dimensions', None)} vs {u.dimensions}")
        # Unit cancellation makes a dimensionless value; NEVER float(Physical)
        # before checking dimension or selecting the target unit.
        r = q/u
        if hasattr(r, "dimensions"):
            if any(r.dimensions):
                raise AssertionError("Unit cancellation failed")
            return float(r.value)
        return float(r)

    def check(name, q, unit, ref, label):
        val = number(q, unit)
        passed = math.isclose(val, float(ref), rel_tol=rel_tol, abs_tol=abs_tol)
        records.append({"Quantity": name, "Unit": label,
            "Unit-aware calculation": val, "Existing solver": float(ref),
            "Absolute difference": abs(val-float(ref)), "Pass": passed})

    sx = 0*k
    sy = 0*k
    wx = 0*kN
    wy = 0*kN
    for _, r in wp.iterrows():
        kx = float(r["kx (kN/m)"]) * k
        ky = float(r["ky (kN/m)"]) * k
        sx += kx
        sy += ky
        wx += ky * (float(r["x (m)"])*m)
        wy += kx * (float(r["y (m)"])*m)
    check("Total X stiffness", sx, k, p["sum_kx"], "kN/m")
    check("Total Y stiffness", sy, k, p["sum_ky"], "kN/m")
    Xcr = wx/sy
    Ycr = wy/sx
    check("Xcr", Xcr, m, p["Xcr"], "m")
    check("Ycr", Ycr, m, p["Ycr"], "m")
    J = 0*M
    for _, r in wp.iterrows():
        kx = float(r["kx (kN/m)"])*k
        ky = float(r["ky (kN/m)"])*k
        dx = float(r["x (m)"])*m-Xcr
        dy = float(r["y (m)"])*m-Ycr
        J_i = ky*dx**2+kx*dy**2
        J += J_i
        check(f"{r['Wall Name']} — J contribution", J_i, M,
              float(r["ky*xbar^2 (kN·m)"])+float(r["kx*ybar^2 (kN·m)"]), "kN.m")
    check("J torsional stiffness", J, M, p["J"], "kN.m")

    for direction in ("X", "Y"):
        V = float(settings["Fx"] if direction == "X" else settings["Fy"])*kN
        e_real = (float(settings["Ycm"])*m-Ycr) if direction=="X" else (float(settings["Xcm"])*m-Xcr)
        D = float(settings["Ly"] if direction=="X" else settings["Lx"])*m
        sense = -1 if direction=="X" else +1
        for sign, fac in (("−",-1),("+",+1)):
            cases = result["x_cases"] if direction=="X" else result["y_cases"]
            case = cases[cases["Accidental Sign"] == sign].reset_index(drop=True)
            mz = sense*V*(e_real+fac*float(settings["acc"])*D)
            check(f"{direction} {sign} torsional moment", mz, M,
                  case.iloc[0]["Mz (kN·m)"], "kN.m")
            for i, r in wp.iterrows():
                c = case.iloc[i]
                kx = float(r["kx (kN/m)"])*k
                ky = float(r["ky (kN/m)"])*k
                dx = float(r["x (m)"])*m-Xcr
                dy = float(r["y (m)"])*m-Ycr
                Vx_t = -mz*kx*dy/J
                Vy_t = mz*ky*dx/J
                check(f"{direction} {sign} {r['Wall Name']} Vx torsion", Vx_t, kN,
                      c["Torsion Vx (kN)"], "kN")
                check(f"{direction} {sign} {r['Wall Name']} Vy torsion", Vy_t, kN,
                      c["Torsion Vy (kN)"], "kN")
                Vx_d = (V*kx/sx) if direction=="X" else 0*kN
                Vy_d = (V*ky/sy) if direction=="Y" else 0*kN
                check(f"{direction} {sign} {r['Wall Name']} Vx direct", Vx_d, kN,
                      c["Direct Vx (kN)"], "kN")
                check(f"{direction} {sign} {r['Wall Name']} Vy direct", Vy_d, kN,
                      c["Direct Vy (kN)"], "kN")
                V_parallel = (Vx_d + Vx_t) if str(r["Direction"]) == "X" else (Vy_d + Vy_t)
                check(f"{direction} {sign} {r['Wall Name']} diaphragm parallel", V_parallel, kN,
                      c["Diaphragm wall-parallel V (kN)"], "kN")
                V_final = V_parallel + float(c["Local wall force (kN)"])*kN
                check(f"{direction} {sign} {r['Wall Name']} total parallel", V_final, kN,
                      c["Wall-parallel V (kN)"], "kN")
                sign_force = -1.0 if number(V_final, kN) < 0 else 1.0
                v = (sign_force*V_final)/(float(r["Wall Length (m)"])*m)
                check(f"{direction} {sign} {r['Wall Name']} unit shear", v, k,
                      c["v = |V|/Lwall (kN/m)"], "kN/m")
    return pd.DataFrame(records)


def wood_secant_trace(V_kN: float, delta_mm: float, k_expected: float):
    """Human-readable Handcalcs equation + independent dimensional check.

    Works for the single-storey and stacked-wall presentations; kN / m.
    """
    if delta_mm <= 0 or not math.isfinite(delta_mm):
        raise ValueError("Secant trace requires a positive total deformation in mm")
    if not all(math.isfinite(float(v)) for v in (V_kN, k_expected)):
        raise ValueError("Non-finite wall shear or stiffness")
    fn = equation_functions()
    latex, values = fn["secant"](float(V_kN), float(delta_mm)/1000.0)
    k_calc = float(values["k_secant"])
    if not math.isclose(k_calc, float(k_expected), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Secant stiffness mismatch: expression {k_calc:g}, solver {k_expected:g} kN/m")
    import forallpeople as si
    si.environment("default", top_level=False)
    KN = 1000*si.N
    metre = si.m
    dimensional_k = abs(float(V_kN))*KN / (float(delta_mm)/1000.0*metre)
    required_unit = KN / metre
    if dimensional_k.dimensions != required_unit.dimensions:
        raise AssertionError("Secant stiffness has inconsistent units")
    ratio = dimensional_k/required_unit
    value = float(ratio.value) if hasattr(ratio,"value") else float(ratio)
    if not math.isclose(value,float(k_expected),rel_tol=1e-6,abs_tol=1e-6):
        raise ValueError("Unit-aware wood-wall stiffness differs from solver")
    return latex, k_calc


def render_wood_secant_panel(V_kN: float, delta_mm: float, k_expected: float):
    """Small reusable Streamlit widget; deliberately does not change the solver."""
    import streamlit as st
    with st.expander("View secant stiffness calculation and units", expanded=False):
        st.caption("V / total deflection = secant stiffness. This is a report of the existing wood mechanics calculation, not a separate design method.")
        try:
            latex, value = wood_secant_trace(V_kN, delta_mm, k_expected)
            st.latex(latex.strip().removeprefix(r"\[").removesuffix(r"\]"))
            st.success(f"Handcalcs and ForAllPeople checks agree: k = {value:,.2f} kN/m")
        except ImportError as exc:
            st.warning(f"Install Handcalcs and ForAllPeople to see the equation: {exc}")
        except Exception as exc:
            st.error(f"Stiffness calculation trace could not be validated: {exc}")
