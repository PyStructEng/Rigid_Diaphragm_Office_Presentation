"""Transparent rigid-diaphragm calculation worksheet + printable PDF.

This module reads the EXISTING solver's output. It does not replace or
recalculate the engineering analysis, and so it follows the core engine's
sign conventions and treatment of local wall forces. Designed to accompany
rigid_diaphragm_core.py and the Streamlit application.

PDF dependency: reportlab>=4.0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape
import math
import re

import pandas as pd


@dataclass
class Section:
    heading: str
    notes: list[str] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)
    calculations: list[str] = field(default_factory=list)
    equations: list[tuple[str, str]] = field(default_factory=list)  # (caption, Handcalcs LaTeX)


def f(value: float, places: int = 4) -> str:
    v = float(value)
    if abs(v) < 0.5 * 10**(-places):
        v = 0.0
    return f"{v:,.{places}f}"


def build_calculation_sections(walls: pd.DataFrame, settings: dict, result: dict) -> list[Section]:
    """Build all equations from one model/solver result (no independent solver)."""
    wp = result["wall_properties"].copy().reset_index(drop=True)
    p = result["properties"]
    Lx, Ly = float(settings["Lx"]), float(settings["Ly"])
    Xcm, Ycm = float(settings["Xcm"]), float(settings["Ycm"])
    Fx, Fy = float(settings["Fx"]), float(settings["Fy"])
    a = float(settings["acc"])
    sections: list[Section] = []

    s = Section("01 | Input data and coordinate system", [
        "Purpose: distribute diaphragm forces to X- and Y-resisting elements using a rigid in-plane diaphragm model.",
        "Coordinate convention: +X to the right, +Y upward (plan). X walls resist Vx; Y walls resist Vy. Wall x/y locate their resisting axes; stiffness k is a translation spring in the wall's resisting direction.",
        "For each wall, supplied k is a secant/effective stiffness in kN/m. Wall length enters the reported unit shear, but does not alter force distribution unless k is also changed.",
        "All calculations below use signed applied forces and the same model data as the results screen. The accidental eccentricity ratio is an INPUT to be verified for the project's applicable code/edition; no code-prescribed value is inferred.",
    ])
    s.tables.append(("Building inputs", pd.DataFrame([
        ("Lx (plan dimension in X)", f(Lx,3)+" m"),
        ("Ly (plan dimension in Y)", f(Ly,3)+" m"),
        ("Xcm", f(Xcm,3)+" m"), ("Ycm", f(Ycm,3)+" m"),
        ("Fx", f(Fx,3)+" kN"), ("Fy", f(Fy,3)+" kN"),
        ("Accidental ratio, a", f(a,4)),
    ], columns=["Parameter", "Current value"])))
    s.tables.append(("Wall model", wp[["Wall Name", "Direction", "x (m)", "y (m)", "k (kN/m)", "Wall Length (m)", "Local X Force (kN)", "Local Y Force (kN)"]].copy()))
    sections.append(s)

    s = Section("02 | Directional stiffness and stiffness share", [
        "Define kx,i = ki for a wall resisting X (otherwise 0); ky,i = ki for a wall resisting Y (otherwise 0).",
        "Direct load share in each direction = stiffness in that direction / total directional stiffness. A wall oriented perpendicular to the applied force has no direct force, but it may receive torsional force.",
        "Relative stiffness is useful for initial allocation; when members are redesigned, use updated physical/effective k values and re-run the model.",
    ])
    d = wp[["Wall Name", "Direction", "kx (kN/m)", "ky (kN/m)"]].copy()
    d["kx / sum(kx)"] = wp["kx (kN/m)"] / p["sum_kx"]
    d["ky / sum(ky)"] = wp["ky (kN/m)"] / p["sum_ky"]
    s.tables.append(("Directional stiffness", d))
    s.calculations.extend([
        "Sum(kx) = " + " + ".join(f(x,2) for x in wp["kx (kN/m)"].tolist()) + " = " + f(p["sum_kx"],2) + " kN/m",
        "Sum(ky) = " + " + ".join(f(x,2) for x in wp["ky (kN/m)"].tolist()) + " = " + f(p["sum_ky"],2) + " kN/m",
    ])
    sections.append(s)

    s = Section("03 | Centre of rigidity (CR)", [
        "Xcr is weighted by the Y-direction stiffnesses and their x-coordinates; Ycr is weighted by the X-direction stiffnesses and their y-coordinates.",
        "Xcr = Sum(ky,i * xi) / Sum(ky); Ycr = Sum(kx,i * yi) / Sum(kx).",
    ])
    t = wp[["Wall Name", "kx (kN/m)", "ky (kN/m)", "x (m)", "y (m)"]].copy()
    t["ky * x (kN)"] = wp["ky (kN/m)"] * wp["x (m)"]
    t["kx * y (kN)"] = wp["kx (kN/m)"] * wp["y (m)"]
    s.tables.append(("Centre-of-rigidity contributions", t))
    s.calculations.extend([
        f"Xcr = {f(t['ky * x (kN)'].sum(),3)} / {f(p['sum_ky'],3)} = {f(p['Xcr'],4)} m",
        f"Ycr = {f(t['kx * y (kN)'].sum(),3)} / {f(p['sum_kx'],3)} = {f(p['Ycr'],4)} m",
        f"Natural offset ex = Xcm - Xcr = {f(Xcm)} - {f(p['Xcr'])} = {f(p['ex_signed'])} m",
        f"Natural offset ey = Ycm - Ycr = {f(Ycm)} - {f(p['Ycr'])} = {f(p['ey_signed'])} m",
    ])
    sections.append(s)

    s = Section("04 | Wall lever arms and torsional rigidity, J", [
        "xbar,i = xi - Xcr; ybar,i = yi - Ycr.",
        "J = Sum(ky,i * xbar,i^2 + kx,i * ybar,i^2), in kN.m. Here J is torsional stiffness of the lateral system, not the geometric polar second moment of area of a slab.",
        "With rigid diaphragm rotation theta (rad), resisting torsional moment is Mz = J * theta. This idealizes each resisting wall as a linear in-plane spring.",
    ])
    t = wp[["Wall Name", "Direction", "xbar (m)", "ybar (m)", "ky*xbar^2 (kN·m)", "kx*ybar^2 (kN·m)"]].copy()
    t["J contribution (kN.m)"] = t["ky*xbar^2 (kN·m)"] + t["kx*ybar^2 (kN·m)"]
    s.tables.append(("Individual wall contributions to J", t))
    for _, r in wp.iterrows():
        ky, kx = float(r["ky (kN/m)"]), float(r["kx (kN/m)"])
        xb, yb = float(r["xbar (m)"]), float(r["ybar (m)"])
        s.calculations.append(f"{r['Wall Name']}: J_i = ({f(ky,2)} x {f(xb)}^2) + ({f(kx,2)} x {f(yb)}^2) = {f(ky*xb**2+kx*yb**2,3)} kN.m")
    s.calculations.append(f"Total J = {f(t['J contribution (kN.m)'].sum(),3)} kN.m")
    sections.append(s)

    s = Section("05 | Natural and accidental torsional moments", [
        "Evaluate both negative and positive accidental offsets, separately for each load direction.",
        "X load: e_real = Ycm - Ycr, e_acc = a * Ly; e_total = e_real +/- e_acc; Mz = -Fx * e_total.",
        "Y load: e_real = Xcm - Xcr, e_acc = a * Lx; e_total = e_real +/- e_acc; Mz = +Fy * e_total.",
        "Sign convention: positive Mz is counterclockwise when viewed from above. The sign of Mz under X load differs because +Fx at positive y causes a clockwise moment about CR.",
    ])
    cases_input = []
    for direction, force, dim, er in (("X", Fx, Ly, Ycm-p["Ycr"]),("Y", Fy, Lx, Xcm-p["Xcr"])):
        for sign in ("−", "+"):
            e_acc_signed = (-1 if sign == "−" else +1) * a * dim
            e_total = er+e_acc_signed
            moment = (-1 if direction == "X" else +1) * force*e_total
            cases_input.append([direction,sign,force,er,e_acc_signed,e_total,moment])
            s.calculations.append(f"{direction} load, {sign} accidental: e_total = {f(er)} {'-' if sign == '−' else '+'} {f(a)} x {f(dim)} = {f(e_total)} m; Mz = {'-' if direction=='X' else '+'}({f(force)}) x ({f(e_total)}) = {f(moment,3)} kN.m")
    s.tables.append(("All four simultaneous cases",pd.DataFrame(cases_input,columns=["Load","Sign","F (kN)","e_real (m)","e_acc signed (m)","e_total (m)","Mz (kN.m)"])))
    sections.append(s)

    s = Section("06 | Direct shear distribution", [
        "X load: Vx,direct,i = Fx * kx,i / Sum(kx). Direct Vy = 0.",
        "Y load: Vy,direct,i = Fy * ky,i / Sum(ky). Direct Vx = 0.",
        "These direct components distribute the translational load only; torsional contributions are added later.",
    ])
    t = wp[["Wall Name", "Direction"]].copy()
    t["X load, direct Vx (kN)"] = Fx*wp["kx (kN/m)"]/p["sum_kx"]
    t["Y load, direct Vy (kN)"] = Fy*wp["ky (kN/m)"]/p["sum_ky"]
    s.tables.append(("Direct shear by wall",t))
    for _, r in wp.iterrows():
        kx, ky = float(r["kx (kN/m)"]), float(r["ky (kN/m)"])
        s.calculations.append(f"{r['Wall Name']} | X: {f(Fx)} x {f(kx,2)} / {f(p['sum_kx'],2)} = {f(Fx*kx/p['sum_kx'])} kN; Y: {f(Fy)} x {f(ky,2)} / {f(p['sum_ky'],2)} = {f(Fy*ky/p['sum_ky'])} kN")
    sections.append(s)

    s = Section("07 | Torsional shear and combined wall forces", [
        "Using the signed Mz from Step 05: Vx,torsion,i = -Mz * kx,i * ybar,i / J; Vy,torsion,i = +Mz * ky,i * xbar,i / J.",
        "For EACH load case: Vx,diaphragm = Vx,direct + Vx,torsion; Vy,diaphragm = Vy,direct + Vy,torsion.",
        "For X-oriented walls, the wall-parallel diaphragm force is Vx. For Y-oriented walls, it is Vy. Add only the wall's local force parallel to its resisting axis (X local during X loading of X walls; Y local during Y loading of Y walls), as implemented by this solver.",
        "Unit shear is |V_wall-parallel| / wall length (kN/m). The cross-axis torsional contribution is reported on the perpendicular walls and is a real part of the simultaneous force state.",
    ])
    J = float(p["J"])
    for direction in ("X","Y"):
        case_frame = result["x_cases"] if direction == "X" else result["y_cases"]
        force = Fx if direction == "X" else Fy
        for sign in ("−","+"):
            case = case_frame[case_frame["Accidental Sign"] == sign].reset_index(drop=True)
            if len(case) != len(wp):
                raise ValueError("Calculated cases do not match wall properties. Please re-run the model.")
            mz = float(case.iloc[0]["Mz (kN·m)"])
            s.calculations.append(f"--- {direction} LOADING; accidental sign {sign}; F={f(force,3)} kN; Mz={f(mz,3)} kN.m ---")
            for i, r in wp.iterrows():
                c = case.iloc[i]
                if str(r["Wall Name"]) != str(c["Wall Name"]):
                    raise ValueError("Wall order does not match solver output.")
                name = str(r["Wall Name"])
                kx,ky=float(r["kx (kN/m)"]),float(r["ky (kN/m)"])
                xb,yb=float(r["xbar (m)"]),float(r["ybar (m)"])
                vxdir = float(c["Direct Vx (kN)"])
                vydir = float(c["Direct Vy (kN)"])
                vxt = float(c["Torsion Vx (kN)"])
                vyt = float(c["Torsion Vy (kN)"])
                lp = float(c["Local wall force (kN)"])
                vd = float(c["Diaphragm wall-parallel V (kN)"])
                vt = float(c["Wall-parallel V (kN)"])
                unit = float(c["v = |V|/Lwall (kN/m)"])
                s.calculations.extend([
                    f"{name} ({r['Direction']} wall): Vx,tor = -({f(mz,3)}) x {f(kx,2)} x ({f(yb)}) / {f(J,3)} = {f(vxt)} kN; Vy,tor = ({f(mz,3)}) x {f(ky,2)} x ({f(xb)}) / {f(J,3)} = {f(vyt)} kN",
                    f"{name}: Vx,diaphragm = {f(vxdir)} + {f(vxt)} = {f(float(c['Vx diaphragm (kN)']))} kN; Vy,diaphragm = {f(vydir)} + {f(vyt)} = {f(float(c['Vy diaphragm (kN)']))} kN",
                    f"{name}: Vparallel = Vdiaphragm,parallel + Vlocal = {f(vd)} + {f(lp)} = {f(vt)} kN; v = |{f(vt)}| / {f(float(r['Wall Length (m)']))} = {f(unit)} kN/m",
                ])
            t = case[["Wall Name", "Direction", "Direct Vx (kN)", "Direct Vy (kN)", "Torsion Vx (kN)", "Torsion Vy (kN)", "Wall-parallel V (kN)", "v = |V|/Lwall (kN/m)"]].copy()
            s.tables.append((f"{direction} load, accidental sign {sign}: simultaneous wall forces",t))
    sections.append(s)

    s = Section("08 | Force envelope and governing accidental sign", [
        "For each wall and EACH load direction, select the greater absolute signed wall-parallel V among + and - accidental-eccentricity cases.",
        "The X-load and Y-load envelopes are retained separately. Max directional envelope is a comparison only; it is NOT a vector combination of X and Y loading.",
        "Envelope values for different walls may come from different accidental signs; they are NOT a simultaneous load case. Use Step 07 for a simultaneous force state.",
    ])
    s.tables.append(("Governing forces from the existing solver",result["envelope"].copy()))
    for _, r in result["envelope"].iterrows():
        s.calculations.append(f"{r['Wall Name']}: X load envelope |V| = {f(r['X-load envelope |V| (kN)'])} kN (sign {r['X-load acc. sign']}); Y load envelope |V| = {f(r['Y-load envelope |V| (kN)'])} kN (sign {r['Y-load acc. sign']}); maximum directional = {f(r['Max directional envelope |V| (kN)'])} kN")
    sections.append(s)

    s = Section("09 | Force and torsional-moment equilibrium", [
        "The equilibrium check uses the diaphragm-distributed Vx and Vy, BEFORE adding local wall forces. Local wall forces are external to the global diaphragm load model and must not be counted as a second diaphragm force.",
        "Check Sum(Vx,diaphragm) = Fx (or 0 for a Y-only case); Sum(Vy,diaphragm) = Fy (or 0 for X-only); and Sum[xbar * Vy,diaphragm - ybar * Vx,diaphragm] = Mz.",
        "Residuals indicate numerical equilibrium only. They do not establish adequate wall strength, load path continuity, diaphragm flexibility, deflection limits, or code compliance.",
    ])
    s.tables.append(("Solver equilibrium checks (residuals should be ~ 0)",result["checks"].copy()))
    for _, r in result["checks"].iterrows():
        s.calculations.append(f"{r['Load Direction']} {r['Accidental Sign']}: Sum(Vx)={f(r['ΣVx (kN)'],5)}, target={f(r['Target Vx (kN)'],5)}, delta={f(r['ΔVx (kN)'],8)}; Sum(Vy)={f(r['ΣVy (kN)'],5)}, target={f(r['Target Vy (kN)'],5)}, delta={f(r['ΔVy (kN)'],8)}; Sum(M)={f(r['ΣM@CoR (kN·m)'],5)}, target={f(r['Target Mz (kN·m)'],5)}, delta={f(r['ΔM (kN·m)'],8)}")
    sections.append(s)

    s = Section("10 | Engineering interpretation and iteration workflow", [
        "1. Confirm the applicable NBCC/BCBC seismic provisions, diaphragm assumption, accidental eccentricity, torsional sensitivity, and analysis procedure; document selected code edition and project assumptions.",
        "2. Establish lateral loads (Fx and Fy) and the locations of the centre of mass. Confirm plan dimensions, wall locations and resisting directions.",
        "3. Obtain wall stiffnesses consistent with the expected cracked/connection/hold-down state and lateral demand. A simple k proportional to length model is preliminary, not a verified physical secant stiffness.",
        "4. Compute CR and J, apply natural and both signs of accidental torsion, combine direct and torsional force components, and examine four simultaneous load cases.",
        "5. Design wall strength and connections including overturning, collectors, chords, anchorage, compatibility and force-transfer path as applicable. Examine drift/deflection and flexible-diaphragm comparison where required.",
        "6. Update the wall design and physically meaningful secant/effective stiffness (e.g., k = |V|/deflection where appropriate), rerun force distribution, and iterate until changes are acceptably small. Record convergence criteria and check all relevant load combinations.",
        "7. This worksheet documents a single-level, in-plane rigid-diaphragm lateral force distribution for entered data. It is NOT by itself a full structural design or code compliance check, and it does not certify a proposed rigid-diaphragm assumption.",
    ])
    sections.append(s)
    return sections


def _display_table(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    for col in t.columns:
        if pd.api.types.is_numeric_dtype(t[col]):
            t[col] = t[col].map(lambda v: round(float(v), 5) if math.isfinite(float(v)) else v)
    return t


def _latex_formula_png(latex: str, *, dpi: int = 170) -> list[BytesIO]:
    """Render normalized Handcalcs expressions as PDF-ready PNGs.

    Complex KaTeX alignment environments are removed by the shared converter.
    This avoids bringing a TeX distribution into Streamlit Cloud.
    """
    from matplotlib.mathtext import math_to_image
    from engineering_calculations import latex_math_lines
    out = []
    for line in latex_math_lines(latex):
        # Matplotlib mathtext uses \mathrm for descriptive fragments.
        line = line.replace(r"\text{", r"\mathrm{")
        try:
            buffer = BytesIO()
            math_to_image("$" + line + "$", buffer, dpi=dpi,
                          format="png", color="#21354A")
            buffer.seek(0)
            out.append(buffer)
        except (ValueError, TypeError):
            # Keep the numeric solver trace; never substitute an invented value.
            continue
    return out


def render_calculation_tab(walls: pd.DataFrame, settings: dict, result: dict) -> None:
    """Existing ten-section worksheet, enhanced with Handcalcs + unit audit."""
    import streamlit as st
    from engineering_calculations import equation_by_sections, audit_units, latex_math_lines

    st.subheader("Step-by-step rigid diaphragm calculations")
    st.caption("Live results from the existing validated solver. Rendered equations provide a second, scalar calculation trace; ForAllPeople checks the dimensions separately.")
    sections = build_calculation_sections(walls, settings, result)

    try:
        rendered = equation_by_sections(result, settings)
        for number, equations in rendered.items():
            sections[number-1].equations.extend(equations)
        st.success(f"Handcalcs verified {sum(len(x) for x in rendered.values())} rendered arithmetic steps against the active model.")
    except ImportError as exc:
        st.warning(f"Handcalcs unavailable ({exc}). Install the packages from requirements.txt. The original worksheet is still available.")
    except Exception as exc:
        st.error(f"Equation verification failed: {exc}. The existing solver and plain-text worksheet remain available; check this discrepancy before design use.")

    try:
        audit = audit_units(result, settings)
        passed = bool(audit["Pass"].all())
        if passed:
            st.success(f"ForAllPeople dimensional and magnitude audit: {len(audit)} / {len(audit)} checks passed.")
        else:
            st.error(f"ForAllPeople audit: {int(audit['Pass'].sum())}/{len(audit)} passed. Investigate before design use.")
        with st.expander("Unit audit details (ForAllPeople)", expanded=not passed):
            st.dataframe(_display_table(audit), hide_index=True, use_container_width=True)
            st.download_button("Download unit-audit CSV", audit.to_csv(index=False).encode("utf-8"),
                               file_name="rigid_diaphragm_unit_audit.csv", mime="text/csv", key="units_audit_csv")
    except ImportError as exc:
        st.warning(f"ForAllPeople unavailable ({exc}). Unit checks are not active until the dependencies are installed.")
    except Exception as exc:
        st.error(f"Unit audit could not finish: {exc}. Do not regard this model as independently dimension-verified.")

    # A cache avoids repeatedly converting mathematical equations to PDF on
    # every unrelated Streamlit slider change that leaves this model unchanged.
    try:
        import hashlib
        import pickle
        @st.cache_data(show_spinner=False, max_entries=6)
        def _cached_pdf(_detailed_sections: list[Section], fingerprint: str) -> bytes:
            return make_calculation_pdf(_detailed_sections)
        signature = hashlib.sha256(pickle.dumps(sections, protocol=4)).hexdigest()
        pdf = _cached_pdf(sections, signature)
    except ImportError as exc:
        st.warning(f"PDF export requires ReportLab and Matplotlib: {exc}")
    except Exception as exc:
        st.error(f"PDF generation error: {exc}")
    else:
        st.download_button("Download complete calculations (PDF) — ready to print", pdf,
                           file_name="rigid_diaphragm_handcalcs_calculations.pdf",
                           mime="application/pdf", use_container_width=True,
                           key="rigid_calc_pdf_download")
    st.info("Read Steps 01–10 in order. X/Y loading and both accidental eccentricity signs are distinct cases. An envelope is NOT one simultaneous force state.")
    for sec in sections:
        st.markdown("### " + sec.heading)
        for note in sec.notes:
            st.markdown(note)
        for label, table in sec.tables:
            st.markdown("**" + label + "**")
            st.dataframe(_display_table(table), hide_index=True, use_container_width=True)
        if sec.equations:
            st.markdown("**Handcalcs — symbolic equations and numerical substitutions**")
            for caption, latex in sec.equations:
                st.caption(caption)
                # A single KaTeX expression is much more portable than handing
                # Streamlit Handcalcs' entire aligned LaTeX environment.
                for math_line in latex_math_lines(latex):
                    st.latex(math_line)
        if sec.calculations:
            with st.expander("Full numerical working and solver trace (plain text)", expanded=not bool(sec.equations)):
                for line in sec.calculations:
                    if line.startswith("--- "):
                        st.markdown("**" + line.strip("- ") + "**")
                    else:
                        st.code(line, language=None)
        st.divider()


def make_calculation_pdf(sections: list[Section], title: str="Rigid Diaphragm - Detailed Calculation Sheet") -> bytes:
    """Return printable multi-page PDF bytes including ALL content shown on screen."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle, KeepTogether, HRFlowable

    # Use installed font if present for symbols/engineering units. Font file is never embedded as a separate artifact.
    font = "Helvetica"
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if regular.exists():
        try:
            if "RD-DejaVu" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont("RD-DejaVu",str(regular)))
            font = "RD-DejaVu"
        except Exception:
            pass
    mem = BytesIO()
    doc = SimpleDocTemplate(mem,pagesize=A4,rightMargin=14*mm,leftMargin=14*mm,topMargin=16*mm,bottomMargin=17*mm,
        title=title, author="Rigid Diaphragm Research Lab")
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="RdTitle",fontName=font,fontSize=15,leading=20,textColor=colors.HexColor("#172633"),spaceAfter=9))
    styles.add(ParagraphStyle(name="RdHeading",fontName=font,fontSize=11,leading=15,textColor=colors.HexColor("#253c52"),spaceBefore=11,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name="RdBody",fontName=font,fontSize=8.1,leading=12.5,spaceAfter=5,alignment=TA_LEFT))
    styles.add(ParagraphStyle(name="RdCalc",fontName=font,fontSize=7.15,leading=11.5,spaceAfter=3,leftIndent=8,wordWrap="CJK"))
    styles.add(ParagraphStyle(name="RdSmall",fontName=font,fontSize=6.5,leading=9.5,wordWrap="CJK"))
    styles.add(ParagraphStyle(name="RdSub",fontName=font,fontSize=8.1,leading=12,spaceBefore=5,spaceAfter=4,keepWithNext=True))
    story=[Paragraph(escape(title),styles["RdTitle"]),
           Paragraph("Live model inputs and solver outputs | Both loading axes and both accidental-eccentricity signs | Units as labelled",styles["RdBody"]),
           HRFlowable(width="100%",thickness=1,color=colors.HexColor("#bfcad4")),Spacer(1,6)]
    usable= A4[0]-28*mm
    for section in sections:
        story.append(Paragraph(escape(section.heading),styles["RdHeading"]))
        for note in section.notes:
            story.append(Paragraph(escape(note),styles["RdBody"]))
        for label, table in section.tables:
            story.append(Paragraph(escape(label),styles["RdSub"]))
            table = table.copy().reset_index(drop=True)
            head=list(table.columns)
            rows=[]
            for row in table.itertuples(index=False,name=None):
                rows.append([f(v,4) if isinstance(v,(int,float)) and not isinstance(v,bool) else str(v) for v in row])
            n=max(1,len(head))
            col_width=usable/n
            # Long technical headings need wrapping; plenty of rows may need to spill onto next page.
            cell=lambda val:Paragraph(escape(str(val)),styles["RdSmall"])
            data=[[cell(h) for h in head]]+[[cell(v) for v in row] for row in rows]
            tb=LongTable(data,colWidths=[col_width]*n,repeatRows=1,hAlign="LEFT")
            tb.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#edf2f7")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#fafbfd")]),
                ("LINEBELOW",(0,0),(-1,0),0.6,colors.HexColor("#cbd5e1")),
                ("VALIGN",(0,0),(-1,-1),"TOP"),
                ("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4),
                ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ]))
            story.append(tb);story.append(Spacer(1,6))
        if section.equations:
            story.append(Paragraph("Rendered engineering equations (Handcalcs)",styles["RdSub"]))
            from reportlab.platypus import Image
            from PIL import Image as PILImage
            for caption, latex in section.equations:
                story.append(Paragraph(escape(caption),styles["RdSmall"]))
                rendered_lines = _latex_formula_png(latex)
                if not rendered_lines:
                    story.append(Paragraph("Equation image not supported by PDF math renderer; use numeric trace below.", styles["RdSmall"]))
                for formula in rendered_lines:
                    with PILImage.open(formula) as pic:
                        image_width, image_height = pic.size
                    formula.seek(0)
                    # Keep every equation within printed margins and keep text legible.
                    scale = min(0.68, usable / max(1, image_width), 70.0 / max(1, image_height))
                    display_width = image_width * scale
                    display_height = image_height * scale
                    story.append(Image(formula, width=display_width, height=display_height, hAlign="LEFT"))
                story.append(Spacer(1,3))
        if section.calculations:
            story.append(Paragraph("Full numeric substitutions and solver working",styles["RdSub"]))
            for line in section.calculations:
                if line.startswith("--- "):
                    story.append(Paragraph(escape(line.strip("- ")),styles["RdSub"]))
                else:
                    story.append(Paragraph(escape(line),styles["RdCalc"]))
        story.append(Spacer(1,5))
    def footer(canvas,doc):
        canvas.saveState(); canvas.setFont(font,7); canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(14*mm,10*mm,"Generated from Rigid Diaphragm Research Lab - verify inputs and assumptions")
        canvas.drawRightString(A4[0]-14*mm,10*mm,f"Page {doc.page}")
        canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return mem.getvalue()
