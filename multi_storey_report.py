"""Printable, self-contained multi-storey model report using ReportLab."""
from __future__ import annotations
from io import BytesIO
from xml.sax.saxutils import escape
import math
import pandas as pd


def make_multistorey_pdf(result: dict) -> bytes:
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, KeepTogether, PageBreak)
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.units import inch
    buffer=BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=(612,792),rightMargin=39,leftMargin=39,topMargin=50,bottomMargin=48,title="Multi-storey rigid diaphragm research calculations")
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LabTitle",parent=styles['Title'],fontName='Helvetica-Bold',fontSize=17,leading=20,textColor=colors.HexColor('#18293d'),spaceAfter=9))
    styles.add(ParagraphStyle(name="LabHead",parent=styles['Heading2'],fontName='Helvetica-Bold',fontSize=11,leading=14,textColor=colors.HexColor('#1a3856'),spaceBefore=12,spaceAfter=6))
    styles.add(ParagraphStyle(name="LabText",parent=styles['BodyText'],fontSize=8.3,leading=12,spaceAfter=6))
    styles.add(ParagraphStyle(name="LabTiny",parent=styles['BodyText'],fontSize=6.9,leading=9))
    p=lambda s:Paragraph(escape(str(s)),styles['LabText'])
    flow=[Paragraph("Multi-storey rigid diaphragm - calculation record",styles['LabTitle']),
       p("Educational storey-shear-spring model with three degrees of freedom (Ux, Uy, theta) per floor. All quantities reflect the current exported inputs and the selected model assumptions."),
       p("ENGINEERING LIMITATION: Not a complete seismic code analysis or a mechanics-based continuous-wall model. No design-period cap, vertical irregularity, drift amplification, P-delta, strength design, or building-code acceptance is performed."),
       ]
    s=result['settings']; f=result['input_floors'];w=result['input_walls']
    def heading(title): flow.append(Paragraph(title,styles['LabHead']))
    def table(data,widths=None):
        def cell(v): return Paragraph(escape(str(v)), styles['LabTiny'])
        content=[[cell(x) for x in row] for row in data]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign='LEFT',splitByRow=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef5')),
            ('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#203c56')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fc')]),
            ('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor('#6a8093')),
            ('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
            ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
            ('VALIGN',(0,0),(-1,-1),'TOP')]))
        flow.append(t)
    heading("1. Input assumptions")
    flow.append(p(f"Storeys: {len(f)} | Plan: {s['Lx']:.3f} x {s['Ly']:.3f} m | Global reference: ({s['Lx']/2:.3f}, {s['Ly']/2:.3f}) m | Accidental fraction: {100*s['acc']:.1f}% per side."))
    flow.append(p("Seismic weight is treated as uniform mass over a rectangular rigid floor. Entered floor forces are supplied by the user, not produced by an NBCC equivalent-static or dynamic spectrum procedure."))
    table([["Floor","h (m)","W (kN)","Fx (kN)","Fy (kN)","CM x,y (m)"]] +
       [[int(r['Storey']),f"{r['Height (m)']:.2f}",f"{r['Weight (kN)']:.1f}",f"{r['Fx (kN)']:.1f}",f"{r['Fy (kN)']:.1f}",f"{r['Xcm (m)']:.2f}, {r['Ycm (m)']:.2f}"] for _,r in f.iterrows()],
       [37,54,80,80,80,203])
    heading("2. Governing model equations")
    for eq in [
       "At storey i, q_i = [Ux_i, Uy_i, theta_i]. Relative storey deformation = q_i - q_(i-1); q_0 = 0.",
       "For an X-oriented wall: a_j = [1, 0, -(y_j-y_ref)]. For Y: a_j = [0, 1, x_j-x_ref].",
       "Wall shear V_j = k_j * a_j^T * (q_i - q_(i-1)).",
       "Storey stiffness S_i = sum( k_j * a_j * a_j^T ), assembled into global K as K_ii += S_i, K_(i-1,i-1) += S_i, and K_(i,i-1) -= S_i.",
       "Static equilibrium K q = F, with floor torsion from each load's horizontal eccentricity including both signs of accidental eccentricity.",
       "Uniform rectangular diaphragm mass M_i = m_i*(ax*ax^T + ay*ay^T) + diag(0,0,m_i*(Lx^2+Ly^2)/12), m_i=W_i/g.",
       "Modal periods solve K phi = omega^2 M phi, T=2*pi/omega.",
       "Rayleigh check T_R=2*pi*sqrt[(q^T M q)/(q^T K q)] using that specific static load case's deformed shape.",
       "Maximum elastic storey drift: check Ux and Uy differences at ALL FOUR plan corners and divide by the storey height.",
    ]:flow.append(p(eq))
    heading("2A. Numerical substitution - one wall, X+ load case")
    first=w.iloc[0]
    idx=int(first['Storey'])-1
    q=result['case_results']['X +']['q']
    dq=q[idx]-(q[idx-1] if idx else 0.0)
    x0=float(s['Lx'])/2.;y0=float(s['Ly'])/2.
    arm=-(float(first['y (m)'])-y0) if first['Direction']=='X' else float(first['x (m)'])-x0
    drift=float(dq[0] if first['Direction']=='X' else dq[1]) + float(dq[2])*arm
    local_k=float(result['k'][0])
    first_force=result['wall_forces'].query('Case == "X +"')
    first_force=first_force[(first_force['Storey']==first['Storey'])&(first_force['Wall Name']==first['Wall Name'])]
    flow.append(p(f"Wall {first['Wall Name']}, Storey {idx+1}, {first['Direction']} direction, lever coefficient a_theta = {arm:.3f} m; k = {local_k:,.3f} kN/m."))
    flow.append(p(f"Floor relative DOF: delta Ux = {dq[0]:.7f} m, delta Uy = {dq[1]:.7f} m, delta theta = {dq[2]:.8f} rad."))
    flow.append(p(f"Along-wall relative displacement = {drift:.7f} m. Wall shear = ({local_k:.3f} kN/m) x ({drift:.7f} m) = {local_k*drift:.3f} kN."))
    flow.append(p(f"Calculated force in solved case = {float(first_force.iloc[0]['Wall shear (kN)']):.3f} kN. This substitution matches the global solution; the other wall forces are tabulated in Section 9."))
    Kfirst=result['storey_K'][0]
    flow.append(p(f"First-storey local matrix sample: S[Ux,Ux]={Kfirst[0,0]:,.3f} kN/m; S[Uy,Uy]={Kfirst[1,1]:,.3f} kN/m; S[theta,theta]={Kfirst[2,2]:,.3f} kN-m (per radian)."))
    heading("3. Summary of analysis")
    table([["Check","Calculated"]]+[
       ["First modal period (s)",f"{result['first_mode_s']:.4f}"],
       ["X-dominant period (s)",f"{result['modal_x_s']:.4f}"],
       ["Y-dominant period (s)",f"{result['modal_y_s']:.4f}"],
       ["Maximum unamplified elastic drift (%)",f"{result['max_drift_pct']:.4f}"],
       ["Maximum abs static residual (kN or kN-m)",f"{result['max_residual']:.3g}"],
       ["Demand-dependent parametric iteration", "ON" if result['nonlinear'] else "OFF"],
       ["Iteration converged", "YES" if result['converged'] else "NO - do not use final iterative results as converged"],
    ],[335,199])
    heading("4. Modes and participating mass")
    table([["Mode","T(s)","Freq(Hz)","X mass %","Y mass %"]]+[
        [int(r['Mode']),f"{r['Period (s)']:.4f}",f"{r['Frequency (Hz)']:.3f}",f"{r['X effective mass (%)']:.2f}",f"{r['Y effective mass (%)']:.2f}"]
        for _,r in result['modes'].head(min(9,len(result['modes']))).iterrows()],[48,85,95,153,153])
    heading("5. Force-case-specific Rayleigh estimates")
    table([["Load case","Rayleigh T (s)","max residual (kN or kN-m)"]] + [
        [name,f"{cs['rayleigh_s']:.5f}" if math.isfinite(cs['rayleigh_s']) else "n/a",f"{cs['residual']:.2e}"]
        for name,cs in result['case_results'].items()],[90,135,309])
    heading("6. Reference-height diagnostic (NOT a code period)")
    flow.append(p("Interpolation at z=2H/3; V_D is the total entered lateral load (base shear) over ALL floors, W is the total building weight. k_ref=V_ref/|u_ref|; T=2*pi*sqrt(W_eff/(g*k_ref)). This reproduces the arithmetic FORMAT of the APEGBC 2015 alternative estimate, but is NOT an accepted seismic design period for the current project."))
    table([["Dir","2H/3 (m)","u (mm)","V_D(kN)","k(kN/m)","T diag (s)"]]+[
      [r["Direction"],f"{r['Reference height (m)']:.2f}",f"{r['Translation at reference (mm)']:.2f}",f"{r['Total applied V_D (kN)']:.2f}",f"{r['Equivalent k (kN/m)']:.1f}",f"{r['Illustrative T_2/3 (s)']:.3f}"] for _,r in result['approx'].iterrows()],[43,94,91,100,100,106])
    if result['nonlinear']:
        heading("7. Uncalibrated stiffness sensitivity, fixed lateral load vectors")
        flow.append(p(f"k_target = max({s['min_stiffness_ratio']:.2f} k_initial, k_initial * max(|V_envelope|/{s['reference_force']:.2f} kN, 1) ^(1-{s['exponent']:.3f})). Relaxation = {s['relaxation']:.2f}, relative convergence tolerance = {100*s['tolerance']:.2f}%. This is a parametric stiffness law, NOT mechanics-based nail/anchorage behaviour, and does NOT recalculate seismic actions from the changing period."))
        table([["Iteration","T1 (s)","Max drift %","k delta %","Period delta %"]]+[
          [int(r['Iteration']),f"{r['T1 (s)']:.4f}",f"{r['Max drift (%)']:.4f}",f"{r['k change (%)']:.3f}",f"{r['Period change (%)']:.3f}"]
          for _,r in result['history'].iterrows()],[80,112,112,115,115])
    heading("8. Floor response by distinct load case")
    for case in ("X +","X -","Y +","Y -"):
        flow.append(p(f"Case {case} (DO NOT combine with another envelope row as a simultaneous state)"))
        subset=result['floors'].query("Case == @case")
        table([["Floor","Ux (mm)","Uy (mm)","theta (mrad)","Drift max %"]]+[
          [int(r['Storey']),f"{r['Ux (mm)']:.2f}",f"{r['Uy (mm)']:.2f}",f"{r['Rotation (mrad)']:.3f}",f"{r['Max drift (%)']:.3f}"] for _,r in subset.iterrows()],[70,119,119,119,107])
    heading("9. Wall-shear ENVELOPE (NOT one simultaneous state)")
    table([["Floor","Wall","Dir","Governs","abs V(kN)","k used(kN/m)"]]+[
       [int(r['Storey']),r['Wall Name'],r['Direction'],r['Governing case'],f"{r['Envelope |V| (kN)']:.2f}",f"{r['k used (kN/m)']:.1f}"]
       for _,r in result['envelope'].iterrows()],[48,109,48,78,112,139])
    heading("10. Design limitations and next steps")
    for txt in [
        "Input floors must have physical, project-specific seismic weights and floor force vectors. The illustrated default force distribution is NOT a building-code distribution.",
        "Rigid floor kinematics imply compatibility. Individual walls can have different demands and drifts because of torsion; displacements are not artificially equalized at all wall locations.",
        "Continuous wood shearwall flexure, lower-storey hold-down rotation, coupling, openings, diaphragm deformation, collector/chord demands, strength, redundancy, and vertical irregularities require separate models/checks.",
        "For code seismic demand, determine the governing permitted period and force distribution using the applicable NBCC/BCBC edition; evaluate required response-spectrum / dynamic analysis and drift amplification separately.",
        "Compare with independent multi-storey software benchmarks before engineering use. These results remain research/prototype output.",
    ]:flow.append(p(txt))
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#607080'))
        canvas.drawString(39,27,'RIGID DIAPHRAGM RESEARCH LAB  |  EDUCATIONAL MODEL - NOT A DESIGN CERTIFICATION')
        canvas.drawRightString(573,27,f'Page {doc.page}')
    doc.build(flow,onFirstPage=footer,onLaterPages=footer)
    return buffer.getvalue()
