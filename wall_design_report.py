"""Printable wall schedule; evaluation results are expressly trial snapshots."""
from __future__ import annotations
from io import BytesIO
import math
from xml.sax.saxutils import escape
import pandas as pd

def make_wall_design_pdf(assemblies:pd.DataFrame, schedule:pd.DataFrame, results:pd.DataFrame, source_label:str)->bytes:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, KeepTogether
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import landscape, A4
    b=BytesIO()
    s=getSampleStyleSheet()
    s.add(ParagraphStyle(name='HdrWD',parent=s['Heading2'],fontSize=10,spaceBefore=12,spaceAfter=6,textColor=colors.HexColor('#1E405A')))
    s.add(ParagraphStyle(name='SmallWD',parent=s['Normal'],fontSize=8.1,leading=11,spaceAfter=6))
    s.add(ParagraphStyle(name='CellWD',parent=s['Normal'],fontSize=7,leading=9))
    doc=SimpleDocTemplate(b,pagesize=landscape(A4),leftMargin=35,rightMargin=35,topMargin=37,bottomMargin=40,title='Wall assembly design research schedule')
    flow=[Paragraph('Wood shearwall design studio — trial mechanics schedule',s['Title']),
          Paragraph('Research/development calculation aid. Not a CSA O86 design certification or a globally compatible stacked-wall analysis.',s['SmallWD']),
          Paragraph('Demand source: '+escape(source_label),s['SmallWD']),
          Paragraph('Trial secant stiffness is derived from a single-storey isolated wall with the stated shear demand. The nonlinear/global model has NOT been iterated with these computed values.',s['SmallWD'])]
    def fmt(x):
        if isinstance(x,(int,float)):
            return f'{x:.3f}' if math.isfinite(x) else '—'
        return str(x)
    def table(df, cols, widths=None):
        data=[[Paragraph(escape(str(k)),s['CellWD']) for k in cols]]
        for _,r in df.iterrows():
            data.append([Paragraph(escape(fmt(r.get(k,''))),s['CellWD']) for k in cols])
        t=Table(data,repeatRows=1,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F6')),
                               ('GRID',(0,0),(-1,-1),.25,colors.HexColor('#CCD6E0')),
                               ('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#FAFCFF')]),
                               ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
        flow.append(t)
    flow.append(Paragraph('Individual wall trial results',s['HdrWD']))
    table(results,["Storey","Wall Name","Assembly ID","Trial |V| (kN)","Trial deflection (mm)","Trial k secant (kN/m)","Trial drift (%)","Drift review flag","Shear status","Hold-down tension status"],
          [32,54,70,66,82,94,52,140,86,82])
    flow.append(Paragraph('Wall schedule — design inputs',s['HdrWD']))
    table(schedule,["Storey","Wall Name","Assembly ID","Wall length (m)","Trial shear (kN)","Factored shear demand (kN)","Verified shear resistance (kN)","Factored tension demand (kN)","Verified tension resistance (kN)"],
          [37,63,79,74,74,102,113,107,113])
    flow.append(Paragraph('Assembly specifications',s['HdrWD']))
    for _,r in assemblies.iterrows():
        val=(f"<b>{escape(str(r['Assembly ID']))}</b>: {escape(str(r['Panel Type']))} {r['Panel thickness (mm)']} mm, "
             f"{escape(str(r['Panel sides']))}, nails {r['Nail diameter (mm)']} mm x {r['Nail length (mm)']} mm; "
             f"edge {r['Nail edge spacing (mm)']} mm / field {r['Nail field spacing (mm)']} mm; "
             f"{escape(str(r['Species']))} {escape(str(r['Grade']))}, {escape(str(r['Stud size']))}, "
             f"{r['Chord studs / end']} chord studs per end, rod {escape(str(r['Rod model']))}, "
             f"take-up {escape(str(r['Take-up device']))}.")
        flow.append(Paragraph(val,s['SmallWD']))
    flow.append(Paragraph('Engineering boundaries and outstanding verification',s['HdrWD']))
    for item in [
      'The 2% trial-drift flag is a research screening heuristic, NOT a BCBC drift acceptance criterion or model calibration limit.',
      'The panel rigidity and nail-slip equations are inherited from the existing historical CWC/FPInnovations-inspired research engine; they are not a general nonlinear connection calibration.',
      'Nail field spacing, nail length and assembly notes are recorded but do NOT change stiffness in this model; nail edge spacing does.',
      'Shear and hold-down strengths are NOT calculated by this tool. Strength status is displayed only when the engineer enters separate factored demands AND independently verified resistances.',
      'Zero shears intentionally produce undefined secant stiffness; negative shears use magnitude only. Signed anchorage engagement and load reversal are not represented.',
      'Preload, take-up seating and simplified chord/bearing force assumptions can materially affect deformation. Confirm reference shear, service gravity and boundary conditions.',
      'The current whole-building model remains a linear storey-spring approximation. Any one-way transferred stiffness is a trial what-if study, not a compatible multistorey nonlinear equilibrium solution.',
      'Check code edition, CSA O86 design capacities, connection detailing, structural irregularity, drift, diaphragm flexibility, load path and all applicable load combinations independently.',
    ]:
        flow.append(Paragraph('• '+escape(item),s['SmallWD']))
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setFont('Helvetica',8)
        canvas.setFillColor(colors.HexColor('#708090'))
        canvas.drawString(35,23,'RESEARCH / NOT FOR DESIGN WITHOUT INDEPENDENT VERIFICATION')
        canvas.drawRightString(807,23,f'Page {doc.page}')
        canvas.restoreState()
    doc.build(flow,onFirstPage=footer,onLaterPages=footer)
    return b.getvalue()
