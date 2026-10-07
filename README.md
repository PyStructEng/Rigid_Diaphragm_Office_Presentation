RIGID DIAPHRAGM RESEARCH LAB — WOOD PARAMETRIC ATLAS V7
========================================================

FILES
-----
1. rigid_diaphragm_parametric_app.py   Updated Streamlit app
2. rigid_diaphragm_core.py             Existing rigid-diaphragm calculation engine
3. wood_shearwall_mechanics.py         Existing mechanics-based wood-wall engine
4. wood_parametric_study.py            NEW batch parametric atlas engine
5. requirements.txt                    Python dependencies

WHAT WAS ADDED
--------------
Batch suite now contains two subtabs:
- Core diaphragm batch
- Wood shearwall parametric atlas

The Wood shearwall parametric atlas uses the current one-storey Wood Wall Lab row
as the baseline/fixed design and sweeps:
- Panel type
- Panel thickness (only type/thickness pairs present in the embedded Bv table)
- Panel sides (S.S / B.S)
- Nail spacing
- Wall height H
- Aspect ratio H/L (wall length is calculated as H/(H/L))
- Demand

Demand can be defined as:
- Baseline force multiplier
- Constant unit shear (kN/m)
- Constant total force (kN)

OUTPUTS STORED FOR EVERY CASE
-----------------------------
- Full input/configuration traceability
- Δ bending
- Δ panel shear
- Δ nail slip
- Δ anchorage
- Δ total
- Component fractions
- Secant stiffness k
- k/L
- Reference stiffness at the same H, L and V
- k/reference same condition
- Global baseline stiffness and k/global baseline
- Rod utilization flag
- Analysis status / error text

IMPORTANT ENGINEERING BOUNDARY
------------------------------
The atlas is a mechanics/stiffness study. Panel type/thickness is screened against
the embedded Bv database, but selected nail spacing is NOT checked against a CSA
approved shearwall resistance table and the atlas does not establish factored wall
resistance. Verify permitted wall construction and strength separately before using
a configuration on a project.

RECOMMENDED FIRST OFFICE-REFERENCE RUN
--------------------------------------
Panel types: all available
Panel thicknesses: 9.5, 12.5, 15.5, 18.5 mm (expand later)
Panel sides: S.S, B.S
Nail spacing research levels: 75, 150, 300 mm
Wall heights: 2.4, 3.0, 3.6 m
Aspect ratios H/L: 0.5, 0.75, 1.0, 1.5, 2.0
Demand basis: Baseline force multiplier
Demand levels: 0.25, 0.5, 0.75, 1.0, 1.5

This produces about 4,050 cases with the current embedded panel database and is a
good first dataset before expanding to additional thicknesses/spacings.

HOW TO COLLECT DATA FOR CHATGPT
-------------------------------
1. In Wood Wall Lab, set Number of stacked storeys = 1.
2. Enter the baseline wall construction and the framing/rod/anchorage assumptions
   you want held constant.
3. Go to Batch suite > Wood shearwall parametric atlas.
4. Select the sweep variables and click Run wood shearwall parametric atlas.
5. Download: wood_shearwall_parametric_atlas.xlsx
6. Upload that workbook to ChatGPT.
7. Ask:
   "Analyze this wood shearwall parametric study for my rigid-diaphragm presentation.
   Generate the key graphs, sensitivity ranking, engineering comments, limitations,
   and office-reference conclusions. Focus on what controls stiffness, demand-
   dependence, diminishing returns, deformation components, and whether k is
   proportional to wall length."

The workbook contains:
- README
- Study_Metadata
- Baseline_Design
- Assembly_Catalog
- Study_Summary
- Data_Dictionary
- Parametric_Data

SECOND-STAGE STUDY
------------------
After the wall atlas is understood, use the mechanics/coupled diaphragm tools to
connect selected wall stiffness differences to:
- CR movement
- natural/accidental torsion
- direct shear redistribution
- individual wall force changes

That second stage is where the office reference can answer when a simple relative-
stiffness assumption is adequate and when mechanics-based stiffness materially
changes the rigid-diaphragm result.
