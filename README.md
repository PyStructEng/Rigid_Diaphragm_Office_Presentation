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


RIGID DIAPHRAGM RESEARCH LAB - NEW STEP-BY-STEP CALCULATIONS TAB
================================================================

WHAT IS NEW
- One new 'Step-by-step calculations' tab at the end of the existing app.
- Ten calculation sections, formulas and substitutions for every wall.
- X and Y loading, plus/minus accidental eccentricity, CR, J, direct/torsional forces,
  local wall forces, wall envelopes, equilibrium checks, and next-step design workflow.
- A PDF download button, showing the CURRENT app inputs and results and suitable for printing.

DEPLOYMENT TO STREAMLIT COMMUNITY CLOUD / GITHUB
1. Back up your existing working GitHub repository first.
2. Extract this ZIP file. Copy the following files together into your repository's app folder:
   rigid_diaphragm_core.py
   wood_shearwall_mechanics.py
   diaphragm_calculation_sheet.py          [NEW]
   requirements.txt                        [ensure reportlab>=4.0]
3. For the main Mechanics V6 application (the most complete 1,900-line source), replace
   rigid_diaphragm_parametric_app.py with the copy here.
   If your deployment instead uses 'rigid_diaphragm_parametric_app_GLOBAL_VISUAL_V2.py',
   replace that file with the copy here; both are updated.
   You only need to choose the entrypoint that your Streamlit deployment currently uses.
4. Commit / push your changes and wait for Streamlit to redeploy.
5. Open the new 'Step-by-step calculations' tab. Use its PDF button.

LOCAL RUN (with Python 3.10+)
  pip install -r requirements.txt
  streamlit run rigid_diaphragm_parametric_app.py

OPTIONAL TESTS
  python test_calculation_sheet.py

CAUTIONS
- Calculation sheet reports existing solver outputs; it is NOT a second analysis engine.
- For each X/Y case, the reported torsional equilibrium uses diaphragm forces before
  local wall forces; this mirrors the solver's assumptions.
- Your model's specified accidental eccentricity ratio is not automatically selected from code.
- Current result is single-level in-plane rigid diaphragm force distribution. Confirm
  all relevant code/load provisions and separate strength, stiffness, load-path and
  drift design checks before applying on a real project.

The included sample PDF is generated using the application's FPInnovations preset.



RIGID DIAPHRAGM RESEARCH LAB — HANDCALCS + FORALLPEOPLE INTEGRATION
=================================================================

BASED ON
The full Mechanics V6 app with the 10-section calculations tab and PDF export
prepared on October 8, 2026. Its structural solver has not been rewritten.

WHAT CHANGED
1. Step-by-step calculations tab: Handcalcs renders real Python scalar
   expressions with symbolic variables, numerical substitutions and results.
   It includes direction-specific stiffness share, centres of rigidity,
   eccentricity, individual wall J, torsional moments, direct and torsional
   wall forces, final wall-parallel force and unit shear.
2. Every Handcalcs scalar result is checked numerically against the output of
   the existing analysis engine. Discrepancies show an error in Streamlit.
3. ForAllPeople checks physical dimensions and numerical unit conversion for
   combined directional stiffnesses, Xcr, Ycr, J, four load-case moments,
   each wall's direct and torsional shear, final force and unit shear.
   A complete downloadable unit-audit CSV is included in the UI.
4. The PDF includes both an equation section and the original ALL-wall numeric
   working and tables. The math is rasterized with Matplotlib for portable PDF
   generation (no LaTeX / TeX system is required by Streamlit Cloud).
5. Wood-wall laboratory: an optional display verifies and visualizes the
   service/seismic secant stiffness k = |V| / delta using both libraries.
   The stacked mechanics and coupled iteration solver equations are unchanged.
6. All previous visualization, parameter studies, mechanics, coupled-iteration,
   benchmark, Excel/CSV exports and both app entrypoint versions are preserved.
7. Requirements updated to include Handcalcs 1.11.0 and ForAllPeople 3.0.0.

INSTALLATION FOR GITHUB + STREAMLIT CLOUD
1. Download this ZIP and unzip it.
2. Save a backup/commit of your existing working repository before copying.
3. Copy all Python files and requirements.txt into the SAME directory in the
   repository. Do not change your existing Streamlit Cloud entrypoint name.
4. If you run the full Mechanics V6 app, use:
      rigid_diaphragm_parametric_app.py
   If your deployment is the lighter alternative, use:
      rigid_diaphragm_parametric_app_GLOBAL_VISUAL_V2.py
   Both have the same new calculation renderer; choose ONE app entrypoint.
5. Commit and push changes to GitHub. Streamlit Cloud should install the
   packages from the new requirements.txt and redeploy.
6. Open Step-by-step calculations; verify Handcalcs and ForAllPeople pass.
   Click the PDF download button and print the exported report.

RUN LOCALLY (Python 3.10+)
   python -m pip install -r requirements.txt
   python test_calculation_sheet.py
   python test_upgrade_live.py
   streamlit run rigid_diaphragm_parametric_app.py

TESTING AND LIMITATIONS
- Pre-existing baseline regression tests pass: 6/6 structural solver tests,
  5/5 wood mechanics tests; also checked FPInnovations, symmetric,
  asymmetric/local-force, signed-negative-load and zero-load models.
- Offline mocked-dependency tests verified report arithmetic, units and PDF
  pathways. Run test_upgrade_live.py after installation to test the ACTUAL
  package versions. Due to restricted network access in the build environment,
  live third-party-package compatibility was not executed during packaging.
- Handcalcs is for scalar presentation, not a matrix solver. ForAllPeople is
  used at validation boundaries, NOT inside NumPy/matrix computation.
- The report and unit audit do NOT verify code-prescribed accidental
  eccentricity, model adequacy, CSA design strength, diaphragm rigidity,
  hold-down loads or connection adequacy. User must verify assumptions.
- Treat the PDF as an engineering calculation AID, not a sealed design report.
- On unusually large models, generating every wall's math in the report can
  take additional seconds. The PDF is cached for unchanged model inputs.

NEW FILES
   engineering_calculations.py     - unit audits, Handcalcs functions,
                                     wood stiffness trace
   test_upgrade_live.py           - genuine library end-to-end test

UPDATED FILES
   diaphragm_calculation_sheet.py
   rigid_diaphragm_parametric_app.py
   rigid_diaphragm_parametric_app_GLOBAL_VISUAL_V2.py
   requirements.txt

ORIGINAL ENGINEERING MODULES (UNCHANGED)
   rigid_diaphragm_core.py
   wood_shearwall_mechanics.py
   test_calculation_sheet.py


