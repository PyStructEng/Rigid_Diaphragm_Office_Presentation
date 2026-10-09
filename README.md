# Rigid Diaphragm Research Lab — Wall Design Studio (Phase 1 + Phase 2)

**October 8, 2026** — construction-aware wall records and individually evaluated trial wall mechanics. This is an incremental research release and **does NOT complete the phase-3 whole-building mechanics-based convergence**.

## Install / deploy

1. Back up your GitHub repository. The complete ZIP includes the existing October 8 multi-storey, Handcalcs and ForAllPeople bugfixes and calculation tabs.
2. Extract the **complete** ZIP and copy its Python source files plus `requirements.txt` together into the app directory. Do not rename your existing Streamlit entrypoint.
3. Run the existing entrypoint (`rigid_diaphragm_parametric_app.py`, or alternatively `rigid_diaphragm_parametric_app_GLOBAL_VISUAL_V2.py`). Both contain a new top-level **Wall design studio** tab.
4. If instead applying the **patch** ZIP, use it ONLY atop the October 8 multi-storey period/drift release, because it references the modules already present in that release.
5. Commit and redeploy. Existing solver methods, charts and calculation tabs were not rewritten. No live GitHub modifications were made while preparing these ZIPs.

## Workflow

1. In **Multi-storey periods & drift**, create the desired number of storeys and wall lines; run the global calculation if you want a specific case's trial wall shear.
2. Open **Wall design studio → Assembly library**. Edit or add construction presets: panel (OSB, DFP, CSP), thickness, sheathing sides, edge/field nail spacing, diameter and length, species/grade, studs, chord pack, continuous rod and take-up device. Presets are examples; manufacturer and design applicability must be verified.
3. Open **Wall schedule & checks**. Click **Sync wall IDs** after changing the global model. Preserve existing wall designs by Storey + Wall Name. Enter wall lengths, gravity loads and trial shear; assign presets per segment.
4. To change an individual wall without changing shared presets, expand **Design one individual wall**, choose Storey + Wall Name, edit physical inputs and click **Save this design for selected wall only**. A wall-specific copy of the assembly is created.
5. Select **Manual trial shear** or a **single simultaneous** previously calculated `X +`, `X -`, `Y +`, `Y -` global case. A single case is never an envelope and is not automatically factored or converted to service demand. Re-run if the global analysis has become stale.
6. Click **Evaluate every wall** to calculate per-segment isolated trial deflection from bending, panel shear, nail slip, anchorage; examine service-estimate tension, stiffness and a deflection breakdown. For illustrative high demand, trial drift over **2% is flagged as a research review heuristic, NOT a code acceptance limit**.
7. For *strength comparison only*, enter **independently verified** factored wall shear/tension demands and factored resistances. Zero means **NOT CHECKED**, not a pass. No CSA O86 code capacity is automatically produced.
8. Download JSON (reusable design input project), CSV schedule and a printable PDF of evaluated results. JSON restore requires the same set of global Storey + Wall Name IDs.
9. Optional: from **Export / transfer**, push current positive, finite, non-flagged **trial** stiffnesses one way into the linear multi-storey model and manually rerun it. The application never labels this as globally compatible nonlinear convergence. Stiffness from an overstressed/high-deformation trial is blocked from automatic transfer.

## Added modules

- `wall_design_core.py` — assembly library and validated wall schedule, demand-aware *isolated single-storey* wall snapshots, independent user-input strength comparisons, JSON import/export and safeguards.
- `wall_design_ui.py` — assembly editor, individual-wall card, editable global wall schedule, demand-case selection, review table, JSON/CSV/PDF exports and opt-in one-way transfer.
- `wall_design_report.py` — landscape printable wall trial / assembly specification PDF.
- `test_wall_design.py` — independent wall sensitivity, dimensional calculation, zero-demand, mixed presets, capacity-input, import/export, load-case, and PDF tests.
- `test_wall_design_ui_offline.py` — simulated Streamlit UI render (not a real browser test).
- BOTH `rigid_diaphragm_parametric_app*.py` files are updated to include the final **Wall design studio** tab.

## Meaning of stiffness

For an isolated wall at a specified **nonzero** trial shear:

`k_trial [kN/m] = 1000 * |V_trial [kN]| / Delta_trial [mm]`.

The deformation engine is inherited unchanged from `wood_shearwall_mechanics.py`, with the user's embedded panel Bv, nail-slip and continuous rod/take-up lookups. It uses the legacy research equations rather than a full material hysteresis model. **Wall drift under zero trial shear is intentionally reported as not evaluated** instead of implying an elastic secant from constant take-up seating.

**Crucial limits**:

- The individual wall trial treats each segment as **isolated, single-storey**, with no moment delivered from the wall above or flexural/rocking rotation transferred from lower storeys. It does not satisfy all six-storey compatibility conditions.
- Shear stiffness may vary with demand and is used for trial comparison. Reversing forces currently uses `|V|`, and does not represent distinct tension-end connections, slack/engagement, uplift hysteresis or the sign of gravity preloading.
- Nail **field spacing and nail length are recorded only**; they do NOT currently change shear capacity OR deformational stiffness. Nail **edge spacing and nail diameter** feed the existing nail-slip equation. Panel Bv database entries and the two-sided multiplier reflect the inherited simplifying assumptions, not universal product certifications.
- Actual shear and hold-down factored capacities are **NOT derived** from panel, nail or rod dimensions in this release, and status is only a comparison of user-entered independently verified factored demand and factored resistance. All code factors/adjustments are outside this module.
- High trial drift is a warning of potentially unreasonable demand, extrapolation or inconsistent assembly; **2% is only a prototype screening heuristic**, not a BCBC drift acceptance limit.
- Storey period eigenvalues and force results from `multi_storey_core` remain a preliminary linear storey-spring approximation. An opt-in transfer of *isolated* wall trial stiffness is **not the requested globally compatible nonlinear iteration**.
- No Canadian seismic forces, code period limits, NBCC/BCBC drift acceptance, capacity, anchors, overturning checks, collector/chords, shearwall overturning restraint and load-path verification are automatically established.

## Tests (Python 3.10+)

```bash
python -m pip install -r requirements.txt
python -m unittest test_wall_design test_multi_storey -v
python test_wall_design_ui_offline.py
python test_multi_storey_ui_offline.py
python test_calculation_sheet.py
python test_upgrade_offline.py
python test_upgrade_live.py
streamlit run rigid_diaphragm_parametric_app.py
```

`test_upgrade_live.py` requires actual installed Handcalcs and ForAllPeople dependencies. The app's full runtime/browser interaction has not been tested in this build environment because Streamlit is not installed here.

## Next engineering milestone — Phase 3

1. Implement a validated multi-level wall element delivering a consistent storey stiffness/tangent matrix and signed restoring-force vector, including axial/rocking continuity, load-dependent anchors, gravity, and wall-over-wall flexure.
2. Assemble all elements in the existing 3-DOF/floor global model and solve a **distinct nonlinear equilibrium** per *simultaneous* load case. Avoid using an envelope as a single nonlinear force state.
3. Inner solve: equilibrium and displacement compatibility; outer engineering redesign: choose assemblies, check independently validated CSA O86 capacities/drift and rerun after each revision.
4. Calculate rational modal/Rayleigh periods using converged model stiffness, and build a separately validated NBCC/BCBC seismic-load feedback layer with correct code period limits.
5. Validate against a simple hand-check, independent frame/finite-element model, FPInnovations stacked wood-wall case, and EGBC/CWC published multi-storey cases before any project use.

### Literature

- APEGBC (2015), *5 and 6 Storey Wood Frame Residential Building Projects*, Appendix E: https://www.egbc.ca/getmedia/eea8aecd-8407-4fdf-8076-3f4f6eac5260/APEGBC-Technical_and_Practice_Bulletin_on_Mid-Rise_Buildings.pdf
- FPInnovations / CWC, *Design of stacked multistorey wood shearwalls using a mechanics-based approach*: https://old.cwc.ca/wp-content/uploads/2019/03/Design-of-stacked-multistorey-wood-shearwalls-using-a-mechanics-based-approach.pdf
- CWC (2025), *Innovative Strategies for Light-Frame Mid-Rise Buildings in High-Seismic Regions*: https://cwc.ca/wp-content/uploads/2025/09/Sep1025_CWC_LateralDesignExample.pdf
