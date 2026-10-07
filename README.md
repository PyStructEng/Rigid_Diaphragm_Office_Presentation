RIGID DIAPHRAGM RESEARCH LAB — MECHANICS-BASED WOOD WALL V3
============================================================

DEPLOYMENT FILES
----------------
1. rigid_diaphragm_parametric_app.py     Streamlit entry point
2. rigid_diaphragm_core.py               rigid-diaphragm mechanics / parametric engine
3. wood_shearwall_mechanics.py           stacked wood shear-wall mechanics + embedded databases
4. requirements.txt                      Streamlit Cloud dependencies

STREAMLIT CLOUD
---------------
Point the app entry point to:
    rigid_diaphragm_parametric_app.py

The three Python files must be in the same repository folder.

WOOD-WALL MECHANICS IMPLEMENTED
-------------------------------
Input storeys are ordered BOTTOM -> TOP (Storey 1 is the lowest storey).
All lateral forces supplied to the wood-wall module are SERVICE-LEVEL inputs.
No strength load factors are applied internally.

The stacked-wall calculation is based on the attached FPInnovations/CWC mechanics method:
    Delta_i = Delta_b,i + Delta_s,i + Delta_n,i + Delta_a,i + Delta_r,i

Bending:
    Delta_b,i = V_i H_i^3 / [3(EI)_i] + M_i H_i^2 / [2(EI)_i]

Continuous-rod transformed section:
    n = Et/Ec
    At,tr = n At
    ytr = Ac Lc / (At,tr + Ac)
    Itr = At,tr ytr^2 + Ac (Lc-ytr)^2

Panel shear:
    Delta_s,i = V_i H_i / (L_i Bv,i)

Validated nail-slip relation supplied by user:
    en = [(0.013 * force_per_nail) / d_f^2]^2
    Delta_n,i = 0.0025 H_i en

Anchorage:
    da uses the existing validated script logic:
      take-up device load deformation + seating increment + compression-perpendicular deformation
    Delta_a,i = (H_i/L_i) da_i

Bottom rotation:
    Includes ONLY rotations from storeys below the current storey, per FPInnovations Eq. 13.
    Bottom-storey rotation contribution is therefore zero.

Secant stiffness:
    k_i = V_i / Delta_i

SYMMETRIC Lc RULE
-----------------
The app asks directly for "Chord studs / end" and assumes the same number at both ends.
    Lc = Ls - (n_chord * 38 + cavity)
Default cavity = 228.6 mm (9 in), matching the user's validated spreadsheet assumption.

EMBEDDED DATABASE EXPLORER
--------------------------
The active values are transcribed from shearwallanalysis_R2_faster.py and exposed in the app:
- sheathing Bv
- lumber E parallel / E perpendicular
- take-up devices
- rod resistance
- rod geometry / effective area data
- shear clips
- bearing plates
- panel buckling properties
- sill bolt capacities
- sill nail A and B capacities

R1/R2 NOTE
----------
The app does not silently merge conflicting legacy data between R1 and R2. The R2 data are the active
calculation tables in this V3 package. Database values remain visible in the app so they can be audited.

CURRENT COUPLING SCOPE
----------------------
The rigid-diaphragm model is currently a single diaphragm level. Automatic diaphragm <-> wood-wall
iteration is therefore enabled for a ONE-STOREY Wood Wall Lab model. Multi-storey wall stiffnesses can
be calculated and applied storey-by-storey, but a full simultaneous multi-level diaphragm coupling is a
future extension.

VALIDATION INCLUDED
-------------------
Rigid-diaphragm tests:
- CM/CR symmetry
- uniform stiffness scaling invariance
- fixed-k wall-length force invariance
- force/moment equilibrium
- FPInnovations benchmark

Wood mechanics tests:
- Storey 1 lower-storey rotation contribution = 0
- symmetric Lc geometry
- validated nail-slip equation
- current storey excluded from Eq. 13 lower-storey rotation
- storey shear accumulation

IMPORTANT
---------
The embedded values are reproduced from the supplied Python script; the app does not independently
establish the applicable code/manufacturer edition. Confirm selected property values and source editions
for the project before engineering use.
