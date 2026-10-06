Rigid Diaphragm Parametric Analysis — Streamlit

Files:
- rigid_diaphragm_parametric_app.py
- requirements_rigid_diaphragm.txt

Run locally:
1. Put both files in the same folder.
2. Open Terminal / PowerShell in that folder.
3. Install dependencies:
   pip install -r requirements_rigid_diaphragm.txt
4. Start the app:
   streamlit run rigid_diaphragm_parametric_app.py

Recommended modelling workflow:
- Input plan dimensions and center of mass.
- Input diaphragm-transferred Fx and Fy.
- Input actual lateral stiffness k for each wall/frame in kN/m.
- Use Local X/Y Force only for force resisted directly by the same parallel wall (for example, its own inertial wall weight), not force transferred through the diaphragm.
- Run the FPInnovations benchmark first to confirm the reference mechanics.
- Use the Parametric Sweep tab to vary a wall coordinate or stiffness.

Important:
This tool is for engineering analysis/study and requires independent verification of wall stiffness, code load cases, eccentricity provisions, and final design.
