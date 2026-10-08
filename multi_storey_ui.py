"""Streamlit UI for a self-contained multi-storey shear-building research tab."""
from __future__ import annotations
import math
import hashlib
import pandas as pd
import numpy as np
import streamlit as st
from multi_storey_core import example_inputs, analyze_multistorey, CASES
from multi_storey_report import make_multistorey_pdf


def _signature(floors, walls, args)->str:
    payload=(floors.to_json(orient='split')+walls.to_json(orient='split')+repr(args)).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def render_multi_storey_tab(current_walls:pd.DataFrame, settings:dict)->None:
    st.header("Multi-storey period & drift analysis")
    st.caption("Connected rigid diaphragms, 3 DOF per floor: Ux, Uy and plan rotation. An independent research model; existing one-storey studies are unchanged.")
    st.warning("**Analysis boundary:** A storey-shear-spring approximation, NOT a continuous stacked wood-wall model, complete Canadian seismic procedure, or building-code drift check. Enter your own floor forces and verify wall stiffnesses. The period is calculated from the stiffness and mass; it is NOT used to automatically recalculate seismic loads.")

    if 'ms_floors' not in st.session_state or 'ms_walls' not in st.session_state:
        f,w=example_inputs(current_walls,settings,6)
        st.session_state.ms_floors=f;st.session_state.ms_walls=w
        st.session_state.ms_revision=0
        st.session_state.pop('ms_result',None)

    c1,c2,c3=st.columns([1.0,1.1,2.8])
    n=int(c1.number_input("Desired number of storeys",min_value=1,max_value=12,value=6,step=1,key="ms_target_n"))
    if c2.button("Build / reset storeys",key="ms_build",use_container_width=True):
        f,w=example_inputs(current_walls,settings,n)
        st.session_state.ms_floors=f;st.session_state.ms_walls=w
        st.session_state.ms_revision+=1
        st.session_state.pop('ms_result',None)
        st.rerun()
    c3.info(f"Configured floors: {len(st.session_state.ms_floors)}. Rebuild to copy today's single-storey wall layout to every level. Then edit each level independently.")

    Lx=float(settings['Lx']);Ly=float(settings['Ly'])
    with st.expander("Model assumptions and the meaning of the calculated periods",expanded=False):
        st.markdown("""
- Each floor is a rigid plate with **three coupled in-plane DOFs** at a fixed global reference point.
- Each storey contains independent X- and Y-resisting **lateral springs**, attached between adjacent floors. They contribute plan torsional stiffness through their offsets.
- The mass matrix includes **plan polar mass inertia** for a uniformly distributed rectangular floor and translation/rotation coupling when the centre of mass moves.
- A wall's stiffness is an **entered effective storey stiffness in kN/m**. The program **does not** derive that stiffness from wall material, nails, anchorage or wall moment-curvature.
- **Modal periods:** eigenvalues of the global building K and M. **Rayleigh:** displacement-shape cross-check for a specific force case. **2H/3:** deliberately labelled illustrative and not a code design period.
- Floor forces, accidental eccentricity fraction and drift acceptance limits must be chosen for the **actual applicable code edition**. No automatic code-based design periods, force reduction or amplified drift checks.
- For continuous stacked walls, lower-storey flexure/rocking and nonlocal displacement coupling are important; the independent-storey-spring assumption cannot replace a detailed wall stiffness matrix.
""")

    st.markdown("### 1 — Floor geometry, seismic masses and applied loads")
    st.caption("Bottom storey = 1. Weight and floor X/Y forces are independent inputs. Example loads are illustrative, not a prescribed height distribution.")
    f=st.data_editor(st.session_state.ms_floors,hide_index=True,num_rows="fixed",use_container_width=True,
        key=f"ms_floor_editor_{st.session_state.ms_revision}",
        column_config={"Storey":st.column_config.NumberColumn(disabled=True),
                       "Height (m)":st.column_config.NumberColumn(format="%.3f",min_value=0.01),
                       "Weight (kN)":st.column_config.NumberColumn(format="%.2f",min_value=0.01),
                       "Fx (kN)":st.column_config.NumberColumn(format="%.2f"),
                       "Fy (kN)":st.column_config.NumberColumn(format="%.2f"),
                       "Xcm (m)":st.column_config.NumberColumn(format="%.3f"),
                       "Ycm (m)":st.column_config.NumberColumn(format="%.3f")})
    st.session_state.ms_floors=f
    st.markdown("### 2 — Resisting walls at every storey")
    st.caption("Each segment uses its own effective storey stiffness. A wall with the same name on two floors is represented by two independent shear springs, NOT a continuous bending wall.")
    w=st.data_editor(st.session_state.ms_walls,hide_index=True,num_rows="dynamic",use_container_width=True,
        key=f"ms_wall_editor_{st.session_state.ms_revision}",
        column_config={"Storey":st.column_config.NumberColumn(format="%d",min_value=1,max_value=len(f)),
            "Wall Name":st.column_config.TextColumn(required=True),
            "Direction":st.column_config.SelectboxColumn(options=['X','Y'],required=True),
            "x (m)":st.column_config.NumberColumn(format="%.3f"),
            "y (m)":st.column_config.NumberColumn(format="%.3f"),
            "k (kN/m)":st.column_config.NumberColumn(format="%.2f",min_value=0.001)})
    st.session_state.ms_walls=w

    # Explicit ONE-WAY transfer of the last Wood Wall Lab secant stiffness.
    # It is a fixed-demand snapshot, never presented as globally converged.
    mech=st.session_state.get("wood_mechanics_result")
    if isinstance(mech,pd.DataFrame) and {"Storey","k secant (kN/m)"}.issubset(mech.columns):
        with st.expander("Optional: import one stacked-wall line from Wood Wall Lab",expanded=False):
            st.warning("Transfers precomputed storey secant k values only. They reflect the Wood Wall Lab force/geometry assumptions and may include lower-storey rotation; this is NOT a compatible multi-wall mechanics iteration. Confirm the calibration and avoid double-counting flexibility.")
            matching=sorted(set(st.session_state.ms_walls["Wall Name"].astype(str)))
            line=st.selectbox("Building wall line to update",matching,key="ms_import_name")
            if st.button("Import Wood Wall Lab k at matching storeys",key="ms_import_mechanics"):
                try:
                    updated=st.session_state.ms_walls.copy()
                    transferred=0
                    for _,row in mech.iterrows():
                        num=int(row['Storey']); kv=float(row['k secant (kN/m)'])
                        mask=(updated['Storey']==num)&(updated['Wall Name'].astype(str)==line)
                        if mask.any():
                            if not math.isfinite(kv) or kv<=0:
                                raise ValueError(f"Storey {num} wood stiffness is invalid or infinite: {kv}")
                            updated.loc[mask,'k (kN/m)']=kv
                            transferred+=int(mask.sum())
                    if not transferred:
                        raise ValueError('No matching wall names/storeys found.')
                    st.session_state.ms_walls=updated
                    st.session_state.ms_revision+=1
                    st.session_state.pop('ms_result',None)
                    st.rerun()
                except Exception as exc:
                    st.error(f'Wood Lab import not applied: {exc}')

    a,b,c=st.columns(3)
    acc=float(a.number_input('Accidental eccentricity fraction',min_value=0.0,max_value=0.50,value=float(min(settings['acc'],0.50)),step=0.01,format='%.3f',key='ms_acc'))
    b.metric('Plan X span',f"{Lx:.2f} m")
    c.metric('Plan Y span',f"{Ly:.2f} m")
    st.caption("Accidental moments are applied separately as X± and Y±, using offsets ±fraction × perpendicular plan dimension. This is an input, not an automatically determined code requirement.")

    st.markdown("### 3 — Analysis choice")
    nonlinear=st.checkbox("Explore demand-dependent effective stiffness (UNCALIBRATED sensitivity only)",value=False,key='ms_nonlinear')
    exponent=1.15; ref=100.; relax=.5; tol=.01; max_it=30
    if nonlinear:
        st.warning("Sensitivity-only secant law: k_target = max(0.25 k_initial, k_initial × max(|V_envelope|/V_ref,1)^(1-p)). This is NOT the wood-mechanics model or a full period-force convergence design procedure.")
        d1,d2,d3,d4,d5=st.columns(5)
        exponent=float(d1.number_input('Deformation exponent p',min_value=1.,max_value=2.,value=1.15,step=.05))
        ref=float(d2.number_input('Reference force (kN)',min_value=0.1,value=100.,step=10.))
        relax=float(d3.number_input('Damping factor',min_value=.05,max_value=1.,value=.5,step=.05))
        tol=float(d4.number_input('Convergence fraction',min_value=.0001,max_value=.20,value=.01,step=.005,format='%.4f'))
        max_it=int(d5.number_input('Max iterations',min_value=1,max_value=100,value=30))
    args=(Lx,Ly,acc,nonlinear,exponent,ref,relax,tol,max_it)
    sig=_signature(f,w,args)
    if st.button('Run multi-storey analysis',type='primary',use_container_width=True,key='ms_run'):
        try:
            with st.spinner('Assembling and solving global matrices, load cases and eigenmodes...'):
                st.session_state.ms_result=analyze_multistorey(f,w,Lx,Ly,acc,nonlinear=nonlinear,
                 exponent=exponent,reference_force=ref,relaxation=relax,tolerance=tol,max_iterations=max_it)
                st.session_state.ms_last_signature=sig
                rr=st.session_state.ms_result
                log=list(st.session_state.get('ms_design_history',[]))
                log.append({
                    'Run':len(log)+1, 'Levels':len(f),
                    'Total Fx (kN)':float(f['Fx (kN)'].sum()),
                    'Total Fy (kN)':float(f['Fy (kN)'].sum()),
                    'T1 (s)':float(rr['first_mode_s']),
                    'TX (s)':float(rr['modal_x_s']),
                    'TY (s)':float(rr['modal_y_s']),
                    'Max drift (%)':float(rr['max_drift_pct']),
                    'Min k / initial':float(min(rr['stiffness_ratio'])),
                    'Converged':bool(rr['converged']),
                })
                st.session_state.ms_design_history=log[-100:]
        except Exception as err:
            st.session_state.pop('ms_result',None)
            st.error(f"Multi-storey model cannot be solved: {err}")
    if 'ms_result' not in st.session_state:
        st.info('Press **Run multi-storey analysis** to generate results. The existing single-storey solver is unaffected.')
        return
    if st.session_state.get('ms_last_signature') !=sig:
        st.warning('These results belong to the previous input configuration. Rerun the multi-storey analysis to update them; exports are disabled until refreshed.')
        return
    r=st.session_state.ms_result
    if nonlinear and not r['converged']:
        st.error("The stiffness sensitivity iteration DID NOT CONVERGE. Increase iteration limit, reduce the damping factor, or inspect stiffness assumptions. Do not rely on its final state as a converged model.")
    else:
        st.success('Global stiffness and mass matrices solved; four distinct static cases and modal periods calculated. Static force/moment residuals checked numerically.')
    st.markdown('### 4 — Global periods, drift and force response')
    m=st.columns(5)
    for block,label,value in zip(m,['First mode period','X-dominant mode','Y-dominant mode','Max elastic drift','Static residual (mixed units)'],
      [f"{r['first_mode_s']:.3f} s",f"{r['modal_x_s']:.3f} s",f"{r['modal_y_s']:.3f} s",f"{r['max_drift_pct']:.3f}%",f"{r['max_residual']:.1e}"]):
        block.metric(label,value)
    if r['max_residual']>1e-5:
        st.error('Force-equilibrium residual is higher than expected; check the numerical conditioning.')
    if not np.isfinite(r['first_mode_s']):
        st.error('Unusable modal period.');return

    rx,ry,rz=st.tabs(['Floor response & wall forces','Periods / iteration / 2H/3','Equations and printable report'])
    with rx:
        name=st.selectbox('One simultaneous load case',list(CASES),key='ms_case')
        selected=r['floors'].query('Case == @name').copy()
        st.dataframe(selected.round(4),hide_index=True,use_container_width=True)
        cc=st.columns(2)
        cc[0].line_chart(selected.set_index('Elevation (m)')[['Ux (mm)','Uy (mm)']],x_label='Elevation (m)',y_label='Floor translation (mm)')
        cc[1].line_chart(selected.set_index('Elevation (m)')[['Max X drift (%)','Max Y drift (%)']],x_label='Elevation (m)',y_label='Elastic drift at worst corner (%)')
        st.markdown('**Wall forces for the selected case (simultaneous)**')
        st.dataframe(r['wall_forces'].query('Case == @name').round(4),hide_index=True,use_container_width=True)
        st.markdown('**Wall force envelope across four separate cases**')
        st.caption('An envelope combines maxima from possibly different accidental-eccentricity signs; do not sum envelope values as a simultaneous force state.')
        st.dataframe(r['envelope'].round(4),hide_index=True,use_container_width=True)
    with ry:
        st.markdown('**Global vibration modes and effective X/Y participating masses**')
        st.dataframe(r['modes'].round(4),hide_index=True,use_container_width=True)
        st.caption('The first mode can be translation, torsion, or mixed. The X/Y-dominant periods identify the modes carrying the most translational effective mass, not separate wall natural periods.')
        st.markdown('**Rayleigh check — full displacement shape for each static load case**')
        df_ray=pd.DataFrame([{"Case":case,"Rayleigh period (s)":v['rayleigh_s'],"Static residual":v['residual']} for case,v in r['case_results'].items()])
        st.dataframe(df_ray.round(6),hide_index=True,use_container_width=True)
        st.markdown('**Two-thirds-height reference diagnostic**')
        st.warning('Only a transparent teaching approximation. Uses total entered base shear V_D (sum of ALL floor forces) and total building weight W, as in the historical APEGBC Appendix E alternative-period arithmetic. This is a rough comparison, NOT a code design period for the project.')
        st.dataframe(r['approx'].round(4),hide_index=True,use_container_width=True)
        st.markdown('**Manual design-run comparison (user-edited wall stiffnesses and floor loads)**')
        design_history=st.session_state.get('ms_design_history',[])
        if design_history:
            dh=pd.DataFrame(design_history)
            dh['Change in T1 (%)']=100*dh['T1 (s)'].pct_change()
            dh['Change in drift (%)']=100*dh['Max drift (%)'].pct_change()
            st.dataframe(dh.round(4),hide_index=True,use_container_width=True)
            st.line_chart(dh.set_index('Run')[['T1 (s)','TX (s)','TY (s)']])
            st.caption('Run-to-run changes are historical comparisons, not proof of convergence: every run can use different user-selected stiffnesses, masses, and seismic loads. Check period, force and drift stability independently.')
            aa,bb=st.columns(2)
            aa.download_button('Export design-run history CSV',dh.to_csv(index=False).encode('utf-8'),file_name='multistorey_design_iteration_history.csv',mime='text/csv',key='ms_history_csv')
            if bb.button('Clear manual run history',key='ms_clear_history'):
                st.session_state.ms_design_history=[]
                st.rerun()
        if nonlinear:
            st.markdown('**Fixed-force sensitivity iteration history**')
            st.dataframe(r['history'].round(5),hide_index=True,use_container_width=True)
            st.line_chart(r['history'].set_index('Iteration')[['T1 (s)']])
            st.caption(f"Converged: {'yes' if r['converged'] else 'NO'}; {len(r['history'])} iterations. Force vectors are fixed across iterations and are NOT recalculated from period.")
    with rz:
        st.markdown('**Analysis equations with explicit model assumptions**')
        st.latex(r"\mathbf q_i=\begin{bmatrix}u_{x,i}&u_{y,i}&\theta_i\end{bmatrix}^{T}")
        st.latex(r"\mathbf a_{X,j}=\begin{bmatrix}1&0&-(y_j-y_0)\end{bmatrix}^{T},\quad \mathbf a_{Y,j}=\begin{bmatrix}0&1&(x_j-x_0)\end{bmatrix}^{T}")
        st.latex(r"\mathbf S_i=\sum_{j\in i}k_j\mathbf a_j\mathbf a_j^{T},\qquad V_{j,i}=k_j\mathbf a_j^{T}(\mathbf q_i-\mathbf q_{i-1})")
        st.latex(r"\mathbf K\mathbf q=\mathbf F,\qquad\mathbf K\boldsymbol\phi=\omega^2\mathbf M\boldsymbol\phi,\qquad T=\frac{2\pi}{\omega}")
        st.latex(r"T_{R}=2\pi\sqrt{\frac{\mathbf q^{T}\mathbf M\mathbf q}{\mathbf q^{T}\mathbf K\mathbf q}}")
        st.latex(r"h_{ref}=\frac{2H}{3},\quad K_{ref}=\frac{|V_{ref}|}{|u(h_{ref})|},\quad T_{ref}=2\pi\sqrt{\frac{W_{total}}{gK_{ref}}}")
        st.caption('The stiffness matrix uses wall inputs k in kN/m; mass is in kN s²/m; theta is in radians. Streamlit shows equations via st.latex; numerical matrix calculations use NumPy.')
        s=int(st.selectbox('Inspect assembled storey stiffness matrix',list(range(1,len(f)+1)),key='ms_Ki'))
        st.dataframe(pd.DataFrame(r['storey_K'][s-1],index=['Ux','Uy','theta'],columns=['Ux','Uy','theta']).round(4),use_container_width=True)
        st.caption('Rotational matrix terms have different dimensional units (kN·m, kN·m²) from translational terms (kN/m). These are NOT interchangeable scalar spring stiffnesses.')
        st.markdown('**Export calculations, inputs, cases, modes, and envelope**')
        if r['nonlinear'] and not r['converged']:
            st.warning('PDF will visibly flag non-convergence in the results summary.')
        try:
            pdf=make_multistorey_pdf(r)
            st.download_button('Download multi-storey calculation report (PDF)',data=pdf,
                file_name='multi_storey_period_drift_calculations.pdf',mime='application/pdf',use_container_width=True,key='ms_pdf')
        except Exception as exc:
            st.error(f'Could not generate calculation PDF: {exc}')
        st.download_button('Download floor response CSV',data=r['floors'].to_csv(index=False).encode('utf-8'),file_name='multistorey_floor_response.csv',mime='text/csv',key='ms_floor_csv')
        st.download_button('Download all wall load-case forces CSV',data=r['wall_forces'].to_csv(index=False).encode('utf-8'),file_name='multistorey_wall_forces.csv',mime='text/csv',key='ms_wall_csv')
        st.download_button('Download wall envelope CSV',data=r['envelope'].to_csv(index=False).encode('utf-8'),file_name='multistorey_wall_envelope.csv',mime='text/csv',key='ms_envelope_csv')
