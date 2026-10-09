"""Wall design studio, deliberately isolated from existing global solver."""
from __future__ import annotations

import hashlib
import math
import pandas as pd
import numpy as np
import streamlit as st

from multi_storey_core import example_inputs, WALL_COLS
from wood_shearwall_mechanics import SHEATHING_BV, ROD_GEOMETRY, TAKEUP_DEVICES, LUMBER_E
from wall_design_core import (ASSEMBLY_COLS, WALL_FIELDS, default_assemblies,
    default_wall_schedule, sync_schedule, validate_assemblies, validate_schedule,
    evaluate_schedule, export_design_json, import_design_json, strength_review)
from wall_design_report import make_wall_design_pdf


def _same_global_snapshot(result, floors, walls):
    """Never silently reuse obsolete forces after the wall stiffness/layout changes."""
    if not isinstance(result, dict) or 'input_walls' not in result or 'input_floors' not in result:
        return False
    try:
        x=result['input_walls'].sort_values(['Storey','Wall Name']).reset_index(drop=True)
        y=walls[list(WALL_COLS)].sort_values(['Storey','Wall Name']).reset_index(drop=True)
        a=result['input_floors'].sort_values('Storey').reset_index(drop=True)
        b=floors[a.columns].sort_values('Storey').reset_index(drop=True)
        return (x.equals(y) and a.equals(b) and bool(result.get('converged',False))
            and math.isclose(float(result.get('settings',{}).get('acc',-1.0)),
                float(st.session_state.get('ms_acc',result.get('settings',{}).get('acc',-2.0))),
                rel_tol=0,abs_tol=1e-12))
    except Exception:
        return False


def _edit_revision():
    st.session_state.wd_rev=int(st.session_state.get('wd_rev',0))+1


def render_wall_design_studio(current_walls:pd.DataFrame, settings:dict)->None:
    st.header('Wall design studio')
    st.caption('Assembly library → wall-by-wall construction → isolated mechanics trial → optional one-way stiffness transfer to the global model.')
    st.warning('**Engineering boundary:** A trial wood shearwall deformation calculator, NOT automatic CSA O86 design, complete stacked-wall analysis, or compatible nonlinear building convergence. Strengths must be separately verified. Every transfer requires an explicit button.')

    if 'ms_floors' not in st.session_state or 'ms_walls' not in st.session_state:
        f,w=example_inputs(current_walls,settings,6)
        st.session_state.ms_floors=f;st.session_state.ms_walls=w
        st.session_state.ms_revision=0
    f=st.session_state.ms_floors.copy()
    w=st.session_state.ms_walls.copy()
    if 'wd_assemblies' not in st.session_state:
        st.session_state.wd_assemblies=default_assemblies()
    if 'wd_schedule' not in st.session_state:
        seeded=default_wall_schedule(w)
        if 'Wall Length (m)' in current_walls.columns:
            lengths=dict(zip(current_walls['Wall Name'].astype(str),current_walls['Wall Length (m)']))
            seeded['Wall length (m)']=seeded.apply(lambda z:float(lengths.get(str(z['Wall Name']),4.8)),axis=1)
        st.session_state.wd_schedule=seeded
    if 'wd_rev' not in st.session_state:
        st.session_state.wd_rev=0
    st.info(f"Global model: {len(f)} storeys, {len(w)} segments. Wall IDs come from **Multi-storey periods & drift**. Changes to the global wall layout should be followed by Sync wall IDs below.")

    area_a,area_b,area_c=st.tabs(['Assembly library','Wall schedule & checks','Export / transfer'])
    with area_a:
        st.markdown('### 1 — Reusable wall assemblies')
        st.caption('The library holds actual construction selections; it does not invent code shear resistance. Listed rod/take-up properties are research references, not project approvals.')
        a=st.data_editor(st.session_state.wd_assemblies,hide_index=True,num_rows='dynamic',use_container_width=True,
            key=f'wd_assembly_editor_{st.session_state.wd_rev}',column_config={
            'Panel Type':st.column_config.SelectboxColumn(options=['OSB','DFP','CSP'],required=True),
            'Panel sides':st.column_config.SelectboxColumn(options=['S.S','B.S'],required=True),
            'Stud size':st.column_config.SelectboxColumn(options=['2x4','2x6'],required=True),
            'Assembly ID':st.column_config.TextColumn(required=True)})
        st.session_state.wd_assemblies=a
        if st.button('Validate assembly library',key='wd_audit_assemblies'):
            try:
                validate_assemblies(a)
                st.success('Assembly fields and existing mechanics lookups are valid. Strength capacity is NOT checked.')
            except Exception as exc:
                st.error(str(exc))
        st.caption('Only the **edge nail spacing** and **nail diameter** feed the inherited nail-slip equation. Nail length and field spacing are documented design/detailing inputs pending a validated connection-capacity model.')

    with area_b:
        st.markdown('### 2 — Storey-by-storey wall schedule')
        ct1,ct2=st.columns([1.0,2.5])
        if ct1.button('Sync wall IDs',key='wd_sync',help='Retains designs for matching Storey + Wall Name, adds new walls and drops deleted walls'):
            st.session_state.wd_schedule=sync_schedule(st.session_state.wd_schedule,w)
            _edit_revision()
            st.session_state.pop('wd_result',None)
            st.rerun()
        ct2.caption('Rows are uniquely identified by **Storey + Wall Name**; wall coordinates/direction are defined in the global tab. The schedule stores individual segment lengths and gravity loads.')
        d=st.data_editor(st.session_state.wd_schedule,num_rows='fixed',hide_index=True,use_container_width=True,
            key=f'wd_schedule_editor_{st.session_state.wd_rev}',disabled=['Storey','Wall Name'],
            column_config={'Assembly ID':st.column_config.SelectboxColumn(options=sorted(st.session_state.wd_assemblies['Assembly ID'].dropna().astype(str).tolist()),required=True),
                'Wall length (m)':st.column_config.NumberColumn(min_value=0.01),
                'Trial shear (kN)':st.column_config.NumberColumn(min_value=0.0),
                'Verified shear resistance (kN)':st.column_config.NumberColumn(min_value=0.0),
                'Verified tension resistance (kN)':st.column_config.NumberColumn(min_value=0.0)})
        st.session_state.wd_schedule=d
        with st.expander('Design one individual wall — panel, nail, framing and anchorage',expanded=False):
            st.caption('Saving here creates/updates a PRIVATE assembly for the selected wall. Other walls using shared presets are unaffected. This is an individual design option, not an automatic code capacity calculation.')
            pairs=[(int(z['Storey']),str(z['Wall Name'])) for _,z in d.iterrows()]
            selected=st.selectbox('Selected wall',pairs,format_func=lambda v:f'Storey {v[0]} — {v[1]}',key='wd_select_wall')
            entry=d[(d['Storey']==selected[0])&(d['Wall Name']==selected[1])].iloc[0]
            ass=st.session_state.wd_assemblies
            match=ass[ass['Assembly ID'].astype(str)==str(entry['Assembly ID'])]
            if match.empty:
                st.error('Selected wall refers to a missing assembly. Repair its Assembly ID above.')
            else:
                current=match.iloc[0]
                with st.form(key=f'wd_unique_form_{selected[0]}_{selected[1]}_{st.session_state.wd_rev}'):
                    c1,c2,c3=st.columns(3)
                    panel=c1.selectbox('Panel type', ['OSB','DFP','CSP'],index=['OSB','DFP','CSP'].index(str(current['Panel Type'])))
                    thickness=c2.number_input('Panel thickness (mm) — exact database value',min_value=1.,value=float(current['Panel thickness (mm)']),step=.5)
                    faces=c3.selectbox('Sheathing sides',['S.S','B.S'],index=['S.S','B.S'].index(str(current['Panel sides'])))
                    d1,d2,d3,d4=st.columns(4)
                    dia=d1.number_input('Nail diameter (mm)',min_value=.1,value=float(current['Nail diameter (mm)']),step=.1)
                    edge=d2.number_input('Edge nail spacing (mm)',min_value=1.,value=float(current['Nail edge spacing (mm)']),step=10.)
                    field=d3.number_input('Field nail spacing (mm)',min_value=1.,value=float(current['Nail field spacing (mm)']),step=10.)
                    length=d4.number_input('Nail length (mm)',min_value=1.,value=float(current['Nail length (mm)']),step=5.)
                    e1,e2,e3,e4=st.columns(4)
                    species=e1.selectbox('Species',sorted(LUMBER_E['Species'].astype(str).unique()),index=sorted(LUMBER_E['Species'].astype(str).unique()).index(str(current['Species'])))
                    grades=sorted(LUMBER_E.loc[LUMBER_E['Species']==species,'Grade'].astype(str).unique())
                    grade=e2.selectbox('Grade',grades,index=grades.index(str(current['Grade'])) if str(current['Grade']) in grades else 0)
                    stud=e3.selectbox('Stud size',['2x4','2x6'],index=['2x4','2x6'].index(str(current['Stud size'])))
                    chord=e4.number_input('Chord studs per end',min_value=1,max_value=40,value=int(current['Chord studs / end']),step=1)
                    r1,r2=st.columns(2)
                    rod_ids=[x for x in ROD_GEOMETRY['Strong Rod Standard'].tolist()+ROD_GEOMETRY['Strong Rod High Strength'].tolist() if isinstance(x,str)]
                    rod=r1.selectbox('Rod model',rod_ids,index=rod_ids.index(str(current['Rod model'])) if str(current['Rod model']) in rod_ids else 0)
                    tud_ids=TAKEUP_DEVICES['Model No.'].astype(str).tolist()
                    takeup=r2.selectbox('Take-up device',tud_ids,index=tud_ids.index(str(current['Take-up device'])) if str(current['Take-up device']) in tud_ids else 0)
                    notes=st.text_input('Assembly source / engineering notes',value=str(current.get('Source / notes','')))
                    save_unique=st.form_submit_button('Save this design for selected wall only',use_container_width=True)
                if save_unique:
                    import re
                    unique_id=('W-'+str(selected[0])+'-'+re.sub('[^A-Za-z0-9_-]','_',str(selected[1]))[:24]+'-'+hashlib.sha256(str(selected[1]).encode('utf-8')).hexdigest()[:8])
                    rec={'Assembly ID':unique_id,'Panel Type':panel,'Panel thickness (mm)':thickness,'Panel sides':faces,
                         'Nail diameter (mm)':dia,'Nail edge spacing (mm)':edge,'Nail field spacing (mm)':field,
                         'Nail length (mm)':length,'Species':species,'Grade':grade,'Stud size':stud,
                         'Chord studs / end':chord,'Rod model':rod,'Take-up device':takeup,'Source / notes':notes}
                    updated=ass.loc[ass['Assembly ID'].astype(str)!=unique_id].copy()
                    updated=pd.concat([updated,pd.DataFrame([rec])],ignore_index=True)
                    try:
                        validate_assemblies(updated)
                        st.session_state.wd_assemblies=updated
                        schedule=st.session_state.wd_schedule.copy()
                        mask=(schedule['Storey']==selected[0])&(schedule['Wall Name']==selected[1])
                        schedule.loc[mask,'Assembly ID']=unique_id
                        st.session_state.wd_schedule=schedule
                        st.session_state.pop('wd_result',None)
                        _edit_revision()
                        st.rerun()
                    except Exception as exc:
                        st.error(f'Individual design was not saved: {exc}')
        with st.expander('Explain input loads and capacity fields',expanded=False):
            st.markdown('**Trial shear** is the unfactored/service-level force used only for the wall mechanics model; it is not the NBCC code seismic demand. **Factored demands and verified resistances** are separate, independently supplied inputs. Zero means **not checked**, never passing. Rod manufacturer lookups are not used as a substitute for verified resistance.')
        r=st.session_state.get('ms_result')
        fresh=_same_global_snapshot(r,f,w)
        case_options=['Manual trial shear from wall schedule']
        if fresh:
            case_options += ['X +','X -','Y +','Y -']
        else:
            st.caption('Run or rerun the global model with current inputs to use its simultaneous wall forces as trial deformation demands. Stale, nonconverged, or changed global results are not accepted.')
        chosen=st.selectbox('Trial shear source',case_options,key='wd_load_source')
        case_forces=None
        if chosen!='Manual trial shear from wall schedule':
            case_forces=r['wall_forces'].loc[r['wall_forces']['Case']==chosen].copy()
            st.caption('Using the **one selected simultaneous case**, not an envelope. These global shears are input-level analysis shears and are not automatically converted into factored design forces.')
        if st.button('Evaluate every wall',type='primary',use_container_width=True,key='wd_eval'):
            try:
                checked_a=validate_assemblies(st.session_state.wd_assemblies)
                checked_d=validate_schedule(st.session_state.wd_schedule,checked_a,w,f)
                result=evaluate_schedule(checked_d,checked_a,w,f,case_forces)
                st.session_state.wd_result=result
                st.session_state.wd_result_meta={'source':chosen,'assem_sig':checked_a.to_json(orient='split'),
                    'sched_sig':checked_d.to_json(orient='split'),'floor_sig':f.to_json(orient='split'),
                    'wall_sig':w.to_json(orient='split')}
                st.success(f'Computed {len(result)} isolated-wall trial responses. No global model changes have been made.')
            except Exception as exc:
                st.session_state.pop('wd_result',None)
                st.error(f'Cannot evaluate all wall designs: {exc}')
        last=st.session_state.get('wd_result')
        meta=st.session_state.get('wd_result_meta',{})
        up_to_date=(isinstance(last,pd.DataFrame) and not last.empty
          and meta.get('assem_sig')==st.session_state.wd_assemblies[ASSEMBLY_COLS].to_json(orient='split')
          and meta.get('sched_sig')==st.session_state.wd_schedule[WALL_FIELDS].to_json(orient='split')
          and meta.get('floor_sig')==f.to_json(orient='split') and meta.get('wall_sig')==w.to_json(orient='split')
          and meta.get('source')==chosen)
        st.session_state.wd_result_is_current=bool(up_to_date)
        if isinstance(last,pd.DataFrame):
            if not up_to_date:
                st.warning('Trial results are STALE. Re-evaluate after editing the assemblies, wall schedule, floors, global walls or demand source. Transfer/export of results is disabled.')
            else:
                st.markdown('### 3 — Trial deformation / stiffness results')
                show=['Storey','Wall Name','Assembly ID','Source shear (kN)','Trial deflection (mm)',
                      'Trial k secant (kN/m)','Trial drift (%)','Drift review flag','Nail force (N)','Rod service tension estimate (kN)',
                      'Shear status','Hold-down tension status']
                st.dataframe(last[show].round(3),hide_index=True,use_container_width=True)
                st.caption('Trial k = |V| / Δ. The single-storey mechanics model includes bending, panel shear, nail slip and anchorage. Constant device seating and simplified bearing/hold-down forces require project verification. Drift above 2% is highlighted as a research review heuristic, NOT a code limit.')
                high=int(last['Drift review flag'].str.startswith('HIGH').sum())
                if high:
                    st.error(f'{high} individual wall trial responses exceed 2% deformation and are flagged for engineering investigation. The historical nail-slip law may be outside its suitable demand range.')
                ids=[(int(z['Storey']),str(z['Wall Name'])) for _,z in last.iterrows()]
                pick=st.selectbox('View individual wall deformation breakdown',ids,format_func=lambda x:f'Storey {x[0]} · {x[1]}',key='wd_detail_pick')
                one=last[(last['Storey']==pick[0])&(last['Wall Name']==pick[1])].iloc[0]
                comp={k:one[k] for k in ['Bending (mm)','Panel shear (mm)','Nail slip contribution (mm)','Anchorage contribution (mm)']}
                st.bar_chart(pd.DataFrame({'Deflection (mm)':comp}))
                st.latex(r'k_{\mathrm{sec}}=\frac{|V|}{\Delta_{\mathrm{trial}}}')
                st.caption('The trial analysis excludes rotations inherited from lower storeys. The global model has NOT been re-solved with the calculated values.')
    with area_c:
        st.markdown('### 4 — Save, restore and compare')
        try:
            a=validate_assemblies(st.session_state.wd_assemblies)
            d=validate_schedule(st.session_state.wd_schedule,a,w,f)
        except Exception as exc:
            st.error(f'Correct input data before exporting or transferring: {exc}')
            a=d=None
        if a is not None:
            st.download_button('Download wall-design project JSON',data=export_design_json(a,d),file_name='wall_assembly_design_project.json',mime='application/json',key='wd_json')
            st.download_button('Download wall schedule CSV',data=d.to_csv(index=False).encode('utf-8'),file_name='individual_wall_design_schedule.csv',mime='text/csv',key='wd_csv')
        imported=st.file_uploader('Restore a previously exported wall-design project JSON',type=['json'],key='wd_import_file')
        if imported is not None and st.button('Validate and load JSON',key='wd_import_do'):
            try:
                new_a,new_d=import_design_json(imported.getvalue(),w,f)
                st.session_state.wd_assemblies=new_a
                st.session_state.wd_schedule=new_d
                st.session_state.pop('wd_result',None)
                _edit_revision()
                st.rerun()
            except Exception as exc:
                st.error(f'Import not applied: {exc}')
        last=st.session_state.get('wd_result')
        good=bool(st.session_state.get('wd_result_is_current',False))
        if a is not None and good and isinstance(last,pd.DataFrame):
            try:
                pdf=make_wall_design_pdf(a,d,last,st.session_state.get('wd_result_meta',{}).get('source','Manual trial shear'))
                st.download_button('Download trial wall-design report (PDF)',data=pdf,file_name='wall_assembly_trial_mechanics.pdf',mime='application/pdf',key='wd_pdf')
            except Exception as exc:
                st.error(f'Could not generate wall design report: {exc}')
        st.divider()
        st.markdown('### 5 — Optional: one-way stiffness transfer')
        st.warning('**NOT building-wide nonlinear convergence.** This copies isolated wall trial secant stiffness into the existing LINEAR shear-spring model, after which you must rerun it. It does NOT handle stacked-wall flexural/rocking continuity, design resistance, new seismic forces or iterative equilibrium. Do not use transferred values as a verified building stiffness.')
        valid_transfer=(a is not None and good and isinstance(last,pd.DataFrame) and len(last)==len(w)
            and np.isfinite(last['Trial k secant (kN/m)'].to_numpy(float)).all()
            and (last['Trial k secant (kN/m)']>0).all()
            and not last['Drift review flag'].str.startswith('HIGH').any())
        if st.button('Copy trial k to multi-storey model (one-way)',disabled=not valid_transfer,key='wd_transfer_k'):
            lookup={(int(x['Storey']),str(x['Wall Name'])):float(x['Trial k secant (kN/m)']) for _,x in last.iterrows()}
            updated=w.copy()
            for i,x in updated.iterrows():
                updated.at[i,'k (kN/m)']=lookup[(int(x['Storey']),str(x['Wall Name']))]
            st.session_state.ms_walls=updated
            st.session_state.ms_revision=int(st.session_state.get('ms_revision',0))+1
            st.session_state.pop('ms_result',None)
            st.session_state.pop('wd_result',None)
            st.session_state.wd_result_is_current=False
            st.success('Trial stiffness values copied. Open Multi-storey periods & drift and press Run. The imported state has not been analyzed or declared converged.')
            st.rerun()
        if not valid_transfer:
            st.caption('Transfer is available only after a current, complete trial check with positive, finite secant stiffness for every segment. Set a nonzero trial shear for each wall and investigate any high trial-drift flags (research review threshold: 2%).')
