"""Run AFTER pip install -r requirements.txt. Uses REAL Handcalcs / ForAllPeople."""
import math
from rigid_diaphragm_core import fpinnovations_preset, default_walls, analyze_model
from engineering_calculations import audit_units, equation_by_sections, wood_secant_trace
from diaphragm_calculation_sheet import build_calculation_sections, make_calculation_pdf


def check(label, settings, walls):
    result=analyze_model(walls, settings['Lx'], settings['Ly'], settings['Xcm'],
                         settings['Ycm'], settings['Fx'], settings['Fy'], settings['acc'])
    eq=equation_by_sections(result,settings)
    audit=audit_units(result,settings)
    failures=audit.loc[~audit['Pass']]
    assert failures.empty, failures.to_string(index=False)
    secs=build_calculation_sections(walls,settings,result)
    for i,lines in eq.items():secs[i-1].equations.extend(lines)
    output=make_calculation_pdf(secs)
    assert output.startswith(b'%PDF') and len(output)>10000
    print(f'{label}: {len(audit)} dimensions/magnitude checks PASS; {sum(len(v) for v in eq.values())} equations PASS; PDF PASS ({len(output)} bytes)')


if __name__=='__main__':
    s,w=fpinnovations_preset();check('FPInnovations',s,w)
    w=default_walls()
    s={'Lx':20.,'Ly':12.,'Xcm':10.,'Ycm':6.,'Fx':100.,'Fy':100.,'acc':0.05}
    check('Symmetric',s,w)
    w.loc[w['Wall Name']=='X1','k (kN/m)']*=0.7
    w.loc[w['Wall Name']=='Y2','k (kN/m)']*=1.4
    w.loc[w['Wall Name']=='X1','Local X Force (kN)']=11.
    w.loc[w['Wall Name']=='Y1','Local Y Force (kN)']=-8.
    check('Asymmetric with local forces',s,w)
    s['Fx']=-150.;s['Fy']=-30.;check('Signed negative loads',s,w)
    s['Fx']=0.;s['Fy']=0.;s['acc']=0.;check('Zero load',s,w)
    _,k=wood_secant_trace(50.,12.,50./.012)
    assert math.isclose(k,50./.012)
    print('Wood-wall Handcalcs / ForAllPeople stiffness trace PASS')
