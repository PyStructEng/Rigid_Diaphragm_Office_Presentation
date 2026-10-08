"""Run directly with: python test_calculation_sheet.py"""
import math
from rigid_diaphragm_core import analyze_model, default_walls, fpinnovations_preset, run_self_tests
from wood_shearwall_mechanics import run_mechanics_self_tests
from diaphragm_calculation_sheet import build_calculation_sections, make_calculation_pdf


def run_case(walls, s, label):
    result = analyze_model(walls, s['Lx'],s['Ly'],s['Xcm'],s['Ycm'],s['Fx'],s['Fy'],s['acc'])
    sections = build_calculation_sections(walls,s,result)
    assert len(sections)==10
    p=result['properties']
    contrib=result['wall_properties']
    assert math.isclose(p['J'],float(contrib['ky*xbar^2 (kN·m)'].sum()+contrib['kx*ybar^2 (kN·m)'].sum()),abs_tol=1e-9)
    for d in ('X','Y'):
        cases=result['x_cases'] if d=='X' else result['y_cases']
        assert len(cases)==2*len(walls)
        for sign in ('−','+'):
            group=cases[cases['Accidental Sign']==sign]
            assert len(group)==len(walls)
            for _, row in group.iterrows():
                if row['Direction']=='X':
                    expected_local = row['Local wall force (kN)'] if d=='X' else 0
                    assert math.isclose(float(row['Local wall force (kN)']),expected_local,abs_tol=1e-9)
                    expected_v=row['Vx diaphragm (kN)']+row['Local wall force (kN)']
                else:
                    expected_v=row['Vy diaphragm (kN)']+row['Local wall force (kN)']
                assert math.isclose(float(row['Wall-parallel V (kN)']),expected_v,abs_tol=1e-9)
    pdf=make_calculation_pdf(sections)
    assert pdf.startswith(b'%PDF') and len(pdf)>10000
    print(f'PASS {label}: {len(sections)} sections, 4 cases, PDF {len(pdf)} bytes')
    return pdf

if __name__=='__main__':
    tests=run_self_tests()
    print('Rigid engine:',tests['Pass'].sum(),'/',len(tests))
    assert tests['Pass'].all()
    t=run_mechanics_self_tests()
    print('Wood mechanics:',t['Pass'].sum(),'/',len(t))
    assert t['Pass'].all()
    s,w=fpinnovations_preset()
    pdf=run_case(w,s,'FPInnovations')
    open('/mnt/data/rigid_diaphragm_sample_FPI_calculations.pdf','wb').write(pdf)
    w=default_walls()
    s={'Lx':20.0,'Ly':12.0,'Xcm':10.0,'Ycm':6.0,'Fx':100.0,'Fy':100.0,'acc':0.05}
    run_case(w,s,'symmetric')
    w.loc[w['Wall Name']=='X1','k (kN/m)'] *= .7
    w.loc[w['Wall Name']=='Y2','k (kN/m)'] *= 1.4
    w.loc[w['Wall Name']=='X1','Local X Force (kN)'] = 11
    w.loc[w['Wall Name']=='Y1','Local Y Force (kN)'] = -8
    run_case(w,s,'asymmetric + local forces')
    s['Fx']=-150.0; s['Fy']=-30.0; run_case(w,s,'signed negative loads')
    s['Fx']=0.0; s['Fy']=0.0; s['acc']=0.0; run_case(w,s,'zero loads')
