"""Regression tests: run python -m unittest test_multi_storey -v"""
import io
import math
import unittest
import numpy as np
import pandas as pd

from rigid_diaphragm_core import default_walls
from multi_storey_core import analyze_multistorey, example_inputs, _static_force, G
from multi_storey_report import make_multistorey_pdf


def sym_model(n=1):
    f=pd.DataFrame([{"Storey":i,"Height (m)":3.,"Weight (kN)":G*10.,
        "Fx (kN)":100.,"Fy (kN)":80.,"Xcm (m)":5.,"Ycm (m)":5.} for i in range(1,n+1)])
    walls=[]
    for i in range(1,n+1):
        walls.extend([{"Storey":i,"Wall Name":"X1","Direction":"X","x (m)":2.,"y (m)":0.,"k (kN/m)":500.},
         {"Storey":i,"Wall Name":"X2","Direction":"X","x (m)":8.,"y (m)":10.,"k (kN/m)":500.},
         {"Storey":i,"Wall Name":"Y1","Direction":"Y","x (m)":0.,"y (m)":5.,"k (kN/m)":800.},
         {"Storey":i,"Wall Name":"Y2","Direction":"Y","x (m)":10.,"y (m)":5.,"k (kN/m)":800.}])
    return f,pd.DataFrame(walls)


class MultiStoreyTests(unittest.TestCase):
    def test_one_level_closed_form(self):
        f,w=sym_model()
        res=analyze_multistorey(f,w,10,10,acc=0)
        self.assertAlmostEqual(res['modal_x_s'],2*math.pi*math.sqrt(10/1000),places=9)
        self.assertAlmostEqual(res['modal_y_s'],2*math.pi*math.sqrt(10/1600),places=9)
        np.testing.assert_allclose(res['case_results']['X +']['q'][0],np.array([.1,0,0]),atol=1e-12)
        shear=res['wall_forces'].query('Case == "X +"')
        self.assertAlmostEqual(shear['Wall shear (kN)'].sum(),100.,places=10)
        self.assertAlmostEqual(float(shear.loc[shear['Wall Name']=='X1','Wall shear (kN)'].iloc[0]),50.,places=9)
        self.assertAlmostEqual(res['max_residual'],0,delta=1e-9)

    def test_two_storeys_series_shear_modes(self):
        f,w=sym_model(2)
        r=analyze_multistorey(f,w,10,10,acc=0)
        theoretical=2*math.pi/math.sqrt((1000/10)*(3-math.sqrt(5))/2)
        self.assertAlmostEqual(r['modal_x_s'],theoretical,places=9)
        self.assertEqual(len(r['modes']),6)
        self.assertAlmostEqual(r['modes']['X effective mass (%)'].sum(),100.,places=7)
        self.assertAlmostEqual(r['modes']['Y effective mass (%)'].sum(),100.,places=7)

    def test_four_cases_equilibrium_per_storey(self):
        f,w=sym_model(6)
        f.loc[0,'Xcm (m)']=6.0
        f.loc[1,'Ycm (m)']=4.25
        for sign in (0.0,.05,.1):
            r=analyze_multistorey(f,w,10,10,acc=sign)
            for case in ['X +','X -','Y +','Y -']:
                loads=_static_force(f,10,10,sign,case).reshape(-1,3)
                for floor in range(1,7):
                    force=r['wall_forces'].query('Case == @case and Storey == @floor')
                    rows=w.query('Storey == @floor')
                    # Free body of floor and everything above: story shear resultants.
                    R=np.zeros(3)
                    for j,entry in force.iterrows():
                        wl=rows[rows['Wall Name']==entry['Wall Name']].iloc[0]
                        V=entry['Wall shear (kN)']
                        if wl['Direction']=='X': R+=np.array([V,0,-(wl['y (m)']-5.)*V])
                        else: R+=np.array([0,V,(wl['x (m)']-5.)*V])
                    np.testing.assert_allclose(R,loads[floor-1:].sum(axis=0),rtol=1e-9,atol=1e-8)
            self.assertLess(r['max_residual'],1e-8)

    def test_torsion_sign_reverses(self):
        f,w=sym_model()
        r=analyze_multistorey(f,w,10,10,acc=.05)
        theta_plus=r['case_results']['X +']['q'][0,2]
        theta_minus=r['case_results']['X -']['q'][0,2]
        self.assertAlmostEqual(theta_plus,-theta_minus,places=12)
        self.assertNotEqual(theta_plus,0)
        self.assertGreater(r['max_drift_pct'],0)

    def test_zero_loading(self):
        f,w=sym_model(3)
        f[['Fx (kN)','Fy (kN)']]=0.
        r=analyze_multistorey(f,w,10,10,acc=.05)
        self.assertAlmostEqual(r['max_drift_pct'],0.)
        self.assertAlmostEqual(r['envelope']['Envelope |V| (kN)'].sum(),0)
        self.assertTrue(math.isnan(r['case_results']['X +']['rayleigh_s']))
        self.assertTrue(math.isfinite(r['first_mode_s']))

    def test_stiffness_softening_no_change_for_linear(self):
        f,w=sym_model(4)
        elastic=analyze_multistorey(f,w,10,10)
        nonlin=analyze_multistorey(f,w,10,10,nonlinear=True,exponent=1.,reference_force=10.,max_iterations=20)
        self.assertTrue(nonlin['converged'])
        np.testing.assert_allclose(elastic['k'],nonlin['k'],rtol=0,atol=1e-9)
        self.assertAlmostEqual(elastic['first_mode_s'],nonlin['first_mode_s'],places=9)

    def test_nonlinear_softening_converges_and_re_equilibrates(self):
        f,w=sym_model(4)
        r=analyze_multistorey(f,w,10,10,nonlinear=True,exponent=1.3,reference_force=15.,relaxation=.5,tolerance=.005,max_iterations=60)
        self.assertTrue(r['converged'])
        self.assertTrue((r['stiffness_ratio']<=1+1e-12).all())
        self.assertTrue((r['stiffness_ratio']>=.25-1e-12).all())
        self.assertTrue((r['stiffness_ratio']<1).any())
        self.assertLess(r['max_residual'],1e-8)

    def test_input_rejections(self):
        f,w=sym_model(2)
        for inval in [0,-10]:
            ww=w.copy();ww.loc[0,'k (kN/m)']=inval
            with self.assertRaises(ValueError):analyze_multistorey(f,ww,10,10)
        ww=w[w['Direction']=='X']
        with self.assertRaises(ValueError):analyze_multistorey(f,ww,10,10)
        ww=w.copy(); ww.loc[ww['Wall Name']=='X2','y (m)']=0;ww.loc[ww['Wall Name']=='Y2','x (m)']=0
        with self.assertRaisesRegex(ValueError,'unstable'):analyze_multistorey(f,ww,10,10)
        ff=f.copy();ff.loc[0,'Weight (kN)']=-10
        with self.assertRaises(ValueError):analyze_multistorey(ff,w,10,10)

    def test_pdf_and_example_six_storeys(self):
        s={'Lx':20.,'Ly':12.,'Xcm':10.,'Ycm':6.}
        f,w=example_inputs(default_walls(),s,n=6)
        r=analyze_multistorey(f,w,20,12,.05)
        self.assertEqual(len(r['modes']),18)
        self.assertEqual(len(r['floors']),24)
        self.assertEqual(len(r['wall_forces']),4*len(w))
        self.assertEqual(len(r['envelope']),len(w))
        self.assertTrue(np.allclose(r['approx']['Reference height (m)'],11.2,atol=1e-12))
        data=make_multistorey_pdf(r)
        self.assertTrue(data.startswith(b'%PDF'))
        self.assertGreater(len(data),10000)

if __name__=='__main__': unittest.main()
