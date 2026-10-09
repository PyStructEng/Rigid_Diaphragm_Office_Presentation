"""Run: python -m unittest test_wall_design -v"""
import io
import math
import unittest
import numpy as np
import pandas as pd

from rigid_diaphragm_core import default_walls
from multi_storey_core import example_inputs, analyze_multistorey
from wall_design_core import (ASSEMBLY_COLS, default_assemblies, default_wall_schedule,
    validate_assemblies, validate_schedule, evaluate_schedule, wall_trial_response,
    sync_schedule, strength_review, export_design_json, import_design_json)
from wall_design_report import make_wall_design_pdf


class TestWallDesign(unittest.TestCase):
    def setUp(self):
        self.f, self.w = example_inputs(default_walls(), {'Xcm':10.0,'Ycm':6.0},n=2)
        self.a=default_assemblies()
        self.d=default_wall_schedule(self.w)

    def test_default_assembly_and_global_schedule(self):
        self.assertEqual(len(validate_assemblies(self.a)),3)
        self.assertEqual(len(validate_schedule(self.d,self.a,self.w,self.f)),len(self.w))

    def test_stiffness_units_and_components(self):
        r=evaluate_schedule(self.d,self.a,self.w,self.f)
        for _,z in r.iterrows():
            self.assertAlmostEqual(z['Trial k secant (kN/m)'],1000*z['Trial |V| (kN)']/z['Trial deflection (mm)'],places=8)
            self.assertAlmostEqual(z['Trial deflection (mm)'],z['Bending (mm)']+z['Panel shear (mm)']+z['Nail slip contribution (mm)']+z['Anchorage contribution (mm)'],places=8)
        self.assertTrue(r['Trial k secant (kN/m)'].gt(0).all())

    def test_tighter_edge_nailing_increases_trial_stiffness(self):
        a=self.a.copy()
        baseline=evaluate_schedule(self.d,a,self.w,self.f).iloc[0]
        a.loc[a['Assembly ID']=='SW-OSB-1','Nail edge spacing (mm)']=75.
        optimized=evaluate_schedule(self.d,a,self.w,self.f).iloc[0]
        self.assertLess(optimized['Nail slip (mm)'],baseline['Nail slip (mm)'])
        self.assertGreater(optimized['Trial k secant (kN/m)'],baseline['Trial k secant (kN/m)'])

    def test_field_spacing_and_nail_length_are_record_only(self):
        b=evaluate_schedule(self.d,self.a,self.w,self.f).iloc[0]
        a=self.a.copy()
        a.loc[a['Assembly ID']=='SW-OSB-1','Nail field spacing (mm)']=100.
        a.loc[a['Assembly ID']=='SW-OSB-1','Nail length (mm)']=75.
        c=evaluate_schedule(self.d,a,self.w,self.f).iloc[0]
        self.assertAlmostEqual(b['Trial deflection (mm)'],c['Trial deflection (mm)'],places=12)

    def test_different_assemblies_independent(self):
        d=self.d.copy()
        d.loc[(d['Storey']==1)&(d['Wall Name']=='X1'),'Assembly ID']='SW-DFP-1'
        r=evaluate_schedule(d,self.a,self.w,self.f)
        x=r.query('Storey == 1 and `Wall Name` == "X1"').iloc[0]['Trial k secant (kN/m)']
        y=r.query('Storey == 1 and `Wall Name` == "X2"').iloc[0]['Trial k secant (kN/m)']
        self.assertNotAlmostEqual(x,y)

    def test_zero_and_negative_trial_shear(self):
        d=self.d.copy()
        d.loc[d.index[0],'Trial shear (kN)']=0.
        r=evaluate_schedule(d,self.a,self.w,self.f)
        self.assertTrue(math.isnan(r.iloc[0]['Trial k secant (kN/m)']))
        self.assertTrue(math.isnan(r.iloc[0]['Trial deflection (mm)']))
        row=self.d.iloc[0]
        aa=self.a.iloc[0]
        one=wall_trial_response(row,aa,2.8,45.)
        two=wall_trial_response(row,aa,2.8,-45.)
        self.assertAlmostEqual(one['Trial k secant (kN/m)'],two['Trial k secant (kN/m)'],places=12)

    def test_factored_resistance_requires_verified_values(self):
        row=self.d.iloc[0].copy()
        row['Factored shear demand (kN)']=100.
        self.assertEqual(strength_review(row)['Shear status'],'NOT CHECKED')
        row['Verified shear resistance (kN)']=120.
        self.assertEqual(strength_review(row)['Shear status'],'Within supplied capacity')
        row['Verified shear resistance (kN)']=80.
        self.assertIn('REVIEW',strength_review(row)['Shear status'])
        self.assertEqual(strength_review(row)['Hold-down tension status'],'NOT CHECKED')

    def test_incorrect_missing_and_duplicate_assembly_rejected(self):
        a=self.a.copy()
        a.loc[a.index[1],'Assembly ID']=a.iloc[0]['Assembly ID']
        with self.assertRaisesRegex(ValueError,'unique'):validate_assemblies(a)
        d=self.d.copy();d.loc[d.index[0],'Assembly ID']='SOMETHING-UNKNOWN'
        with self.assertRaisesRegex(ValueError,'Undefined'):validate_schedule(d,self.a,self.w,self.f)

    def test_schedule_changes_and_id_sync(self):
        changed=self.w.iloc[1:].copy()
        with self.assertRaisesRegex(ValueError,'does not match'):validate_schedule(self.d,self.a,changed,self.f)
        d=self.d.copy();d.at[0,'Wall length (m)']=3.2
        synced=sync_schedule(d,changed)
        self.assertEqual(len(synced),len(changed))
        self.assertFalse(((synced['Storey']==self.w.iloc[0]['Storey'])&(synced['Wall Name']==self.w.iloc[0]['Wall Name'])).any())

    def test_case_requires_one_simultaneous_load_vector(self):
        global_response=analyze_multistorey(self.f,self.w,20.,12.,.05)
        forces=global_response['wall_forces'].query('Case == "X +"')
        r=evaluate_schedule(self.d,self.a,self.w,self.f,forces)
        self.assertEqual(len(r),len(self.w))
        self.assertTrue(np.isfinite(r['Trial deflection (mm)']).all())
        bad=global_response['wall_forces']
        with self.assertRaisesRegex(ValueError,'one simultaneous'):evaluate_schedule(self.d,self.a,self.w,self.f,bad)

    def test_roundtrip_json_validation(self):
        raw=export_design_json(self.a,self.d)
        aa,dd=import_design_json(raw,self.w,self.f)
        self.assertEqual(len(aa),3)
        self.assertEqual(len(dd),len(self.w))
        with self.assertRaises(ValueError):import_design_json(b'{"schema":"wrong"}',self.w,self.f)
        with self.assertRaises(ValueError):import_design_json(raw,self.w.iloc[:-1],self.f)

    def test_pdf_contains_all_segments(self):
        r=evaluate_schedule(self.d,self.a,self.w,self.f)
        raw=make_wall_design_pdf(self.a,self.d,r,'Manual trial shear')
        self.assertTrue(raw.startswith(b'%PDF'))
        self.assertGreater(len(raw),4000)

if __name__=='__main__':unittest.main()
