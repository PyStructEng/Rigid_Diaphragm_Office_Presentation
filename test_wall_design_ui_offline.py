"""Simulated UI integration check. Does not test installed Streamlit/browser."""
import sys
import types
import contextlib
from unittest.mock import patch
from test_multi_storey_ui_offline import Stub
from rigid_diaphragm_core import default_walls

class WallStub(Stub):
    def __getattr__(self,name):
        if name in ('form','expander'):
            return lambda *a,**kw:contextlib.nullcontext()
        if name=='form_submit_button':
            return lambda *a,**kw:False
        if name=='text_input':
            return lambda *a,**kw:kw.get('value','')
        if name=='file_uploader':
            return lambda *a,**kw:None
        if name=='divider':
            return lambda *a,**kw:None
        if name=='bar_chart':
            return lambda *a,**kw:None
        if name=='button':
            return lambda *a,**kw:kw.get('key')=='wd_eval'
        return super().__getattr__(name)

if __name__=='__main__':
    fake=WallStub()
    with patch.dict(sys.modules,{'streamlit':fake}):
        from wall_design_ui import render_wall_design_studio
        render_wall_design_studio(default_walls(),{'Lx':20.,'Ly':12.,'Xcm':10.,'Ycm':6.,'acc':0.05})
        assert len(fake.session_state.wd_result)==len(fake.session_state.ms_walls)
        assert fake.session_state.wd_result_is_current
        assert sum(kw.get('mime')=='application/pdf' for _,kw in fake.downloads)==1
        assert sum(kw.get('mime')=='application/json' for _,kw in fake.downloads)==1
        render_wall_design_studio(default_walls(),{'Lx':20.,'Ly':12.,'Xcm':10.,'Ycm':6.,'acc':0.05})
        assert fake.session_state.wd_result_is_current
        print('PASS: wall design studio offline UI, individual wall form, 6-storey trial evaluation, PDF & JSON exports')
