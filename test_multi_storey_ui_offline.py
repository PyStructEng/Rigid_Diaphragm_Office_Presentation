"""Offline Streamlit UI smoke test without installing Streamlit (not a live browser test)."""
import sys
import types
import contextlib
from unittest.mock import patch

class State(dict):
    def __getattr__(self,k):
        try:return self[k]
        except KeyError:raise AttributeError(k)
    def __setattr__(self,k,v):self[k]=v

class Stub:
    def __init__(self):
        self.session_state=State()
        self.column_config=types.SimpleNamespace(NumberColumn=lambda *a,**k:None,
            TextColumn=lambda *a,**k:None,SelectboxColumn=lambda *a,**k:None)
        self.latexes=[];self.downloads=[]
    def __getattr__(self,name):
        if name=='columns':return lambda arg:[self]* (len(arg) if isinstance(arg,(tuple,list)) else arg)
        if name=='tabs':return lambda choices:[contextlib.nullcontext() for c in choices]
        if name=='expander':return lambda *a,**kw:contextlib.nullcontext()
        if name=='number_input':return lambda *a,**kw:kw.get('value',0)
        if name=='button':return lambda *a,**kw:kw.get('key')=='ms_run'
        if name=='checkbox':return lambda *a,**kw:kw.get('value',False)
        if name=='data_editor':return lambda data,**kw:data.copy()
        if name=='selectbox':return lambda label,options,**kw:list(options)[0]
        if name=='latex':return lambda val:self.latexes.append(val)
        if name=='download_button':return lambda *a,**kw:self.downloads.append((a,kw))
        if name=='spinner':return lambda *a,**kw:contextlib.nullcontext()
        if name in {'header','caption','warning','markdown','info','metric','success','error','dataframe','line_chart','rerun'}:
            return lambda *a,**kw:None
        raise AttributeError(name)

if __name__=='__main__':
    fake=Stub()
    with patch.dict(sys.modules,{'streamlit':fake}):
        from multi_storey_ui import render_multi_storey_tab
        from rigid_diaphragm_core import default_walls
        render_multi_storey_tab(default_walls(),{'Lx':20.,'Ly':12.,'Xcm':10.,'Ycm':6.,'acc':0.05})
        r=fake.session_state['ms_result']
        assert len(r['floors'])==24
        assert len(fake.latexes)==6
        assert len(fake.downloads)>=4
        assert len(fake.session_state['ms_design_history'])==1
        assert any(kw.get('mime')=='application/pdf' for _a,kw in fake.downloads)
        print('PASS: Offline Streamlit UI render, 6 latex expressions, PDF and CSV downloads, run history, 6-storey solve')
