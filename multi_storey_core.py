"""Educational multi-storey rigid diaphragm shear-building model.

Each floor has (Ux, Uy, theta_z) at one FIXED global plan reference.
Vertical elements are independent storey-shear springs. This does NOT
represent continuous-wall bending/rocking, shearwall redesign, full seismic
code demand, P-delta, vertical irregularity checks or diaphragm flexibility.

Units: metres, kN, seconds, radians, kN/m; force vectors use kN, kN*m.
Mass uses kN*s**2/m and kg-equivalent polar mass inertia.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd

G = 9.80665
CASES = ("X +", "X -", "Y +", "Y -")
FLOOR_COLS = ("Storey", "Height (m)", "Weight (kN)", "Fx (kN)", "Fy (kN)", "Xcm (m)", "Ycm (m)")
WALL_COLS = ("Storey", "Wall Name", "Direction", "x (m)", "y (m)", "k (kN/m)")


def example_inputs(current_walls: pd.DataFrame, settings: dict[str, float], n: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Illustrative 6-storey prototype; floor loads are NOT NBCC-generated."""
    if not 1 <= n <= 12:
        raise ValueError("Storey count must be between 1 and 12.")
    floors = pd.DataFrame([{
        "Storey": i, "Height (m)": 2.8, "Weight (kN)": 1000.0,
        "Fx (kN)": round(15.0 + 10.0*i, 2),
        "Fy (kN)": round(12.0 + 8.0*i, 2),
        "Xcm (m)": float(settings["Xcm"]), "Ycm (m)": float(settings["Ycm"]),
    } for i in range(1, n+1)])
    source = current_walls.rename(columns={"Wall Name":"Wall Name"})
    walls = pd.concat([pd.DataFrame({
        "Storey": i, "Wall Name": source["Wall Name"].astype(str).to_numpy(),
        "Direction": source["Direction"].astype(str).to_numpy(),
        "x (m)": source["x (m)"].astype(float).to_numpy(),
        "y (m)": source["y (m)"].astype(float).to_numpy(),
        "k (kN/m)": source["k (kN/m)"].astype(float).to_numpy(),
    }) for i in range(1, n+1)],ignore_index=True)
    return floors, walls


def validate_inputs(floors: pd.DataFrame, walls: pd.DataFrame, Lx: float, Ly: float, acc: float) -> tuple[pd.DataFrame,pd.DataFrame]:
    if not np.isfinite([Lx,Ly,acc]).all() or min(Lx,Ly) <= 0 or not (0 <= acc <= .5):
        raise ValueError("Plan dimensions must be positive and accidental-eccentricity fraction between 0 and 0.5.")
    missing_f = set(FLOOR_COLS) - set(floors.columns)
    missing_w = set(WALL_COLS) - set(walls.columns)
    if missing_f or missing_w:
        raise ValueError(f"Missing floor columns: {sorted(missing_f)}; wall columns: {sorted(missing_w)}")
    f = floors.loc[:, FLOOR_COLS].copy().reset_index(drop=True)
    w = walls.loc[:, WALL_COLS].copy().reset_index(drop=True)
    if not 1 <= len(f) <= 12:
        raise ValueError("Model must have 1–12 floor levels.")
    if w.empty or len(w) > 600:
        raise ValueError("Specify at least one and at most 600 wall segments.")
    for df, cols in ((f, list(FLOOR_COLS)[0:1]+list(FLOOR_COLS)[1:]), (w, ["Storey","x (m)","y (m)","k (kN/m)"])):
        for col in cols:
            df[col] = pd.to_numeric(df[col],errors="raise")
            if not np.isfinite(df[col].to_numpy(dtype=float)).all():
                raise ValueError(f"Nonfinite values found in {col}.")
    for df in (f,w):
        if not np.all(df["Storey"].to_numpy(float) == df["Storey"].to_numpy(int)):
            raise ValueError("Storey values must be integers.")
        df["Storey"] = df["Storey"].astype(int)
    f=f.sort_values("Storey").reset_index(drop=True)
    w=w.sort_values(["Storey","Wall Name"]).reset_index(drop=True)
    if f["Storey"].tolist() != list(range(1,len(f)+1)):
        raise ValueError("Floor storey numbers must be consecutive, starting from 1 (bottom).")
    if not w["Storey"].isin(f["Storey"]).all():
        raise ValueError("A wall row references a nonexistent storey.")
    if (f["Height (m)"]<=0).any() or (f["Weight (kN)"]<=0).any():
        raise ValueError("Floor heights and seismic weights must be > 0.")
    if (w["k (kN/m)"]<=0).any():
        raise ValueError("Wall stiffness must be strictly positive.")
    if w["Wall Name"].isna().any() or (w["Wall Name"].astype(str).str.strip()=="").any():
        raise ValueError("Wall names are required.")
    if w.duplicated(["Storey","Wall Name"]).any():
        raise ValueError("Wall names must be unique within each storey.")
    if not w["Direction"].isin(["X","Y"]).all():
        raise ValueError("Each wall must resist X or Y.")
    for i in f["Storey"]:
        if not set(w.loc[w["Storey"]==i,"Direction"]) >= {"X","Y"}:
            raise ValueError(f"Storey {i}: include both X and Y resisting walls.")
    return f,w


def wall_vector(direction: str,x:float,y:float,x0:float,y0:float)->np.ndarray:
    # Positive theta_z moves point in x by -theta*(y-y0), in y by +theta*(x-x0).
    if direction=="X": return np.array([1.,0.,-(y-y0)])
    if direction=="Y": return np.array([0.,1.,(x-x0)])
    raise ValueError("Invalid wall direction")


def _matrices(f:pd.DataFrame,w:pd.DataFrame,k:np.ndarray,Lx:float,Ly:float)->tuple[np.ndarray,np.ndarray,list[np.ndarray]]:
    n=len(f); x0=Lx/2.;y0=Ly/2.; K=np.zeros((3*n,3*n)); M=np.zeros_like(K)
    H=[]
    storeys=w["Storey"].to_numpy(int)
    for i,r in f.iterrows():
        s=i+1; ks=np.zeros((3,3))
        for j in np.flatnonzero(storeys==s):
            a=wall_vector(w.at[j,"Direction"],float(w.at[j,"x (m)"]),float(w.at[j,"y (m)"]),x0,y0)
            ks += k[j]*np.outer(a,a)
        # A mechanism at an individual storey cannot be restrained by floors above.
        ev=np.linalg.eigvalsh(ks)
        if ev[0] <= max(ev[-1],1.)*1e-11:
            raise ValueError(f"Storey {s} has an unstable/near-singular X–Y–torsion restraint. Add non-collinear lateral elements.")
        H.append(ks)
        j=slice(3*i,3*i+3)
        K[j,j]+=ks
        if i:
            p=slice(3*(i-1),3*i)
            K[p,p]+=ks
            K[j,p]-=ks
            K[p,j]-=ks
        mass=float(r["Weight (kN)"])/G
        ex=float(r["Xcm (m)"])-x0; ey=float(r["Ycm (m)"])-y0
        ax=np.array([1.,0.,-ey]);ay=np.array([0.,1.,ex]);
        # Positive-mass rigid plate with rectangular, uniform in-plan mass distribution.
        Icg=mass*(Lx**2+Ly**2)/12.
        M[j,j]=mass*(np.outer(ax,ax)+np.outer(ay,ay))+np.diag([0.,0.,Icg])
    return K,M,H


def _static_force(f:pd.DataFrame,Lx:float,Ly:float,acc:float,case:str)->np.ndarray:
    n=len(f); loads=np.zeros(3*n); x0=Lx/2.; y0=Ly/2.
    direction=case[0];sign=1 if case.endswith("+") else -1
    for i,row in f.iterrows():
        F=float(row["Fx (kN)"] if direction=="X" else row["Fy (kN)"])
        if direction=="X":
            loads[3*i]=F
            loads[3*i+2]=-(float(row["Ycm (m)"])-y0+sign*acc*Ly)*F
        else:
            loads[3*i+1]=F
            loads[3*i+2]=(float(row["Xcm (m)"])-x0+sign*acc*Lx)*F
    return loads


def _modal(K:np.ndarray,M:np.ndarray)->tuple[pd.DataFrame, np.ndarray]:
    L=np.linalg.cholesky(M)
    A=np.linalg.solve(L,np.linalg.solve(L,K).T).T
    A=(A+A.T)/2
    vals,vecs=np.linalg.eigh(A)
    if vals[0]<=0 or not np.isfinite(vals).all():
        raise ValueError("Modal calculation failed: structure is not stable positive definite.")
    modes=np.linalg.solve(L.T,vecs)
    n=K.shape[0]//3
    r_x=np.tile([1.,0.,0.],n)
    r_y=np.tile([0.,1.,0.],n)
    totx=float(r_x@M@r_x);toty=float(r_y@M@r_y)
    rows=[]
    for j,lmbd in enumerate(vals):
        shape=modes[:,j]
        norm=float(shape@M@shape)
        ex=float((shape@M@r_x)**2/norm/totx)
        ey=float((shape@M@r_y)**2/norm/toty)
        rows.append({"Mode":j+1,"Period (s)":float(2*math.pi/math.sqrt(lmbd)),
                     "Frequency (Hz)":float(math.sqrt(lmbd)/(2*math.pi)),
                     "X effective mass (%)":100*ex,"Y effective mass (%)":100*ey})
    return pd.DataFrame(rows),modes


def _results_with_k(f:pd.DataFrame,w:pd.DataFrame,k:np.ndarray,Lx:float,Ly:float,acc:float)->dict[str,Any]:
    n=len(f);K,M,H=_matrices(f,w,k,Lx,Ly)
    modal,modes=_modal(K,M)
    height=f["Height (m)"].to_numpy(float)
    cumheight=np.cumsum(height)
    x0=Lx/2; y0=Ly/2
    positions=((0.,0.),(Lx,0.),(0.,Ly),(Lx,Ly))
    cases={};floor_rows=[];wall_rows=[]
    max_wall_shear=np.zeros(len(w))
    st=w["Storey"].to_numpy(int)
    for case in CASES:
        F=_static_force(f,Lx,Ly,acc,case)
        q=np.linalg.solve(K,F).reshape((n,3))
        r=(K@q.reshape(-1)-F)
        residual=float(np.linalg.norm(r,ord=np.inf))
        # Interpret u relative to fixed centre, not different CM systems floor-to-floor.
        for i in range(n):
            dq=q[i]-(q[i-1] if i>0 else np.zeros(3))
            corners=np.array([(-yy+y0,xx-x0) for xx,yy in positions])
            drifts_x=np.abs(dq[0]+dq[2]*corners[:,0])/height[i]
            drifts_y=np.abs(dq[1]+dq[2]*corners[:,1])/height[i]
            floor_rows.append({"Case":case,"Storey":i+1,"Elevation (m)":cumheight[i],
                 "Ux (mm)":q[i,0]*1000,"Uy (mm)":q[i,1]*1000,"Rotation (mrad)":q[i,2]*1000,
                 "ΔUx (mm)":dq[0]*1000,"ΔUy (mm)":dq[1]*1000,
                 "Max X drift (%)":100*float(drifts_x.max()),"Max Y drift (%)":100*float(drifts_y.max()),
                 "Max drift (%)":100*float(max(drifts_x.max(),drifts_y.max()))})
        for j,row in w.iterrows():
            s=int(row["Storey"])-1
            dq=q[s]-(q[s-1] if s else np.zeros(3))
            a=wall_vector(str(row["Direction"]),float(row["x (m)"]),float(row["y (m)"]),x0,y0)
            disp=float(a@dq)
            shear=float(k[j]*disp)
            max_wall_shear[j]=max(max_wall_shear[j],abs(shear))
            wall_rows.append({"Case":case,"Storey":s+1,"Wall Name":row["Wall Name"],
                "Direction":row["Direction"],"Wall shear (kN)":shear,"Drift at wall (mm)":disp*1000,
                "Wall drift (%)":100*abs(disp)/height[s],"k used (kN/m)":k[j]})
        # Static displacement Rayleigh check using actual full 3DOF mass and stiffness.
        x=q.reshape(-1)
        rq=2*math.pi*math.sqrt(float(x@M@x)/float(x@K@x)) if np.linalg.norm(x)>1e-12 else math.nan
        cases[case]={"loads":F,"q":q,"residual":residual,"rayleigh_s":rq}
    floor_df=pd.DataFrame(floor_rows)
    wall_df=pd.DataFrame(wall_rows)
    envelope=(wall_df.assign(absolute=lambda df: df["Wall shear (kN)"].abs())
              .sort_values("absolute",ascending=False).drop_duplicates(["Storey","Wall Name"])
              .sort_values(["Storey","Wall Name"]).reset_index(drop=True))
    envelope=envelope.rename(columns={"Case":"Governing case","absolute":"Envelope |V| (kN)"})
    # Reference-height approximation: deliberately not a code-derived period.
    h_ref=2*cumheight[-1]/3
    simple=[]
    for direction in "XY":
        cs=cases[f"{direction} +"];component=0 if direction=="X" else 1
        disp=abs(float(np.interp(h_ref,np.r_[0.,cumheight],np.r_[0.,cs["q"][:,component]])))
        floor_forces=f["Fx (kN)" if direction=="X" else "Fy (kN)"].to_numpy(float)
        # The APEGBC worked example uses V_D = SUM of all level lateral forces.
        # This is the supplied total/base shear, NOT the sum above 2H/3.
        Vref=abs(float(np.sum(floor_forces)))
        Kref=Vref/disp if disp>1e-15 else math.nan
        Wtotal=float(f["Weight (kN)"].sum())
        Tref=2*math.pi*math.sqrt(Wtotal/(G*Kref)) if Kref>0 else math.nan
        simple.append({"Direction":direction,"Reference height (m)":h_ref,
            "Translation at reference (mm)":disp*1000,"Total applied V_D (kN)":Vref,
            "Weight used (kN)":Wtotal,"Equivalent k (kN/m)":Kref,"Illustrative T_2/3 (s)":Tref})
    modal_x=modal.sort_values("X effective mass (%)",ascending=False).iloc[0]
    modal_y=modal.sort_values("Y effective mass (%)",ascending=False).iloc[0]
    return {"K":K,"M":M,"storey_K":H,"modes":modal,"mode_shapes":modes,"case_results":cases,
        "floors":floor_df,"wall_forces":wall_df,"envelope":envelope,
        "k":k.copy(),"max_wall_shear":max_wall_shear,"approx":pd.DataFrame(simple),
        "modal_x_s":float(modal_x["Period (s)"]),"modal_y_s":float(modal_y["Period (s)"]),
        "first_mode_s":float(modal.iloc[0]["Period (s)"]),
        "max_drift_pct":float(floor_df["Max drift (%)"].max()),
        "max_residual":max(c["residual"] for c in cases.values())}


def analyze_multistorey(floors:pd.DataFrame,walls:pd.DataFrame,Lx:float,Ly:float,acc:float=0.05,
    nonlinear:bool=False, exponent:float=1.15, reference_force:float=100., relaxation:float=0.5,
    tolerance:float=0.01, max_iterations:int=30, min_stiffness_ratio:float=0.25)->dict[str,Any]:
    """Returns global static/modal response. Optional approximate force-dependent secant law.

    The softening law k_sec=k_ref*max(V/Vref,1)**(1-p) has no physical
    calibration and is NOT a mechanics-based wood-wall stiffness calculation.
    Period is OUTPUT; seismic load vector is not recalculated from period.
    """
    f,w=validate_inputs(floors,walls,Lx,Ly,acc)
    if (not 1 <= exponent <= 2.0 or reference_force<=0 or not 0<relaxation<=1
        or not 1e-6<=tolerance<=0.20 or not 1<=max_iterations<=100
        or not 0 < min_stiffness_ratio <= 1):
        raise ValueError("Invalid secant iteration configuration.")
    kref=w["k (kN/m)"].to_numpy(float); k=kref.copy()
    records=[]; converged=not nonlinear; previous=None
    iterations= max_iterations if nonlinear else 1
    for it in range(1,iterations+1):
        res=_results_with_k(f,w,k,Lx,Ly,acc)
        if not nonlinear:
            records.append({"Iteration":it,"T1 (s)":res["first_mode_s"],"Max drift (%)":res["max_drift_pct"],
                "Max |V| (kN)":float(res["max_wall_shear"].max()),"k change (%)":0.,
                "Period change (%)":0.,"Drift change (%)":0.,"Force change (%)":0.})
            break
        v=res["max_wall_shear"]
        target=np.maximum(kref*min_stiffness_ratio,
           kref * np.maximum(v/reference_force,1.0)**(1.0-exponent))
        updated=(1-relaxation)*k+relaxation*target
        relk=float(np.max(np.abs(updated-k)/kref))
        if previous is not None:
            dT=abs(res["first_mode_s"]-previous["first_mode_s"])/max(res["first_mode_s"],1e-12)
            dd=abs(res["max_drift_pct"]-previous["max_drift_pct"])/max(res["max_drift_pct"],1e-7)
            dF=float(np.max(np.abs(v-previous["max_wall_shear"])/np.maximum(v,1.0)))
        else: dT=dd=dF=math.inf
        records.append({"Iteration":it,"T1 (s)":res["first_mode_s"],"Max drift (%)":res["max_drift_pct"],
            "Max |V| (kN)":float(v.max()),"k change (%)":100*relk,
            "Period change (%)":100*dT,"Drift change (%)":100*dd,"Force change (%)":100*dF})
        if previous is not None and max(relk,dT,dd,dF) <= tolerance:
            # Re-solve at the target accepted stiffness; output must correspond to final k.
            k=updated
            res=_results_with_k(f,w,k,Lx,Ly,acc)
            converged=True
            break
        k=updated
        previous=res
    if nonlinear and not converged:
        # Solve one last time using the final updated stiffness, not the prior iterate.
        res=_results_with_k(f,w,k,Lx,Ly,acc)
    res.update({"history":pd.DataFrame(records),"converged":converged,"nonlinear":nonlinear,
        "input_floors":f,"input_walls":w,"settings":{"Lx":Lx,"Ly":Ly,"acc":acc,"exponent":exponent,
        "reference_force":reference_force,"relaxation":relaxation,"tolerance":tolerance,
        "max_iterations":max_iterations,"min_stiffness_ratio":min_stiffness_ratio},
        "stiffness_ratio":k/kref})
    return res
