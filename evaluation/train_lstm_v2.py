"""Leakage-safe temporal model comparison for entrance-behaviour classification.

Writes only beneath results/evaluation/lstm_v2.  Run ``prepare`` first, then
``screen``/``tune`` (CV only), then exactly once ``final``.  The latter refuses
to tune and evaluates the frozen choice on VL/VS and the internal holdout.
"""
from __future__ import annotations

import argparse, hashlib, json, random, subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from make_holdout_test_v2 import LABEL_MAP, add_group_id

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/evaluation/lstm_v2"
WINDOW, STEP, T = 300, 10, 30
POSE = ["hand_motion", "body_motion", "arm_extension", "upper_body_angle"]
TRACK = ["person_count", "bbox_center_x", "bbox_center_y", "bbox_area", "bbox_center_delta", "avg_speed", "max_speed", "mean_confidence"]
SEQ_COLUMNS = POSE + TRACK
SUMMARY_COLUMNS = [
    "has_tracking", "tracked_person_count", "tracking_frame_count_mean", "tracking_frame_count_max", "tracking_frame_count_std",
    "move_distance_mean", "move_distance_max", "move_distance_std", "avg_speed_mean", "avg_speed_max", "avg_speed_std",
    "max_speed_mean", "max_speed_max", "max_speed_std", "min_speed_mean", "min_speed_max", "min_speed_std",
    "std_speed_mean", "std_speed_max", "std_speed_std", "movement_range_mean", "movement_range_max", "movement_range_std",
    "trajectory_variance_mean", "trajectory_variance_max", "trajectory_variance_std", "has_pose", "pose_frame_count",
    "hand_motion_mean", "hand_motion_max", "hand_motion_std", "body_motion_mean", "body_motion_max", "body_motion_std",
    "arm_extension_mean", "arm_extension_max", "arm_extension_std", "upper_body_angle_mean", "upper_body_angle_max", "upper_body_angle_std",
]

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def csv(path:Path)->pd.DataFrame:
    try: return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError): return pd.DataFrame()

def summary_tracking(frame:pd.DataFrame)->dict[str,float]:
    required={"frame","track_id","x1","y1","x2","y2"}
    if not required.issubset(frame): return {"has_tracking":0.,"tracked_person_count":0., **{x:0. for x in SUMMARY_COLUMNS[2:26]}}
    rows=[]
    for _, g in frame.groupby("track_id"):
        g=g.sort_values("frame")
        if len(g)<10: continue
        x=(g.x1.to_numpy()+g.x2.to_numpy())/2; y=(g.y1.to_numpy()+g.y2.to_numpy())/2
        gap=np.diff(g.frame.to_numpy()); dist=np.hypot(np.diff(x),np.diff(y)); speed=dist[np.nonzero(gap)[0]]/gap[gap>0]
        rows.append(dict(tracking_frame_count=len(g),move_distance=dist.sum(),avg_speed=speed.mean() if len(speed) else 0.,
            max_speed=speed.max() if len(speed) else 0.,min_speed=speed.min() if len(speed) else 0.,std_speed=speed.std() if len(speed) else 0.,
            movement_range=(x.max()-x.min())+(y.max()-y.min()),trajectory_variance=np.var(x)+np.var(y)))
    if not rows: return {"has_tracking":0.,"tracked_person_count":0., **{x:0. for x in SUMMARY_COLUMNS[2:26]}}
    values=pd.DataFrame(rows); result={"has_tracking":1.,"tracked_person_count":float(len(values))}
    for c in values: result.update({f"{c}_mean":values[c].mean(),f"{c}_max":values[c].max(),f"{c}_std":values[c].std(ddof=0)})
    return result

def summary_pose(frame:pd.DataFrame)->dict[str,float]:
    required={"frame",*POSE}
    if not required.issubset(frame): return {"has_pose":0.,"pose_frame_count":0., **{x:0. for x in SUMMARY_COLUMNS[28:]}}
    result={"has_pose":float(not frame.empty),"pose_frame_count":float(len(frame))}
    for c in POSE:
        x=frame[c].to_numpy(); result.update({f"{c}_mean":x.mean() if len(x) else 0.,f"{c}_max":x.max() if len(x) else 0.,f"{c}_std":x.std() if len(x) else 0.})
    return result

def sequence(pose:pd.DataFrame, tracking:pd.DataFrame, start:int, end:int)->tuple[np.ndarray,np.ndarray]:
    """Fixed frame-aligned slots. Missing pose is zero, explicitly marked by mask."""
    values=np.zeros((T,len(SEQ_COLUMNS)),np.float32); mask=np.zeros((T,1),np.float32)
    p=pose[(pose.frame>=start)&(pose.frame<=end)] if "frame" in pose else pose
    tr=tracking[(tracking.frame>=start)&(tracking.frame<=end)] if "frame" in tracking else tracking
    previous:dict[int,tuple[float,float,int]]={}
    for slot in range(T):
        lo, hi=start+slot*STEP, min(start+(slot+1)*STEP-1,end)
        ps=p[(p.frame>=lo)&(p.frame<=hi)] if not p.empty else p
        ts=tr[(tr.frame>=lo)&(tr.frame<=hi)] if not tr.empty else tr
        if not ps.empty:
            values[slot,:len(POSE)]=ps[POSE].mean().to_numpy(np.float32); mask[slot,0]=1.
        if ts.empty or not {"track_id","x1","y1","x2","y2"}.issubset(ts): continue
        centers=[]; speeds=[]
        for ident,g in ts.groupby("track_id"):
            g=g.sort_values("frame"); cx=((g.x1+g.x2)/2).to_numpy(); cy=((g.y1+g.y2)/2).to_numpy(); fr=g.frame.to_numpy()
            centers.extend(zip(cx,cy));
            px,py,pf=previous.get(int(ident),(cx[0],cy[0],fr[0]))
            d=np.hypot(np.diff(np.r_[px,cx]),np.diff(np.r_[py,cy])); gap=np.diff(np.r_[pf,fr]); speeds.extend((d[gap>0]/gap[gap>0]).tolist())
            previous[int(ident)]=(cx[-1],cy[-1],fr[-1])
        area=((ts.x2-ts.x1)*(ts.y2-ts.y1)).to_numpy(); mean_center=np.mean(centers,axis=0)
        delta=np.mean(speeds) if speeds else 0.
        values[slot,len(POSE):]=[ts.track_id.nunique(),mean_center[0],mean_center[1],area.mean(),delta,np.mean(speeds) if speeds else 0.,np.max(speeds) if speeds else 0.,ts.confidence.mean() if "confidence" in ts else 0.]
    return values,mask

def build(annotation_path:Path, split:str)->dict[str,Any]:
    ann=add_group_id(csv(annotation_path)); rows=[]; xs=[]; masks=[]; static=[]
    for video_path, blocks in ann.groupby("video_path",sort=False):
        stem=Path(video_path).stem; pose=csv(ROOT/f"results/pose/{split}/{stem}.csv"); track=csv(ROOT/f"results/tracking/{split}/{stem}.csv")
        for _,r in blocks.iterrows():
            s,e=int(r.start_frame),int(r.end_frame)
            for ws in range(s,e-WINDOW+2,WINDOW):
                we=ws+WINDOW-1; p=pose[(pose.frame>=ws)&(pose.frame<=we)] if "frame" in pose else pose; q=track[(track.frame>=ws)&(track.frame<=we)] if "frame" in track else track
                # Passing the already window-cropped frames is important: a
                # video can be thousands of frames long, while this function
                # only needs the current 300-frame window.
                x,m=sequence(p,q,ws,we); z={**summary_tracking(q),**summary_pose(p)}
                xs.append(x); masks.append(m); static.append([z.get(c,0.) for c in SUMMARY_COLUMNS]); rows.append({"video":r.video,"group_id":r.group_id,"label":r.label,"start_frame":ws,"end_frame":we})
    return {"X":np.asarray(xs,np.float32),"mask":np.asarray(masks,np.float32),"static":np.asarray(static,np.float32),"meta":pd.DataFrame(rows)}

def save_data(name:str,data:dict):
    np.savez_compressed(OUT/f"{name}.npz",X=data["X"],mask=data["mask"],static=data["static"]); data["meta"].to_csv(OUT/f"{name}_meta.csv",index=False,encoding="utf-8-sig")
def load_data(name:str)->dict:
    z=np.load(OUT/f"{name}.npz"); return {"X":z["X"],"mask":z["mask"],"static":z["static"],"meta":csv(OUT/f"{name}_meta.csv")}

class Temporal(nn.Module):
    def __init__(self,kind:str,dim:int,static_dim:int,classes:int,h:int=96,drop:float=.25):
        super().__init__(); self.kind,self.fusion=kind,kind=="fusion"; inp=dim+1
        if kind=="cnn": self.encoder=nn.Sequential(nn.Conv1d(inp,h,3,padding=1),nn.GELU(),nn.Dropout(drop),nn.Conv1d(h,h,3,padding=1),nn.GELU())
        else: self.encoder=nn.GRU(inp,h,batch_first=True,bidirectional=True); h*=2
        self.attn=nn.Linear(h,1); self.drop=nn.Dropout(drop); self.fc=nn.Linear(h+(static_dim if self.fusion else 0),classes)
    def forward(self,x,mask,static):
        z=torch.cat([x,mask],-1)
        if self.kind=="cnn": z=self.encoder(z.transpose(1,2)).transpose(1,2)
        else: z,_=self.encoder(z)
        score=self.attn(z).squeeze(-1).masked_fill(mask.squeeze(-1)==0,-1e4); weight=torch.softmax(score,1).unsqueeze(-1); pooled=(z*weight).sum(1)
        if self.fusion: pooled=torch.cat([pooled,static],1)
        return self.fc(self.drop(pooled))

@dataclass
class Config: kind:str="fusion"; hidden:int=96; dropout:float=.25; lr:float=1e-3; weight_decay:float=1e-4; batch:int=64; epochs:int=50; patience:int=8; seed:int=42

def fit_scalers(train:dict):
    scaler=StandardScaler().fit(train["X"].reshape(-1,len(SEQ_COLUMNS))); static=StandardScaler().fit(train["static"]); return scaler,static
def transform(data,scaler,static):
    return (scaler.transform(data["X"].reshape(-1,len(SEQ_COLUMNS))).reshape(data["X"].shape).astype("float32"),data["mask"],static.transform(data["static"]).astype("float32"))
def train_model(train,valid,cfg:Config,classes:list[str],fixed_epochs:int|None=None):
    device="cuda" if torch.cuda.is_available() else "cpu"; seed_all(cfg.seed); sc,ss=fit_scalers(train); tx,tm,ts=transform(train,sc,ss); vx,vm,vs=transform(valid,sc,ss)
    y=np.array([classes.index(x) for x in train["meta"].label.replace(LABEL_MAP)]); vy=np.array([classes.index(x) for x in valid["meta"].label.replace(LABEL_MAP)])
    weights=torch.tensor(compute_class_weight("balanced",classes=np.arange(len(classes)),y=y),dtype=torch.float32,device=device)
    model=Temporal(cfg.kind,len(SEQ_COLUMNS),len(SUMMARY_COLUMNS),len(classes),cfg.hidden,cfg.dropout).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.epochs); loss=nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
    loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(tm),torch.from_numpy(ts),torch.from_numpy(y)),batch_size=cfg.batch,shuffle=True); best=(-1,None,0,0)
    for epoch in range(fixed_epochs or cfg.epochs):
        model.train()
        for x,m,s,t in loader:
            opt.zero_grad(); out=model(x.to(device),m.to(device),s.to(device)); l=loss(out,t.to(device)); l.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        sch.step(); model.eval()
        with torch.no_grad(): pred=model(torch.from_numpy(vx).to(device),torch.from_numpy(vm).to(device),torch.from_numpy(vs).to(device)).argmax(1).cpu().numpy()
        score=f1_score(vy,pred,average="macro",zero_division=0)
        if score>best[0]: best=(score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch+1,0)
        else:
            best=(best[0],best[1],best[2],best[3]+1)
            if fixed_epochs is None and best[3]>=cfg.patience: break
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        best_pred=model(torch.from_numpy(vx).to(device),torch.from_numpy(vm).to(device),torch.from_numpy(vs).to(device)).argmax(1).cpu().numpy()
    metrics={
        "macro_f1": f1_score(vy,best_pred,average="macro",zero_division=0),
        "accuracy": accuracy_score(vy,best_pred),
        "a21_recall": recall_score(vy,best_pred,labels=[classes.index("A21")],average="macro",zero_division=0),
    }
    return model,sc,ss,metrics,best[2]

def evaluate(model,data,sc,ss,classes,out:Path,prefix:str):
    x,m,s=transform(data,sc,ss); device=next(model.parameters()).device; model.eval()
    with torch.no_grad(): logits=model(torch.from_numpy(x).to(device),torch.from_numpy(m).to(device),torch.from_numpy(s).to(device)); prob=torch.softmax(logits,1).cpu().numpy()
    meta=data["meta"].copy(); meta["label"]=meta.label.replace(LABEL_MAP); meta["prediction"]=[classes[i] for i in prob.argmax(1)]; meta[[f"prob_{c}" for c in classes]]=prob; meta.to_csv(out/f"{prefix}_predictions.csv",index=False,encoding="utf-8-sig")
    report=classification_report(meta.label,meta.prediction,labels=classes,output_dict=True,zero_division=0); pd.DataFrame(report).T.to_csv(out/f"{prefix}_classification_report.csv",encoding="utf-8-sig")
    cm=confusion_matrix(meta.label,meta.prediction,labels=classes); pd.DataFrame(cm,index=classes,columns=classes).to_csv(out/f"{prefix}_confusion_matrix.csv",encoding="utf-8-sig")
    fig,ax=plt.subplots(figsize=(7,7)); ConfusionMatrixDisplay(cm,display_labels=classes).plot(ax=ax,colorbar=False); fig.tight_layout(); fig.savefig(out/f"{prefix}_confusion_matrix.png",dpi=180); plt.close(fig)
    return {"accuracy":accuracy_score(meta.label,meta.prediction),"macro_f1":f1_score(meta.label,meta.prediction,average="macro",zero_division=0),"a21_recall":recall_score(meta.label,meta.prediction,labels=["A21"],average="macro",zero_division=0)}

def cv(data,classes,cfg):
    y=data["meta"].label.replace(LABEL_MAP).to_numpy(); groups=data["meta"].group_id.to_numpy(); rows=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(data["X"],y,groups),1):
        assert not(set(groups[tr])&set(groups[va])); a={k:(v[tr] if isinstance(v,np.ndarray) else v.iloc[tr].reset_index(drop=True)) for k,v in data.items()}; b={k:(v[va] if isinstance(v,np.ndarray) else v.iloc[va].reset_index(drop=True)) for k,v in data.items()}
        _,_,_,metrics,epochs=train_model(a,b,cfg,classes)
        rows.append({"candidate":cfg.kind,"fold":fold,"best_epoch":epochs,**metrics})
    return rows

def xgb_cv(data,classes):
    y=data["meta"].label.replace(LABEL_MAP).to_numpy(); groups=data["meta"].group_id.to_numpy(); rows=[]
    for f,(tr,va) in enumerate(GroupKFold(5).split(data["static"],y,groups),1):
        assert not(set(groups[tr])&set(groups[va])); model=XGBClassifier(objective="multi:softprob",num_class=len(classes),n_estimators=450,max_depth=3,learning_rate=.17676056,subsample=.68054065,colsample_bytree=.97166687,min_child_weight=9,gamma=.1248711,random_state=42,eval_metric="mlogloss")
        yy=np.array([classes.index(x) for x in y]); w=np.ones(len(tr)); counts=np.bincount(yy[tr]); w=np.array([len(tr)/(len(classes)*counts[v]) for v in yy[tr]]); w[y[tr]=="A21"]*=1.5; model.fit(data["static"][tr],yy[tr],sample_weight=w)
        pred=model.predict(data["static"][va])
        rows.append({"candidate":"xgboost_static_baseline","fold":f,"macro_f1":f1_score(yy[va],pred,average="macro",zero_division=0),"accuracy":accuracy_score(yy[va],pred),"a21_recall":recall_score(yy[va],pred,labels=[classes.index("A21")],average="macro",zero_division=0),"best_epoch":np.nan})
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["prepare","screen","tune","final"]); p.add_argument("--trials",type=int,default=10); p.add_argument("--kind",default="fusion",choices=["gru","cnn","fusion"]); p.add_argument("--seed",type=int,default=42); args=p.parse_args(); OUT.mkdir(parents=True,exist_ok=True); classes=["A17","A18_20","A19","A21","N1"]
    if args.command=="prepare":
        # This explicitly invokes only the v2 split writer, never edits original manifests.
        from make_holdout_test_v2 import main as split_main
        import sys; old=sys.argv; sys.argv=[old[0]]; split_main(); sys.argv=old
        split=OUT/"splits"; save_data("train",build(split/"train_annotations_holdout_excluded.csv","train")); save_data("holdout",build(split/"holdout_test.csv","train")); save_data("validation",build(ROOT/"splits/annotations/valid_annotations.csv","valid")); print("Prepared v2-only sequence caches."); return
    train=load_data("train")
    if args.command in {"screen","tune"}:
        if args.command=="tune":
            # Each objective uses only the holdout-excluded training set and
            # writes partial results after every trial for resumable runs.
            all_rows=[]
            def objective(trial):
                cfg=Config(kind=args.kind,hidden=trial.suggest_categorical("hidden",[64,96,128]),dropout=trial.suggest_float("dropout",.10,.40),lr=trial.suggest_float("lr",3e-4,3e-3,log=True),weight_decay=trial.suggest_float("weight_decay",1e-5,1e-3,log=True),batch=trial.suggest_categorical("batch",[32,64]),seed=args.seed+trial.number)
                rows=cv(train,classes,cfg)
                for row in rows: row.update({"trial":trial.number,**asdict(cfg)})
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(OUT/f"cv_results_tune_{args.kind}.csv",index=False,encoding="utf-8-sig")
                trial.set_user_attr("a21_recall_mean",float(np.mean([r["a21_recall"] for r in rows])))
                print(f"trial={trial.number} macro_f1={np.mean([r['macro_f1'] for r in rows]):.6f} a21_recall={trial.user_attrs['a21_recall_mean']:.6f}",flush=True)
                return float(np.mean([r["macro_f1"] for r in rows]))
            study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=args.seed))
            study.optimize(objective,n_trials=args.trials)
            (OUT/f"tune_{args.kind}_best.json").write_text(json.dumps({"best_params":study.best_params,"best_value":study.best_value,"best_trial":study.best_trial.number,"a21_recall_mean":study.best_trial.user_attrs.get("a21_recall_mean")},indent=2),encoding="utf-8")
            return
        configs=[Config(kind=k,seed=args.seed) for k in ["gru","cnn","fusion"]]; rows=xgb_cv(train,classes)
        for base in configs:
            # Deterministic random search; each proposed config is evaluated exclusively by 5-fold grouped CV.
            rng=np.random.default_rng(args.seed)
            for trial in range(1):
                c=base if trial==0 else Config(kind=base.kind,hidden=int(rng.choice([64,96,128])),dropout=float(rng.uniform(.1,.4)),lr=float(10**rng.uniform(-3.5,-2.5)),weight_decay=float(10**rng.uniform(-5,-3)),batch=int(rng.choice([32,64])),seed=args.seed+trial)
                for r in cv(train,classes,c): r.update({"trial":trial,**asdict(c)}); rows.append(r)
        pd.DataFrame(rows).to_csv(OUT/"cv_results_screen_metrics.csv",index=False,encoding="utf-8-sig"); print(pd.DataFrame(rows).groupby("candidate")[["macro_f1","accuracy","a21_recall"]].mean()); return
    # final: values must be supplied by the frozen winning CV configuration, not validation/holdout results.
    cfg_path=OUT/"selected_config.json"
    if not cfg_path.is_file(): raise SystemExit("Create results/evaluation/lstm_v2/selected_config.json from CV result before final.")
    cfg=Config(**json.loads(cfg_path.read_text(encoding="utf-8"))); hold=load_data("holdout"); valid=load_data("validation"); rows=cv(train,classes,cfg); epochs=round(np.mean([r["best_epoch"] for r in rows])); model,sc,ss,_,_=train_model(train,train,cfg,classes,fixed_epochs=max(1,epochs))
    # train_model's validation argument is only used for reporting under fixed epochs; its weights are never selected from it.
    joblib.dump({"sequence_scaler":sc,"static_scaler":ss,"sequence_columns":SEQ_COLUMNS,"summary_columns":SUMMARY_COLUMNS},OUT/"scalers.joblib"); torch.save({"state_dict":model.cpu().state_dict(),"config":asdict(cfg),"classes":classes,"epochs":epochs},OUT/"final_model.pt")
    model.to("cpu"); dummy=(torch.zeros(1,T,len(SEQ_COLUMNS)),torch.ones(1,T,1),torch.zeros(1,len(SUMMARY_COLUMNS))); torch.onnx.export(model,dummy,OUT/"final_model.onnx",input_names=["sequence","mask","summary"],output_names=["logits"],opset_version=17,dynamic_axes={"sequence":{0:"batch"},"mask":{0:"batch"},"summary":{0:"batch"},"logits":{0:"batch"}})
    result={"config":asdict(cfg),"cv":rows,"validation":evaluate(model,valid,sc,ss,classes,OUT,"validation"),"holdout":evaluate(model,hold,sc,ss,classes,OUT,"holdout"),"feature_columns":{"sequence":SEQ_COLUMNS,"summary":SUMMARY_COLUMNS}}
    (OUT/"summary.txt").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
