"""CV-only comparison of temporal candidates and an equally tuned XGBoost.

This script intentionally never loads validation.npz or holdout.npz.  All
artifacts are additional files below results/evaluation/lstm_v2.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier

from train_lstm_v2 import OUT, Config, LABEL_MAP, load_data, train_model, transform

# Keep the label ordering fixed in every generated artifact.
CLASSES = ["A17", "A18_20", "A19", "A21", "N1"]


def labels(data):
    return data["meta"].label.replace(LABEL_MAP).to_numpy()


def xgb_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob", num_class=len(CLASSES), eval_metric="mlogloss", random_state=42,
        n_jobs=1, **params,
    )


def weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(CLASSES))
    value = np.array([len(y) / (len(CLASSES) * counts[idx]) for idx in y])
    value[y == CLASSES.index("A21")] *= 1.5
    return value


def xgb_cv(data, params: dict, candidate: str) -> tuple[list[dict], pd.DataFrame]:
    target = labels(data); encoded = np.array([CLASSES.index(v) for v in target])
    groups = data["meta"].group_id.to_numpy(); rows=[]; oof=[]
    for fold, (tr, va) in enumerate(GroupKFold(5).split(data["static"], encoded, groups), 1):
        assert not (set(groups[tr]) & set(groups[va]))
        model=xgb_model(params); model.fit(data["static"][tr],encoded[tr],sample_weight=weights(encoded[tr]))
        prob=model.predict_proba(data["static"][va]); pred=prob.argmax(1)
        rows.append({"candidate":candidate,"fold":fold,"macro_f1":f1_score(encoded[va],pred,average="macro",zero_division=0),"accuracy":accuracy_score(encoded[va],pred),"a21_recall":recall_score(encoded[va],pred,labels=[CLASSES.index("A21")],average="macro",zero_division=0)})
        meta=data["meta"].iloc[va].copy(); meta["fold"]=fold; meta["label"]=target[va]; meta["prediction"]=[CLASSES[i] for i in pred]
        for index, name in enumerate(CLASSES): meta[f"prob_{name}"]=prob[:,index]
        oof.append(meta)
    return rows, pd.concat(oof,ignore_index=True)


def temporal_oof(data, config: Config, candidate: str) -> tuple[list[dict], pd.DataFrame]:
    target=labels(data); groups=data["meta"].group_id.to_numpy(); rows=[]; oof=[]
    for fold,(tr,va) in enumerate(GroupKFold(5).split(data["X"],target,groups),1):
        assert not (set(groups[tr]) & set(groups[va]))
        train={key:(value[tr] if isinstance(value,np.ndarray) else value.iloc[tr].reset_index(drop=True)) for key,value in data.items()}
        valid={key:(value[va] if isinstance(value,np.ndarray) else value.iloc[va].reset_index(drop=True)) for key,value in data.items()}
        model,scaler,static_scaler,metrics,_=train_model(train,valid,config,CLASSES)
        x,m,s=transform(valid,scaler,static_scaler); model.eval()
        device=next(model.parameters()).device
        with torch.no_grad():
            logits=model(torch.from_numpy(x).to(device),torch.from_numpy(m).to(device),torch.from_numpy(s).to(device))
            prob=torch.softmax(logits,1).cpu().numpy()
        pred=prob.argmax(1); rows.append({"candidate":candidate,"fold":fold,**metrics})
        meta=valid["meta"].copy(); meta["fold"]=fold; meta["label"]=target[va]; meta["prediction"]=[CLASSES[i] for i in pred]
        for index,name in enumerate(CLASSES): meta[f"prob_{name}"]=prob[:,index]
        oof.append(meta)
    return rows,pd.concat(oof,ignore_index=True)


def write_oof(name: str, oof: pd.DataFrame) -> dict:
    oof.to_csv(OUT/f"oof_{name}.csv",index=False,encoding="utf-8-sig")
    report=classification_report(oof.label,oof.prediction,labels=CLASSES,output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT/f"oof_{name}_classification_report.csv",encoding="utf-8-sig")
    matrix=confusion_matrix(oof.label,oof.prediction,labels=CLASSES)
    pd.DataFrame(matrix,index=CLASSES,columns=CLASSES).to_csv(OUT/f"oof_{name}_confusion_matrix.csv",encoding="utf-8-sig")
    figure,axis=plt.subplots(figsize=(7,7)); ConfusionMatrixDisplay(matrix,display_labels=CLASSES).plot(ax=axis,colorbar=False); figure.tight_layout(); figure.savefig(OUT/f"oof_{name}_confusion_matrix.png",dpi=180); plt.close(figure)
    pairs=[]
    for actual_index,actual in enumerate(CLASSES):
        for predicted_index,predicted in enumerate(CLASSES):
            if actual_index != predicted_index and matrix[actual_index,predicted_index]: pairs.append({"actual":actual,"predicted":predicted,"count":int(matrix[actual_index,predicted_index])})
    pd.DataFrame(pairs).sort_values("count",ascending=False).to_csv(OUT/f"oof_{name}_top_errors.csv",index=False,encoding="utf-8-sig")
    return {"macro_f1":f1_score(oof.label,oof.prediction,average="macro",zero_division=0),"accuracy":accuracy_score(oof.label,oof.prediction),"a21_recall":recall_score(oof.label,oof.prediction,labels=["A21"],average="macro",zero_division=0),"report":report}


def tune_xgb(data,trials:int,seed:int):
    all_rows=[]
    def objective(trial):
        params={"n_estimators":trial.suggest_int("n_estimators",150,600,step=50),"max_depth":trial.suggest_int("max_depth",3,8),"learning_rate":trial.suggest_float("learning_rate",.01,.25,log=True),"subsample":trial.suggest_float("subsample",.6,1.0),"colsample_bytree":trial.suggest_float("colsample_bytree",.6,1.0),"min_child_weight":trial.suggest_int("min_child_weight",1,12),"gamma":trial.suggest_float("gamma",0.,2.),"reg_lambda":trial.suggest_float("reg_lambda",1e-3,10.,log=True)}
        rows,_=xgb_cv(data,params,"xgboost_tuned")
        for row in rows: row.update({"trial":trial.number,**params})
        all_rows.extend(rows); pd.DataFrame(all_rows).to_csv(OUT/"cv_results_xgboost_tune.csv",index=False,encoding="utf-8-sig")
        a21=float(np.mean([row["a21_recall"] for row in rows])); trial.set_user_attr("a21_recall_mean",a21)
        value=float(np.mean([row["macro_f1"] for row in rows])); print(f"trial={trial.number} macro_f1={value:.6f} a21_recall={a21:.6f}",flush=True); return value
    study=optuna.create_study(direction="maximize",sampler=optuna.samplers.TPESampler(seed=seed)); study.optimize(objective,n_trials=trials)
    result={"best_trial":study.best_trial.number,"best_value":study.best_value,"best_params":study.best_params,"a21_recall_mean":study.best_trial.user_attrs["a21_recall_mean"]}
    (OUT/"xgboost_tune_best.json").write_text(json.dumps(result,indent=2),encoding="utf-8")


def compare_oof():
    data=load_data("train")
    trial=pd.read_csv(OUT/"cv_results_tune_fusion.csv")
    configs={}
    for number in (11,6):
        row=trial[trial.trial==number].iloc[0]
        configs[f"fusion_trial{number}"]=Config(kind="fusion",hidden=int(row.hidden),dropout=float(row.dropout),lr=float(row.lr),weight_decay=float(row.weight_decay),batch=int(row.batch),epochs=int(row.epochs),patience=int(row.patience),seed=int(row.seed))
    xgb=json.loads((OUT/"xgboost_tune_best.json").read_text(encoding="utf-8"))["best_params"]
    candidates={}
    for name,cfg in configs.items():
        rows,oof=temporal_oof(data,cfg,name); candidates[name]={"folds":rows,**write_oof(name,oof)}
    rows,oof=xgb_cv(data,xgb,"xgboost_tuned"); candidates["xgboost_tuned"]={"folds":rows,**write_oof("xgboost_tuned",oof)}
    # Pair folds exactly by their common GroupKFold index.
    fusion=pd.DataFrame(candidates["fusion_trial11"]["folds"]).sort_values("fold")
    xgb_folds=pd.DataFrame(candidates["xgboost_tuned"]["folds"]).sort_values("fold")
    diff=fusion.macro_f1.to_numpy()-xgb_folds.macro_f1.to_numpy()
    candidates["fusion11_minus_xgboost"]={"fold_macro_f1_difference":diff.tolist(),"mean":float(diff.mean()),"std":float(diff.std(ddof=1)),"wins":int((diff>0).sum()),"folds":int(len(diff))}
    (OUT/"cv_oof_comparison.json").write_text(json.dumps(candidates,ensure_ascii=False,indent=2,default=float),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["tune-xgb","oof"]); parser.add_argument("--trials",type=int,default=20); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args()
    data=load_data("train")
    if args.command=="tune-xgb": tune_xgb(data,args.trials,args.seed)
    else: compare_oof()
if __name__=="__main__": main()
