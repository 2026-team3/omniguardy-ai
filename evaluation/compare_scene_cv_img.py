"""CV-only comparison under photographer/scene groups; never reads validation/holdout."""
from __future__ import annotations
import json,re
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch,matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,classification_report,confusion_matrix,f1_score,recall_score
from sklearn.model_selection import GroupKFold

from train_lstm_v2 import Config,LABEL_MAP,Temporal,fit_scalers,load_data,seed_all,transform
from train_lstm_v2_img import CLASSES,OUT,load,run_fold,subset

SCENE_RE=re.compile(r"(?:^|_)SY(?P<sy>\d+)_P(?P<p>\d+)(?:_|\.)",re.I)
BASE=Path(__file__).resolve().parents[1]/"results/evaluation/lstm_v2"

def scene_groups(meta:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for _,r in meta.iterrows():
        name=Path(str(r.video)).name
        hit=SCENE_RE.search(name)
        if hit:
            # Source is retained so independently named datasets cannot be accidentally merged.
            source=str(r.group_id).split(':',1)[0]
            group=f"{source}:SY{int(hit['sy']):02d}:P{int(hit['p']):02d}"
            rule="parsed_source_SY_P"
        else:
            # Retain original group_id: it keeps an original and all _augN variants together.
            group=f"fallback:{r.group_id}"
            rule="fallback_original_group_id_no_SY_P"
        rows.append((group,rule))
    ans=meta[[c for c in ['video','group_id','source','label'] if c in meta]].copy()
    ans[['scene_group','parse_rule']]=rows
    return ans

def temporal_fold(tr,va,cfg):
    seed_all(cfg.seed);sc,ss=fit_scalers(tr);tx,tm,ts=transform(tr,sc,ss);vx,vm,vs=transform(va,sc,ss)
    y=np.array([CLASSES.index(v) for v in tr['meta'].label.replace(LABEL_MAP)]);vy=np.array([CLASSES.index(v) for v in va['meta'].label.replace(LABEL_MAP)])
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    from sklearn.utils.class_weight import compute_class_weight
    from torch import nn
    from torch.utils.data import DataLoader,TensorDataset
    weights=torch.tensor(compute_class_weight('balanced',classes=np.arange(len(CLASSES)),y=y),dtype=torch.float32,device=dev)
    model=Temporal('fusion',tx.shape[2],ts.shape[1],len(CLASSES),cfg.hidden,cfg.dropout).to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.epochs)
    loss=nn.CrossEntropyLoss(weight=weights,label_smoothing=.03)
    loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(tm),torch.from_numpy(ts),torch.from_numpy(y)),batch_size=cfg.batch,shuffle=True)
    best=(-1,None,0,0)
    for epoch in range(cfg.epochs):
        model.train()
        for x,m,s,t in loader:
            opt.zero_grad();out=model(x.to(dev),m.to(dev),s.to(dev));value=loss(out,t.to(dev));value.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        scheduler.step();model.eval()
        with torch.no_grad(): pred=model(torch.from_numpy(vx).to(dev),torch.from_numpy(vm).to(dev),torch.from_numpy(vs).to(dev)).argmax(1).cpu().numpy()
        score=f1_score(vy,pred,average='macro',zero_division=0)
        if score>best[0]: best=(score,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch+1,0)
        else:
            best=(best[0],best[1],best[2],best[3]+1)
            if best[3]>=cfg.patience: break
    model.load_state_dict(best[1]);model.eval()
    with torch.no_grad(): prob=torch.softmax(model(torch.from_numpy(vx).to(dev),torch.from_numpy(vm).to(dev),torch.from_numpy(vs).to(dev)),1).cpu().numpy()
    met={'macro_f1':f1_score(vy,prob.argmax(1),average='macro',zero_division=0),'accuracy':accuracy_score(vy,prob.argmax(1)),'a21_recall':recall_score(vy,prob.argmax(1),labels=[CLASSES.index('A21')],average='macro',zero_division=0),'best_epoch':best[2]}
    return prob,met,{'sequence_scaler':sc,'static_scaler':ss}

def save_oof(name,rows,oof):
    cv=pd.DataFrame(rows);cv.to_csv(OUT/f'cv_scene_{name}.csv',index=False,encoding='utf-8-sig')
    all_oof=pd.concat(oof,ignore_index=True);all_oof.to_csv(OUT/f'oof_scene_{name}.csv',index=False,encoding='utf-8-sig')
    rep=classification_report(all_oof.label,all_oof.prediction,labels=CLASSES,output_dict=True,zero_division=0)
    pd.DataFrame(rep).T.to_csv(OUT/f'oof_scene_{name}_classification_report.csv',encoding='utf-8-sig')
    cm=confusion_matrix(all_oof.label,all_oof.prediction,labels=CLASSES)
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(OUT/f'oof_scene_{name}_confusion_matrix.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(7,7));ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(ax=ax,colorbar=False);fig.tight_layout();fig.savefig(OUT/f'oof_scene_{name}_confusion_matrix.png',dpi=180);plt.close(fig)
    return cv,all_oof

def main():
    data=load(); meta=data['meta'].copy();sg=scene_groups(meta);sg.to_csv(OUT/'scene_split_mapping.csv',index=False,encoding='utf-8-sig')
    trial=pd.read_csv(BASE/'cv_results_tune_fusion.csv');r=trial[trial.trial==6].iloc[0]
    cfg=Config(kind='fusion',hidden=int(r.hidden),dropout=float(r.dropout),lr=float(r.lr),weight_decay=float(r.weight_decay),batch=int(r.batch),epochs=int(r.epochs),patience=int(r.patience),seed=int(r.seed))
    y=meta.label.replace(LABEL_MAP).to_numpy();g=sg.scene_group.to_numpy();splitter=GroupKFold(5)
    results={};oofs={}
    for name in ['fusion_trial6','fusion_img']:
        rows=[];oof=[]
        for fold,(tr,va) in enumerate(splitter.split(data['X'],y,g),1):
            assert not (set(g[tr]) & set(g[va])),f'scene leakage at fold {fold}'
            if name=='fusion_trial6': prob,met,arts=temporal_fold(subset(data,tr),subset(data,va),cfg)
            else: prob,met,arts=run_fold(subset(data,tr),subset(data,va),cfg,'fusion_img',.2,a21_weight=1.0)
            met.update({'fold':fold,'candidate':name,'scene_train_groups':len(set(g[tr])),'scene_valid_groups':len(set(g[va]))});rows.append(met)
            cur=meta.iloc[va].copy();cur['scene_group']=g[va];cur['fold']=fold;cur['label']=y[va];cur['prediction']=[CLASSES[i] for i in prob.argmax(1)];oof.append(cur)
            joblib.dump(arts,OUT/f'scene_{name}_fold{fold}_artifacts.joblib')
            print(name,fold,met,flush=True)
        results[name],oofs[name]=save_oof(name,rows,oof)
    base=results['fusion_trial6'];img=results['fusion_img'];diff=img.macro_f1.to_numpy()-base.macro_f1.to_numpy()
    old=trial[trial.trial==6][['fold','macro_f1','accuracy','a21_recall']].sort_values('fold')
    old.to_csv(OUT/'group_id_trial6_existing_folds.csv',index=False,encoding='utf-8-sig')
    verdict={'comparison':'scene_group=source:SY:P; fallback=original group_id','fusion_img_wins':int((diff>0).sum()),'mean_macro_f1_difference':float(diff.mean()),'std_macro_f1_difference':float(diff.std(ddof=1)),'fusion_img_a21_recall_mean':float(img.a21_recall.mean()),'criteria':{'wins_at_least_4':bool((diff>0).sum()>=4),'macro_gain_at_least_0_01':bool(diff.mean()>=.01),'a21_recall_at_least_0_766':bool(img.a21_recall.mean()>=.766)},'image_helps':bool((diff>0).sum()>=4 and diff.mean()>=.01 and img.a21_recall.mean()>=.766)}
    (OUT/'scene_cv_comparison.json').write_text(json.dumps(verdict,indent=2),encoding='utf-8')
    summary=pd.concat([results['fusion_trial6'].assign(split='scene_SY_P'),results['fusion_img'].assign(split='scene_SY_P')])
    summary.to_csv(OUT/'scene_cv_fold_comparison.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(verdict,indent=2),flush=True)

if __name__=='__main__': main()
