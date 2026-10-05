"""CV-only image-augmented temporal experiment; never reads validation/holdout."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score,classification_report,confusion_matrix,f1_score,recall_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from train_lstm_v2 import Config,LABEL_MAP,load_data,seed_all

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/evaluation/lstm_v2_img'; BASE=ROOT/'results/evaluation/lstm_v2'; CLASSES=['A17','A18_20','A19','A21','N1']
class ImgFusion(nn.Module):
 def __init__(self,inp,static,classes,hidden,drop,fusion=True,img_drop=.2):
  super().__init__();self.fusion=fusion;self.img_drop=nn.Dropout(img_drop);self.rnn=nn.GRU(inp,hidden,batch_first=True,bidirectional=True);self.att=nn.Linear(hidden*2,1);self.drop=nn.Dropout(drop);self.fc=nn.Linear(hidden*2+(static if fusion else 0),classes)
 def forward(self,x,mask,static):
  z=self.rnn(x)[0];w=torch.softmax(self.att(z).squeeze(-1).masked_fill(mask.squeeze(-1)==0,-1e4),1).unsqueeze(-1);z=(z*w).sum(1)
  if self.fusion:z=torch.cat([z,static],1)
  return self.fc(self.drop(z))
def build_img_cache():
 data=load_data('train');meta=data['meta'];mp=pd.read_csv(OUT/'mapping.csv');lookup={(r.video,r.group_id):r.npz_file for _,r in mp.iterrows()};X=np.zeros((len(meta),30,512),np.float32);present=np.zeros((len(meta),30,1),np.float32)
 for i,r in meta.iterrows():
  f=OUT/'emb'/lookup[(r.video,r.group_id)]
  with np.load(f) as z:
   d=dict(zip(z['frames'].tolist(),range(len(z['frames']))))
   for k in range(30):
    j=d.get(int(r.start_frame)+k*10)
    if j is not None:X[i,k]=z['embeddings'][j];present[i,k,0]=z['present'][j]
 np.savez_compressed(OUT/'train_img_cache.npz',image=X,img_present=present);print('saved',X.shape,present.mean())
def load():
 d=load_data('train');z=np.load(OUT/'train_img_cache.npz');d['image']=z['image'];d['img_present']=z['img_present'];return d
def subset(d,idx):return {k:(v[idx] if isinstance(v,np.ndarray) else v.iloc[idx].reset_index(drop=True)) for k,v in d.items()}
def fit_transform(tr,va,pca_dim=64):
 # Train-only scalers/PCA. Fit image transforms only on visible person crops.
 seq=StandardScaler().fit(tr['X'].reshape(-1,12));sta=StandardScaler().fit(tr['static']);visible=tr['img_present'].reshape(-1).astype(bool);raw=tr['image'].reshape(-1,512);imgsc=StandardScaler().fit(raw[visible]);pca=PCA(n_components=pca_dim,random_state=42).fit(imgsc.transform(raw[visible]))
 def one(d):
  a=seq.transform(d['X'].reshape(-1,12)).reshape(d['X'].shape); im=pca.transform(imgsc.transform(d['image'].reshape(-1,512))).reshape(len(d['X']),30,pca_dim);im*=d['img_present'];return a.astype('float32'),sta.transform(d['static']).astype('float32'),im.astype('float32')
 return one(tr),one(va),(seq,sta,imgsc,pca)
def run_fold(tr,va,cfg,kind,img_drop=.2,a21_weight=1.0):
 seed_all(cfg.seed);(x,s,img),(vx,vs,vimg),arts=fit_transform(tr,va); y=np.array([CLASSES.index(v) for v in tr['meta'].label.replace(LABEL_MAP)]);vy=np.array([CLASSES.index(v) for v in va['meta'].label.replace(LABEL_MAP)])
 if kind=='fusion_img': tx=np.concatenate([x,img,tr['img_present']],2);vtx=np.concatenate([vx,vimg,va['img_present']],2);static=s;vstatic=vs;fusion=True
 else: tx=np.concatenate([img,tr['img_present']],2);vtx=np.concatenate([vimg,va['img_present']],2);static=np.zeros((len(x),0),np.float32);vstatic=np.zeros((len(vx),0),np.float32);fusion=False
 dev='cuda' if torch.cuda.is_available() else 'cpu';counts=np.bincount(y,minlength=5);w=torch.tensor([len(y)/(5*counts[i]) for i in range(5)],dtype=torch.float32,device=dev);w[CLASSES.index('A21')]*=a21_weight
 model=ImgFusion(tx.shape[2],static.shape[1],5,cfg.hidden,cfg.dropout,fusion,img_drop).to(dev);opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay);loss=nn.CrossEntropyLoss(weight=w,label_smoothing=.03);loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(tr['img_present']),torch.from_numpy(static),torch.from_numpy(y)),batch_size=cfg.batch,shuffle=True);best=(-1,None,0,0)
 for e in range(cfg.epochs):
  model.train()
  for a,b,c,t in loader:opt.zero_grad();o=model(a.to(dev),b.to(dev),c.to(dev));l=loss(o,t.to(dev));l.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  model.eval()
  with torch.no_grad():p=model(torch.from_numpy(vtx).to(dev),torch.from_numpy(va['img_present']).to(dev),torch.from_numpy(vstatic).to(dev)).argmax(1).cpu().numpy()
  sc=f1_score(vy,p,average='macro',zero_division=0)
  if sc>best[0]:best=(sc,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},e+1,0)
  else:
   best=(best[0],best[1],best[2],best[3]+1)
   if best[3]>=cfg.patience:break
 model.load_state_dict(best[1]);model.eval()
 with torch.no_grad():prob=torch.softmax(model(torch.from_numpy(vtx).to(dev),torch.from_numpy(va['img_present']).to(dev),torch.from_numpy(vstatic).to(dev)),1).cpu().numpy()
 return prob,{'macro_f1':f1_score(vy,prob.argmax(1),average='macro',zero_division=0),'accuracy':accuracy_score(vy,prob.argmax(1)),'a21_recall':recall_score(vy,prob.argmax(1),labels=[CLASSES.index('A21')],average='macro',zero_division=0),'best_epoch':best[2]},arts
def cv(kind,img_drop=.2):
 d=load();y=d['meta'].label.replace(LABEL_MAP).to_numpy();g=d['meta'].group_id.to_numpy();trial=pd.read_csv(BASE/'cv_results_tune_fusion.csv');r=trial[trial.trial==6].iloc[0];cfg=Config(kind='fusion',hidden=int(r.hidden),dropout=float(r.dropout),lr=float(r.lr),weight_decay=float(r.weight_decay),batch=int(r.batch),epochs=int(r.epochs),patience=int(r.patience),seed=int(r.seed));rows=[];oof=[]
 for fold,(tr,va) in enumerate(GroupKFold(5).split(d['X'],y,g),1):
  assert not(set(g[tr])&set(g[va]));prob,met,arts=run_fold(subset(d,tr),subset(d,va),cfg,kind,img_drop);met.update({'fold':fold,'candidate':kind,'img_dropout':img_drop});rows.append(met);m=d['meta'].iloc[va].copy();m['fold']=fold;m['label']=y[va];m['prediction']=[CLASSES[i] for i in prob.argmax(1)];oof.append(m)
  import joblib;joblib.dump({'pca':arts[3],'image_scaler':arts[2],'sequence_scaler':arts[0],'static_scaler':arts[1]},OUT/f'{kind}_fold{fold}_artifacts.joblib')
 out=pd.concat(oof,ignore_index=True);out.to_csv(OUT/f'oof_{kind}.csv',index=False,encoding='utf-8-sig');pd.DataFrame(rows).to_csv(OUT/f'cv_{kind}.csv',index=False,encoding='utf-8-sig');rep=classification_report(out.label,out.prediction,labels=CLASSES,output_dict=True,zero_division=0);pd.DataFrame(rep).T.to_csv(OUT/f'oof_{kind}_classification_report.csv',encoding='utf-8-sig');cm=confusion_matrix(out.label,out.prediction,labels=CLASSES);pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(OUT/f'oof_{kind}_confusion_matrix.csv',encoding='utf-8-sig');fig,ax=plt.subplots(figsize=(7,7));ConfusionMatrixDisplay(cm,display_labels=CLASSES).plot(ax=ax,colorbar=False);fig.tight_layout();fig.savefig(OUT/f'oof_{kind}_confusion_matrix.png',dpi=180);plt.close(fig);errs=[{'actual':CLASSES[i],'predicted':CLASSES[j],'count':int(cm[i,j])} for i in range(5) for j in range(5) if i!=j and cm[i,j]];pd.DataFrame(errs).sort_values('count',ascending=False).to_csv(OUT/f'oof_{kind}_top_errors.csv',index=False,encoding='utf-8-sig');print(pd.DataFrame(rows).mean(numeric_only=True),flush=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','cv','cv-all']);p.add_argument('--kind',choices=['fusion_img','img_only']);p.add_argument('--img-dropout',type=float,default=.2);a=p.parse_args()
 if a.command=='prepare':build_img_cache()
 elif a.command=='cv':cv(a.kind,a.img_dropout)
 else:
  cv('fusion_img',a.img_dropout);cv('img_only',a.img_dropout)
if __name__=='__main__':main()
