"""Prepare fixed serving candidates from holdout-excluded train data only.

This deliberately does not load validation or internal-holdout caches.
"""
from __future__ import annotations
import hashlib,json,shutil
from dataclasses import asdict
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

from train_lstm_v2 import Config,LABEL_MAP,Temporal,fit_scalers,load_data,seed_all,transform
from train_lstm_v2_img import CLASSES,ImgFusion,fit_transform,load as load_image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/evaluation/lstm_v2_img/final_candidates'
WEIGHTS=ROOT/'results/evaluation/lstm_v2_img/torch_cache/checkpoints/resnet18-f37072fd.pth'

def sha256(path:Path)->str:
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def labels(meta): return np.array([CLASSES.index(v) for v in meta.label.replace(LABEL_MAP)])

def run_epochs(model,tx,mask,static,y,cfg):
 dev='cuda' if torch.cuda.is_available() else 'cpu';model=model.to(dev)
 w=torch.tensor(compute_class_weight('balanced',classes=np.arange(5),y=y),dtype=torch.float32,device=dev)
 opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
 sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.epochs)
 loss=nn.CrossEntropyLoss(weight=w,label_smoothing=.03)
 loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(mask),torch.from_numpy(static),torch.from_numpy(y)),batch_size=cfg.batch,shuffle=True)
 for _ in range(cfg.epochs):
  model.train()
  for x,m,s,t in loader:
   opt.zero_grad();v=loss(model(x.to(dev),m.to(dev),s.to(dev)),t.to(dev));v.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  sch.step()
 return model.eval()

def export_parity(model,inputs,out):
 import onnxruntime as ort
 out.mkdir(parents=True,exist_ok=False)
 model_cpu=model.to('cpu').eval(); names=['sequence','mask','summary'];onnx_path=out/'model.onnx'
 torch.onnx.export(model_cpu,inputs,onnx_path,input_names=names,output_names=['logits'],opset_version=17,dynamo=False)
 with torch.no_grad(): ref=model_cpu(*inputs).numpy()
 sess=ort.InferenceSession(str(onnx_path),providers=['CPUExecutionProvider'])
 actual=sess.run(['logits'],{n:x.numpy() for n,x in zip(names,inputs)})[0]
 stats={'onnx_path':onnx_path.name,'opset':17,'providers':sess.get_providers(),'sample_size':int(inputs[0].shape[0]),'max_abs_error':float(np.max(np.abs(ref-actual))),'mean_abs_error':float(np.mean(np.abs(ref-actual))),'argmax_match_rate':float((ref.argmax(1)==actual.argmax(1)).mean())}
 (out/'onnx_parity.json').write_text(json.dumps(stats,indent=2),encoding='utf-8');return stats

def manifest(candidate,cfg,sequence_dim,mask_semantics,artifacts,parity):
 image=None if candidate!='fusion_img' else {'enabled':True,'model':'torchvision.models.resnet18','weights':'ResNet18_Weights.IMAGENET1K_V1 / resnet18-f37072fd.pth','weights_path':str(WEIGHTS.relative_to(ROOT)) if WEIGHTS.exists() else None,'weights_sha256':sha256(WEIGHTS) if WEIGHTS.exists() else None,'output_dim':512,'person_selection':'largest-area tracking bbox per slot','padding_fraction':.12,'resize':'torchvision ImageNet preset: resize short side 256 then center crop 224x224','normalization':'ImageNet normalization supplied by ResNet18_Weights.IMAGENET1K_V1.transforms','fine_tuned':False}
 return {'candidate':candidate,'training_data':'lstm_v2 train cache only (holdout excluded); validation/holdout not loaded','classes':CLASSES,'config':asdict(cfg),'sequence':{'slots':30,'frame_rule':'window_start_frame + 10*k, k=0..29','sequence_dim':sequence_dim,'mask_semantics':mask_semantics},'image_embedding':image,'artifacts':artifacts,'onnx_parity':parity}

def temporal_candidate(cfg):
 d=load_data('train');sc,ss=fit_scalers(d);x,m,s=transform(d,sc,ss);y=labels(d['meta'])
 model=run_epochs(Temporal('fusion',x.shape[2],s.shape[1],5,cfg.hidden,cfg.dropout),x,m,s,y,cfg)
 out=OUT/'fusion_trial6';parity=export_parity(model,(torch.from_numpy(x[:8]),torch.from_numpy(m[:8]),torch.from_numpy(s[:8])),out)
 torch.save({'state_dict':model.cpu().state_dict(),'config':asdict(cfg),'classes':CLASSES},out/'model.pt');joblib.dump({'sequence_scaler':sc,'summary_scaler':ss},out/'scalers.joblib')
 info=manifest('fusion_trial6',cfg,x.shape[2],'pose_present (1=pose detected)',{'model':'model.pt','preprocessing':'scalers.joblib','onnx':'model.onnx'},parity)
 (out/'serving_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')

def image_candidate(cfg):
 d=load_image();(x,s,img),_,arts=fit_transform(d,d);y=labels(d['meta'])
 tx=np.concatenate([x,img,d['img_present']],axis=2).astype('float32')
 # Standard balanced class weighting, identical to the scene-group comparison.
 model=run_epochs(ImgFusion(tx.shape[2],s.shape[1],5,cfg.hidden,cfg.dropout,True,.2),tx,d['img_present'],s,y,cfg)
 out=OUT/'fusion_img';parity=export_parity(model,(torch.from_numpy(tx[:8]),torch.from_numpy(d['img_present'][:8]),torch.from_numpy(s[:8])),out)
 torch.save({'state_dict':model.cpu().state_dict(),'config':asdict(cfg),'classes':CLASSES,'image_dropout':.2},out/'model.pt')
 joblib.dump({'sequence_scaler':arts[0],'summary_scaler':arts[1],'image_scaler':arts[2],'pca_64':arts[3]},out/'preprocessors.joblib')
 info=manifest('fusion_img',cfg,tx.shape[2],'img_present (1=person crop embedding available)',{'model':'model.pt','preprocessing':'preprocessors.joblib','onnx':'model.onnx','embedding_weights_external':'../torch_cache/checkpoints/resnet18-f37072fd.pth'},parity)
 (out/'serving_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')

def main():
 if OUT.exists(): raise FileExistsError(f'Refusing to overwrite {OUT}')
 trial=pd.read_csv(ROOT/'results/evaluation/lstm_v2/cv_results_tune_fusion.csv');r=trial[trial.trial==6].iloc[0]
 # Mean CV best epochs: Fusion 23.2 -> 23; image CV 14.0 -> 14.
 base=dict(kind='fusion',hidden=int(r.hidden),dropout=float(r.dropout),lr=float(r.lr),weight_decay=float(r.weight_decay),batch=int(r.batch),patience=int(r.patience),seed=int(r.seed))
 seed_all(base['seed']);temporal_candidate(Config(**base,epochs=23))
 seed_all(base['seed']);image_candidate(Config(**base,epochs=14))
 (OUT/'README.txt').write_text('Fixed final candidates only. No validation or holdout data was loaded or evaluated.\n',encoding='utf-8')
 print(f'prepared {OUT}',flush=True)
if __name__=='__main__':main()
