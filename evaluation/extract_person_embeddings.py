"""Extract fixed ResNet18 person-crop embeddings for the v2 train windows only."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from datetime import datetime, timedelta
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch
from torchvision.models import ResNet18_Weights, resnet18

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'results/evaluation/lstm_v2'; OUT=ROOT/'results/evaluation/lstm_v2_img'; EMB=OUT/'emb'
T,STEP=30,10; SIZE=224; PAD=.12
def key(group:str)->str: return hashlib.sha1(group.encode()).hexdigest()[:16]
def progress(line:str)->None:
 with (OUT/'PROGRESS.md').open('a',encoding='utf-8') as handle: handle.write(f'\n- {line}\n')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--limit',type=int,default=20); p.add_argument('--all',action='store_true'); p.add_argument('--batch',type=int,default=64); a=p.parse_args()
 meta=pd.read_csv(SRC/'train_meta.csv'); ann=pd.read_csv(SRC/'splits/train_annotations_holdout_excluded.csv')
 paths=ann[['video','group_id','video_path']].drop_duplicates(['video','group_id']); data=meta.merge(paths,on=['video','group_id'],how='left',validate='many_to_one')
 if data.video_path.isna().any(): raise ValueError('Missing video path in v2 split join')
 videos=list(data.groupby(['video','group_id','video_path'],sort=False)); total=len(videos)
 if not a.all: videos=videos[:a.limit]
 EMB.mkdir(parents=True,exist_ok=True); OUT.mkdir(parents=True,exist_ok=True)
 device='cuda' if torch.cuda.is_available() else 'cpu'; torch.hub.set_dir(str(OUT/'torch_cache')); weights=ResNet18_Weights.IMAGENET1K_V1
 model=resnet18(weights=weights); model.fc=torch.nn.Identity(); model.eval().to(device); norm=weights.transforms(antialias=True)
 total_slots=total_present=0; began=time.perf_counter(); index=[]; completed=0; estimate_per_video=16.3568
 if a.all: progress(f'Full extraction started: PID {os.getpid()}, target={total} train videos, 224px, 30 slots at stride 10, batch={a.batch}. Existing valid NPZ files are skipped.')
 for index_no,((video,group,path),rows) in enumerate(videos,1):
  file=EMB/f'{key(group)}.npz'; index.append({'video':video,'group_id':group,'video_path':path,'embedding_file':file.name})
  if file.exists():
   completed+=1
   continue
  needed=sorted({int(start)+slot*STEP for start in rows.start_frame for slot in range(T)})
  track_path=ROOT/'results/tracking/train'/f'{Path(video).stem}.csv'; tr=pd.read_csv(track_path) if track_path.exists() else pd.DataFrame()
  chosen={}
  if {'frame','x1','y1','x2','y2'}.issubset(tr):
   for frame,g in tr[tr.frame.isin(needed)].groupby('frame'):
    g=g.assign(area=(g.x2-g.x1)*(g.y2-g.y1)); chosen[int(frame)]=g.loc[g.area.idxmax()]
  cap=cv2.VideoCapture(path); crops=[]; crop_frames=[]; embs={}; wanted=set(chosen)
  frame=-1
  while wanted:
   ok,img=cap.read()
   if not ok: break
   frame+=1
   if frame not in wanted: continue
   row=chosen[frame]; h,w=img.shape[:2]; bw=row.x2-row.x1; bh=row.y2-row.y1
   x1=max(0,int(row.x1-bw*PAD)); y1=max(0,int(row.y1-bh*PAD)); x2=min(w,int(row.x2+bw*PAD)); y2=min(h,int(row.y2+bh*PAD)); crop=img[y1:y2,x1:x2]
   wanted.remove(frame)
   if crop.size: crops.append(torch.from_numpy(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).permute(2,0,1)); crop_frames.append(frame)
   if len(crops)>=a.batch:
    batch=torch.stack([norm(x) for x in crops]).to(device)
    with torch.no_grad(),torch.autocast(device_type='cuda',enabled=device=='cuda'): out=model(batch).float().cpu().numpy()
    embs.update(dict(zip(crop_frames,out))); crops=[]; crop_frames=[]
  cap.release()
  if crops:
   batch=torch.stack([norm(x) for x in crops]).to(device)
   with torch.no_grad(),torch.autocast(device_type='cuda',enabled=device=='cuda'): out=model(batch).float().cpu().numpy()
   embs.update(dict(zip(crop_frames,out)))
  vectors=np.stack([embs.get(f,np.zeros(512,np.float32)) for f in needed]).astype(np.float32); present=np.array([f in embs for f in needed],np.uint8)
  np.savez_compressed(file,frames=np.asarray(needed,np.int32),embeddings=vectors,present=present)
  total_slots+=len(needed); total_present+=present.sum(); completed+=1; print(f'{completed}/{len(videos)} {video}: {present.sum()}/{len(needed)}',flush=True)
  if a.all and completed % 100 == 0:
   eta=datetime.now()+timedelta(seconds=(len(videos)-completed)*estimate_per_video)
   progress(f'Embedding progress: {completed}/{len(videos)} videos; estimated completion {eta.isoformat(timespec="minutes")} (based on 16.36 s/video probe).')
 pd.DataFrame(index).drop_duplicates().to_csv(OUT/'embedding_index.csv',index=False,encoding='utf-8-sig')
 elapsed=time.perf_counter()-began; (OUT/('extract_full.json' if a.all else 'extract_probe.json')).write_text(json.dumps({'videos':len(videos),'seconds':elapsed,'slots':total_slots,'present':int(total_present),'model':'resnet18_imagenet1k_v1','embedding_dim':512,'size':SIZE,'padding':PAD},indent=2),encoding='utf-8')
 if a.all: progress(f'Full extraction completed: {completed}/{len(videos)} videos, elapsed {elapsed/3600:.2f} hours.')
if __name__=='__main__': main()
