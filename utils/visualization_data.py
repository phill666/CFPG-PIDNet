import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from metrics import native
from metrics.evaluation import geometry
from utils.common import get_main_output
from utils.cli import load_model
from utils.io import atomic_text,sha256
CONFIGS={};ENTRIES=[];MODELS={};ROWS={};CACHE=Path('results/figures/cache')

def prepare_manifest(path,output):
    global CACHE
    CACHE=output/'cache';CACHE.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(path).read_text())
    for d,info in manifest.items():
        CONFIGS[d]=yaml.safe_load(Path(info['config']).read_text())
        for name,model in info['models'].items():
            ENTRIES.append(dict(dataset=d,model=name,seed=42,checkpoint=model['weights'],
                checkpoint_sha256=sha256(Path(model['weights'])),variant=model['variant'],threshold=model['threshold']))
    return ENTRIES

def entry_for(entries,dataset,model):return next(r for r in entries if r['dataset']==dataset and r['model']==model)
def folder_for(entry):
    path=CACHE/entry['dataset']/entry['model']/'seed42';path.mkdir(parents=True,exist_ok=True);return path

def native_image(row):return native.load_image_mask(CONFIGS[row['dataset']]['dataset'],row['sample'])[:2]
def rotate(value,row):return np.rot90(value) if row['dataset']=='ksdd2' else value

def load(entry):return load_model(entry['variant'],Path(entry['checkpoint'])),{}

def prediction(entries,row,name):
    entry=entry_for(entries,row['dataset'],name);key=(row['dataset'],name)
    if key not in MODELS:MODELS[key]=load(entry)[0].cuda().eval()
    image,gt=native_image(row)
    with torch.inference_mode():
        logits=get_main_output(MODELS[key](native.preprocess_image(image).cuda()))
        logits=F.interpolate(logits.float(),size=gt.shape,mode='bilinear',align_corners=True)
        pred=logits.softmax(1)[0,1].cpu().numpy()>=entry['threshold']
    tp=int((pred & (gt>0)).sum())
    record=dict(sample=row['sample'],Dice=2*tp/max(int(pred.sum())+int(gt.sum()),1),
                **geometry(pred.astype(np.uint8),gt,native))
    ROWS.setdefault(key,{})[row['sample']]=record
    folder=folder_for(entry)
    atomic_text(folder/'locked_per_image.json',json.dumps(list(ROWS[key].values())))
    atomic_text(folder/'locked_result.json',json.dumps(dict(thresholds=dict(MeasurementOpt=entry['threshold']))))
    return pred

def records_for(entries,dataset,name):return ROWS[(dataset,name)]
def upsample(value,shape):return F.interpolate(value.float(),size=shape,mode='bilinear',align_corners=True)[0].cpu().numpy()
def normalize(value):
    low,high=np.percentile(value,[1,99]);return np.clip((value-low)/max(high-low,1e-8),0,1)
