import argparse
from pathlib import Path
import yaml
from models import VARIANTS,build_model
import torch

def arguments(description,weights=False):
    p=argparse.ArgumentParser(description=description)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--variant',default='cfpg')
    p.add_argument('--output',type=Path,default=Path('results'))
    if weights:p.add_argument('--weights',type=Path,required=True)
    a=p.parse_args();cfg=yaml.safe_load(a.config.read_text())
    if a.data_root:cfg['dataset']['root']=str(a.data_root)
    cfg['variant']=a.variant
    a.output.mkdir(parents=True,exist_ok=True)
    return a,cfg

def load_model(variant,weights):
    model=build_model(variant,False)
    state=torch.load(weights,map_location='cpu')
    for key in ['state_dict','model_state_dict','model']:
        if isinstance(state,dict) and isinstance(state.get(key),dict):state=state[key];break
    target=model.state_dict();loaded={}
    for key,value in state.items():
        keys=[key,key.removeprefix('module.'),key.removeprefix('module.').removeprefix('model.')]
        found=next((k for k in keys if k in target and target[k].shape==value.shape),None)
        if found is not None:loaded[found]=value
    missing=[k for k in target if k not in loaded and not k.endswith('num_batches_tracked')]
    if missing:raise ValueError('Incomplete inference weights: '+str(missing[:8]))
    target.update(loaded);model.load_state_dict(target,strict=True)
    return model.eval()
