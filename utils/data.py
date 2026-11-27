from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32
    random.seed(seed);np.random.seed(seed)

def make_loader(dataset,batch_size,workers,shuffle,device,seed=None):
    generator=torch.Generator().manual_seed(int(seed or 42))
    return DataLoader(dataset,batch_size=batch_size,num_workers=workers,shuffle=shuffle,
        pin_memory=device.type=='cuda',drop_last=False,generator=generator,worker_init_fn=seed_worker)

def datasets(cfg,name):
    ds=cfg['dataset'];paths={s:Path(ds['split_dir'])/ds[s+'_split'] for s in ['train','val','test']}
    for p in paths.values():
        if not p.is_file():raise FileNotFoundError(p)
    from datasets.ksdd2_dataset import KolektorSDD2Dataset
    from datasets.magnetic_tile_dataset import MagneticTileDataset
    from datasets.neu_seg_dataset import NEUSegDataset
    result={}
    for split,p in paths.items():
        kw=dict(split_file=p,split_name=split,input_size=512,is_train=split=='train')
        if name=='ksdd2':data=KolektorSDD2Dataset(ds['root'],ds['train_dir'],ds['test_dir'],**kw)
        elif name=='mtd':data=MagneticTileDataset(ds['root'],ds['classes'],ds.get('image_subdir','Imgs'),free_class=ds.get('free_class','MT_Free'),**kw)
        else:data=NEUSegDataset(ds['root'],ds['image_dir'],ds['mask_dir'],**kw)
        result[split]=data
    return paths,result
