import argparse,copy
from pathlib import Path
import yaml
from models import VARIANTS
from utils.trainer import run

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path)
    p.add_argument('--variant',choices=VARIANTS,default='cfpg')
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--output',type=Path,default=Path('results'))
    a=p.parse_args()
    for seed in a.seeds:
        cfg=yaml.safe_load(a.config.read_text());cfg['variant']=a.variant
        if a.data_root:cfg['dataset']['root']=str(a.data_root)
        cfg['train'].update(seed=seed,optimizer='SGD',pretrained=False,num_workers=0)
        folder=a.output/a.config.stem/f'seed{seed}'/a.variant
        cfg['output']=dict(save_dir=str(folder),save_best_name='best.pth')
        run(cfg,a.config.stem)
if __name__=='__main__':main()
