import argparse,json
from pathlib import Path
import numpy as np
from metrics.bootstrap import bootstrap_arrays

def main():
    p=argparse.ArgumentParser(description='Paired positive-image bootstrap, with optional seed hierarchy')
    p.add_argument('--ours',type=Path,required=True);p.add_argument('--other',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('bootstrap.json'));a=p.parse_args()
    first=json.loads(a.ours.read_text());second=json.loads(a.other.read_text())
    assert first['seeds']==second['seeds'] and first['sample_ids']==second['sample_ids']
    # values: [seed, positive image, TP/FP/FN/RAE/LE/BDE/CE].
    point,lo,hi=bootstrap_arrays(np.asarray(first['values'],float),np.asarray(second['values'],float),10000)
    a.output.write_text(json.dumps({m:dict(difference=float(x),ci95_low=float(l),ci95_high=float(h)) for m,x,l,h in zip(['dIoU','RAE','LE','BDE','CE'],point,lo,hi)},indent=2))
if __name__=='__main__':main()
