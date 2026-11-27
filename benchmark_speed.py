import argparse,json
from pathlib import Path
import torch
from models import build_model
from metrics.evaluation import latency
from utils.profiling import profile_shared_modules,capture_graph,graph_latency

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='cfpg');p.add_argument('--output',type=Path,default=Path('benchmark.json'));a=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('Formal timing requires CUDA')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=build_model(a.model,False).cuda().float().eval();sample=torch.randn(1,3,512,512,device='cuda')
    with torch.inference_mode():macs=profile_shared_modules(model,sample)
    eager=latency(model,sample,100,1000);graph,out=capture_graph(model,sample);captured=graph_latency(graph)
    result=dict(model=a.model,Params_M=sum(p.numel() for p in model.parameters())/1e6,GMACs=macs/1e9,GFLOPs=2*macs/1e9,eager=eager,cuda_graph=captured,device=torch.cuda.get_device_name(),dtype='FP32',warmup=100,iterations=1000)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
