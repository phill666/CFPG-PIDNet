import ast,json,re,subprocess,sys,shutil
from pathlib import Path
import torch
from models import build_model,VARIANTS
ROOT=Path(__file__).resolve().parent

def main():
    files=list(ROOT.rglob('*.py'))
    subprocess.run([sys.executable,'-m','py_compile',*[str(p) for p in files]],check=True)
    torch.set_num_threads(4);device='cuda' if torch.cuda.is_available() else 'cpu'
    expected={'baseline':0,'ctrl_conv':56537,'cfpg_periph_only':55060,'cfpg_plus_csa':58900,'cfpg_plus_entropy_no_grn':59248,'cfpg':59504}
    reports=[];base=sum(p.numel() for p in build_model('baseline').parameters())
    for variant in VARIANTS:
        torch.manual_seed(42);model=build_model(variant).to(device).eval()
        with torch.inference_mode():out=model(torch.randn(1,3,512,512,device=device))
        params=sum(p.numel() for p in model.parameters())
        assert out.shape==(1,2,64,64) and torch.isfinite(out).all()
        assert params-base==expected[variant]
        if variant=='cfpg':assert params==7683026
        reports.append(dict(variant=variant,parameters=params,increment=params-base,shape=list(out.shape)))
        del model
    for variant in ['unet','bisenetv2','ddrnet23slim','stdcseg','segformer_b0','ppliteseg_stdc1']:
        model=build_model(variant).to(device).eval()
        with torch.inference_mode():out=model(torch.randn(1,3,512,512,device=device))
        assert out.shape[1]==2 and torch.isfinite(out).all()
        del model
    for p in ROOT.rglob('__pycache__'):
        assert p.resolve().is_relative_to(ROOT);shutil.rmtree(p)
    forbidden={'.pth','.pt','.ckpt','.onnx','.engine','.bin','.zip','.tar','.gz','.docx','.xlsx','.pdf'}
    size=0
    for p in ROOT.rglob('*'):
        if not p.is_file() or '.git' in p.parts:continue
        assert p.suffix not in forbidden,p
        assert p.stat().st_size<5*1024**2,p
        size+=p.stat().st_size
        if p.suffix in ['.py','.yaml','.sh','.ps1']:
            text=p.read_text(encoding='utf-8-sig')
            values=[n.value for n in ast.walk(ast.parse(text)) if isinstance(n,ast.Constant) and isinstance(n.value,str)] if p.suffix=='.py' else [text]
            assert not any(re.search(r'\b[A-Za-z]:[\\/]',v) for v in values),p
            assert not re.search(r'(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_]{20,}',text),p
    assert size<20*1024**2
    print(json.dumps(dict(status='passed',python_files=len(files),total_bytes=size,models=reports),indent=2))
if __name__=='__main__':main()
