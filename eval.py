import copy,json
import torch
from utils.cli import arguments,load_model
from utils.data import datasets
from metrics.evaluation import calibrate,evaluate_thresholds
from metrics.industrial import training_reference,evaluate_extra

def main():
    a,cfg=arguments('Native-resolution validation calibration and test evaluation',True)
    paths,_=datasets(cfg,a.config.stem)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_model(a.variant,a.weights).to(device)
    thresholds=dict(Argmax=.5,**calibrate(model,cfg,paths['val'],device,a.output))
    results=evaluate_thresholds(model,cfg,paths['test'],sorted(set(thresholds.values())),device)
    reference=training_reference(cfg,paths['train'],a.output/'severity_reference')
    extra,_=evaluate_extra(model,cfg,paths['test'],thresholds,reference,device,a.output)
    output={mode:dict(results[t],**extra[mode]) for mode,t in thresholds.items()}
    (a.output/'metrics.json').write_text(json.dumps(dict(thresholds=thresholds,test=output),indent=2,allow_nan=False))
if __name__=='__main__':main()
