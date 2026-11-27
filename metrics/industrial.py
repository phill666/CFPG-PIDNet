"""ETMNet primitives with train-only severity calibration and GT-empty FPR metrics."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .evaluation import measurement_backend
from utils.common import get_main_output
from utils.io import atomic_text,sha256

NEW_METRICS = ['Image_FPR (%)','Pixel_FPR (%)','Mean_Pred_Area (%)',
               'Severity_Acc','Severity_MAE','Spearman_rho']


def mask_features(mask,backend):
    area = float(mask.sum())
    if not area:
        return dict(area=0.,length=0.,irregularity=0.)
    skeleton,_ = backend.skeleton(mask)
    if skeleton is None:
        raise RuntimeError('Skeleton backend is required for train-referenced severity.')
    perimeter = backend.perimeter(mask)
    return dict(area=area,length=float(skeleton.sum()),
                irregularity=perimeter**2/(4*math.pi*max(area,backend.EPS)))


def score(features,reference):
    normalized = []
    for key in ['area','length','irregularity']:
        lo,hi = reference['bounds'][key]
        normalized.append((features[key]-lo)/(hi-lo) if abs(hi-lo)>=1e-6 else 0.)
    return .5*normalized[0]+.3*normalized[1]+.2*normalized[2]


def fit_reference(records):
    if not records:
        raise ValueError('Training GT has no positive samples for severity calibration.')
    reference = dict(bounds={key:[min(r[key] for r in records),max(r[key] for r in records)]
                             for key in ['area','length','irregularity']},
                     weights=dict(area=.5,length=.3,irregularity=.2),
                     shape_definition='ETMNet irregularity=P^2/(4*pi*A); P=boundary pixel count',
                     normalization='positive training GT min-max; no prediction clipping',
                     Severity_MAE_definition='mean absolute difference of continuous severity scores, as ETMNet')
    scores = [score(row,reference) for row in records]
    reference['tertiles'] = np.quantile(scores,[1/3,2/3]).tolist()
    reference['positive_training_samples'] = len(records)
    return reference


def training_reference(cfg,train_path,output):
    output.mkdir(parents=True,exist_ok=True)
    path = output/'severity_train_reference.json'
    identity = dict(dataset=cfg['dataset']['name'],root=cfg['dataset']['root'],
                    training_split=str(train_path),training_split_sha256=sha256(train_path))
    if path.exists():
        reference = json.loads(path.read_text())
        if reference['source']!=identity:
            raise ValueError('Severity reference training split/root mismatch.')
        return reference
    backend = measurement_backend()
    records = []
    digest = hashlib.sha256()
    pairs = None
    if str(cfg['dataset']['name']).lower()=='kolektorsdd2':
        # Match the unchanged training loader: 10301 (copy).png is paired with
        # 10301_GT (copy).png, not the filename guessed by the old test-only CLI.
        from datasets.ksdd2_dataset import scan_ksdd2_split
        pairs,_ = scan_ksdd2_split(Path(cfg['dataset']['root'])/cfg['dataset'].get('train_dir','train'),'train')
    for sample in Path(train_path).read_text().splitlines():
        if not sample.strip():
            continue
        if pairs is not None:
            with Image.open(pairs[sample]['mask']) as image:
                mask = (np.asarray(image.convert('L'))>0).astype(np.uint8)
        else:
            _,mask,_ = backend.load_image_mask(cfg['dataset'],sample)
        digest.update(sample.encode())
        digest.update(np.asarray(mask.shape,dtype=np.int64).tobytes())
        digest.update(mask.tobytes())
        if mask.any():
            records.append(dict(sample=sample,**mask_features(mask,backend)))
    reference = fit_reference(records)
    reference.update(source=identity,training_gt_sha256=digest.hexdigest(),
                     backend_source_sha256=sha256(Path(__file__).with_name('native.py')))
    atomic_text(path,json.dumps(reference,indent=2))
    with (output/'severity_train_gt_features.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['sample','area','length','irregularity','severity_score','severity_grade'])
        writer.writeheader()
        for record in records:
            value = score(record,reference)
            writer.writerow(dict(record,severity_score=value,severity_grade=backend.grade(value,*reference['tertiles'])))
    return reference


def summarize_extra(records,reference,backend):
    negative = [r for r in records if not r['gt_area']]
    positive = [r for r in records if r['gt_area']]
    result = {'Image_FPR (%)':100*sum(r['pred_area']>0 for r in negative)/len(negative) if negative else None,
              'Pixel_FPR (%)':100*sum(r['pred_area'] for r in negative)/sum(r['pixels'] for r in negative) if negative else None,
              'Mean_Pred_Area (%)':100*float(np.mean([r['pred_area']/r['pixels'] for r in negative])) if negative else None,
              'gt_empty_samples':len(negative),'severity_positive_samples':len(positive)}
    if not positive:
        result.update(Severity_Acc=None,Severity_MAE=None,Spearman_rho=None)
        return result
    true = [r['gt_score'] for r in positive]
    predicted = [r['pred_score'] for r in positive]
    q1,q2 = reference['tertiles']
    result['Severity_Acc'] = float(np.mean([backend.grade(a,q1,q2)==backend.grade(b,q1,q2)
                                          for a,b in zip(true,predicted)]))
    result['Severity_MAE'] = float(np.mean(np.abs(np.asarray(true)-np.asarray(predicted))))
    rho = backend.spearman(true,predicted)
    result['Spearman_rho'] = float(rho) if isinstance(rho,(float,int,np.floating)) and np.isfinite(rho) else None
    return result


@torch.no_grad()
def evaluate_extra(model,cfg,test_path,thresholds,reference,device,output):
    backend = measurement_backend()
    model.eval()
    records = {mode:[] for mode in thresholds}
    cm = {mode:np.zeros((2,2),dtype=np.int64) for mode in thresholds}
    for sample in Path(test_path).read_text().splitlines():
        if not sample.strip():
            continue
        image,gt,_ = backend.load_image_mask(cfg['dataset'],sample)
        logits = get_main_output(model(backend.preprocess_image(image).to(device)))
        logits = F.interpolate(logits.float(),size=gt.shape,mode='bilinear',align_corners=True)
        probability = logits.softmax(1)[0,1].cpu().numpy()
        gt_area = int(gt.sum())
        gt_score = score(mask_features(gt,backend),reference) if gt_area else None
        # Identical thresholds share their mask/feature computation.
        predictions = {}
        for mode,threshold in thresholds.items():
            if threshold not in predictions:
                pred = (probability>=threshold).astype(np.uint8)
                predictions[threshold] = (pred,score(mask_features(pred,backend),reference) if gt_area else None)
            pred,pred_score = predictions[threshold]
            backend.update_confusion(cm[mode],pred,gt)
            records[mode].append(dict(sample=sample,gt_area=gt_area,pred_area=int(pred.sum()),pixels=gt.size,
                                      gt_score=gt_score,pred_score=pred_score))
    result = {mode:summarize_extra(rows,reference,backend) for mode,rows in records.items()}
    verification = {mode:backend.seg_metrics(matrix) for mode,matrix in cm.items()}
    output.mkdir(parents=True,exist_ok=True)
    with (output/'industrial_per_image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=['mode','sample','gt_area','pred_area','pixels','gt_score','pred_score'])
        writer.writeheader()
        for mode,rows in records.items():
            writer.writerows(dict(mode=mode,**row) for row in rows)
    atomic_text(output/'industrial_metrics.json',json.dumps(dict(metrics=result,
        recomputed_segmentation=verification,reference_source=reference['source'],
        gt_empty_definition='any foreground pixel flags an image; denominators are GT-empty images/pixels only',
        severity_reference='positive training GT only; frozen bounds and tertile cutpoints'),indent=2,allow_nan=False))
    return result,verification
