"""Native-resolution ETMNet measurement protocol and CUDA-event profiling."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.common import get_main_output


def measurement_backend():
    from . import native
    return native


def geometry(pred, gt, backend):
    if not gt.any():
        return None
    area = float(gt.sum())
    diag = math.hypot(*gt.shape)
    ps, _ = backend.skeleton(pred)
    gs, _ = backend.skeleton(gt)
    cp, cg = backend.centroid(pred), backend.centroid(gt)
    return dict(RAE=abs(float(pred.sum()) - area) / area,
                LE=abs(float(ps.sum()) - float(gs.sum())) / max(float(gs.sum()), 1e-6),
                BDE=backend.boundary_distance_error(pred, gt, diag),
                CE=float(np.linalg.norm(cp - cg)) if cp is not None else diag,
                ConnE=abs(backend.component_count(pred) - backend.component_count(gt)))


@torch.no_grad()
def evaluate_thresholds(model, cfg, split_path, thresholds, device):
    backend = measurement_backend()
    cms = {t: np.zeros((2, 2), dtype=np.int64) for t in thresholds}
    measurements = {t: [] for t in thresholds}
    empty = {t: 0 for t in thresholds}
    model.eval()
    samples = Path(split_path).read_text(encoding='utf-8').splitlines()
    for sample in samples:
        if not sample.strip():
            continue
        image, gt, _ = backend.load_image_mask(cfg['dataset'], sample.strip())
        logits = get_main_output(model(backend.preprocess_image(image).to(device)))
        logits = F.interpolate(logits.float(), size=gt.shape, mode='bilinear', align_corners=True)
        probability = logits.softmax(1)[0, 1].cpu().numpy()
        for threshold in thresholds:
            pred = (probability >= threshold).astype(np.uint8)
            backend.update_confusion(cms[threshold], pred, gt)
            record = geometry(pred, gt, backend)
            if record is not None:
                measurements[threshold].append(record)
                empty[threshold] += int(not pred.any())
    results = {}
    for threshold in thresholds:
        records = measurements[threshold]
        result = backend.seg_metrics(cms[threshold])
        result['dIoU'] = result['defect_IoU']
        result.update({key: float(np.mean([r[key] for r in records])) if records else None
                       for key in ('RAE', 'LE', 'BDE', 'CE', 'ConnE')})
        result['Empty_Prediction_Rate'] = empty[threshold] / len(records) if records else None
        result['positive_samples'] = len(records)
        result['samples'] = len(samples)
        results[threshold] = result
    return results


def choose_thresholds(results):
    dice = min(results, key=lambda t: (-results[t]['Dice'], abs(t - .5), t))
    # Same rule as PIDNet-main/tools/ftm_search_threshold_pidnet.py:
    # retain >=95% of the best validation Dice, then minimize RAE+BDE+CE.
    valid = [t for t in results if results[t]['RAE'] is not None
             and results[t]['Dice'] >= .95 * results[dice]['Dice']]
    measurement = min(valid, key=lambda t: (results[t]['MeasurementScore'], -results[t]['Dice'], abs(t-.5))) if valid else dice
    return dict(DiceOpt=dice, MeasurementOpt=measurement)


def calibrate(model, cfg, val_path, device, out_dir):
    thresholds = [.30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80]
    results = evaluate_thresholds(model, cfg, val_path, thresholds, device)
    for r in results.values():
        r['MeasurementScore'] = r['RAE'] + r['BDE'] + r['CE'] if r['RAE'] is not None else None
    selected = choose_thresholds(results)
    (out_dir / 'calibration.json').write_text(json.dumps(dict(
        selected=selected, validation=results,
        rule='MeasurementOpt: Dice>=0.95*best Dice, minimize RAE+BDE+CE (ETMNet rule)'), indent=2), encoding='utf-8')
    return selected


@torch.inference_mode()
def latency(module, sample, warmup=100, iterations=1000):
    module.eval().float()
    for _ in range(warmup):
        module(sample)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        module(sample)
    end.record()
    end.synchronize()
    ms = start.elapsed_time(end) / iterations
    return dict(Latency_ms=ms, FPS=1000 / ms)
