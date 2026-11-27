from pathlib import Path
import math
import numpy as np
import torch
from PIL import Image
import cv2
from skimage.morphology import skeletonize as skimage_skeletonize
EPS=1e-6
PIDNET_ROOT=Path('.')
IMAGENET_MEAN=np.array([.485,.456,.406],dtype=np.float32)
IMAGENET_STD=np.array([.229,.224,.225],dtype=np.float32)


def path_from_config(value, base=PIDNET_ROOT):
    p = Path(str(value))
    return p if p.is_absolute() else base / p

def find_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return paths[0]

def sample_paths(dataset_cfg, sample_id):
    name = str(dataset_cfg['name']).lower().replace('-', '_').replace(' ', '_')
    root = path_from_config(dataset_cfg['root'])
    if name == 'neu_seg':
        stem = sample_id
        image = root / dataset_cfg.get('image_dir', 'images/training') / f'{stem}.jpg'
        mask = root / dataset_cfg.get('mask_dir', 'annotations/training') / f'{stem}.png'
        return (image, mask, False)
    if name == 'kolektorsdd2':
        stem = sample_id
        candidates = []
        for sub in [dataset_cfg.get('test_dir', 'test'), dataset_cfg.get('train_dir', 'train')]:
            candidates.append(root / sub / f'{stem}.png')
        image = find_existing(candidates)
        mask = image.parent / f'{stem}_GT.png'
        return (image, mask, False)
    if name == 'magnetic_tile_defect':
        (cls, stem) = sample_id.replace('\\', '/').split('/', 1)
        folder = root / cls / dataset_cfg.get('image_subdir', 'Imgs')
        image_candidates = [folder / f'{stem}{ext}' for ext in ['.jpg', '.jpeg', '.bmp', '.png', '.JPG', '.JPEG', '.BMP', '.PNG']]
        image = find_existing(image_candidates)
        mask = folder / f'{stem}.png'
        allow_missing_zero = cls == dataset_cfg.get('free_class', 'MT_Free')
        return (image, mask, allow_missing_zero)
    raise ValueError('Unsupported dataset: {}'.format(dataset_cfg['name']))

def preprocess_image(image):
    resized = image.resize((512, 512), Image.BILINEAR)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)

def load_image_mask(dataset_cfg, sample_id):
    (image_path, mask_path, allow_missing_zero) = sample_paths(dataset_cfg, sample_id)
    if not image_path.exists():
        raise FileNotFoundError('missing image: {}'.format(image_path))
    image = Image.open(image_path).convert('RGB')
    if mask_path.exists():
        mask = Image.open(mask_path).convert('L')
        mask_np = (np.asarray(mask) > 0).astype(np.uint8)
    elif allow_missing_zero:
        mask_np = np.zeros((image.height, image.width), dtype=np.uint8)
    else:
        raise FileNotFoundError('missing mask: {}'.format(mask_path))
    return (image, mask_np, str(image_path))

def update_confusion(cm, pred, target):
    pred = pred.astype(np.int64).ravel()
    target = target.astype(np.int64).ravel()
    valid = (target >= 0) & (target < 2)
    hist = np.bincount(2 * target[valid] + pred[valid], minlength=4).reshape(2, 2)
    cm += hist

def seg_metrics(cm):
    cm = cm.astype(np.float64)
    iou = np.diag(cm) / (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm) + EPS)
    acc = np.diag(cm) / (cm.sum(axis=1) + EPS)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return {'mIoU': float((iou[0] + iou[1]) / 2), 'background_IoU': float(iou[0]), 'defect_IoU': float(iou[1]), 'Dice': float(f1), 'F1': float(f1), 'Precision': float(precision), 'Recall': float(recall), 'Pixel_Accuracy': float((tp + tn) / (cm.sum() + EPS)), 'mAcc': float((acc[0] + acc[1]) / 2)}

def skeleton(mask):
    mask_bool = mask.astype(bool)
    if skimage_skeletonize is not None:
        return (skimage_skeletonize(mask_bool).astype(np.uint8), 'skimage')
    if cv2 is not None and hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
        thinned = cv2.ximgproc.thinning(mask_bool.astype(np.uint8) * 255)
        return ((thinned > 0).astype(np.uint8), 'cv2.ximgproc')
    return (None, None)

def component_count(mask, min_area=5):
    if cv2 is None:
        return None
    (num, labels, stats, _) = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    count = 0
    for idx in range(1, num):
        if stats[idx, cv2.CC_STAT_AREA] >= min_area:
            count += 1
    return count

def boundary(mask):
    if cv2 is None:
        return None
    m = mask.astype(np.uint8)
    if m.max() == 0:
        return np.zeros_like(m, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(m, kernel, iterations=1)
    return (m - eroded > 0).astype(np.uint8)

def centroid(mask):
    (ys, xs) = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)

def perimeter(mask):
    b = boundary(mask)
    if b is None:
        return float(mask.sum())
    return float(b.sum())

def boundary_distance_error(pred, gt, diag):
    if cv2 is None:
        return None
    pred_b = boundary(pred)
    gt_b = boundary(gt)
    if gt.sum() == 0 and pred.sum() == 0:
        return None
    if gt.sum() > 0 and pred.sum() == 0:
        return diag
    if gt.sum() == 0 and pred.sum() > 0:
        return None
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return diag
    dist_to_gt = cv2.distanceTransform((1 - gt_b).astype(np.uint8), cv2.DIST_L2, 3)
    dist_to_pred = cv2.distanceTransform((1 - pred_b).astype(np.uint8), cv2.DIST_L2, 3)
    return float(0.5 * dist_to_gt[pred_b > 0].mean() + 0.5 * dist_to_pred[gt_b > 0].mean())

def rankdata(values):
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(arr, dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = rank
        i = j + 1
    return ranks

def spearman(x, y):
    if len(x) < 2:
        return '-'
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) < EPS or np.std(ry) < EPS:
        return '-'
    return float(np.corrcoef(rx, ry)[0, 1])

def grade(score, q1, q2):
    if score <= q1:
        return 0
    if score <= q2:
        return 1
    return 2
