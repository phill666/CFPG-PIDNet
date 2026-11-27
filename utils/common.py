import random
import numpy as np
import torch
import torch.nn.functional as F

def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def state_dict_to_cpu(state_dict):
    return {key: value.detach().cpu() if torch.is_tensor(value) else value for (key, value) in state_dict.items()}

def get_main_output(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) >= 2 and torch.is_tensor(outputs[-1]) and (outputs[-1].dim() == 4):
            if outputs[-1].shape[1] == 1:
                return outputs[-2]
        return outputs[-1]
    return outputs

def get_semantic_outputs(outputs):
    if not isinstance(outputs, (list, tuple)):
        return [outputs]
    outputs = list(outputs)
    if len(outputs) >= 2 and torch.is_tensor(outputs[-1]) and (outputs[-1].dim() == 4):
        if outputs[-1].shape[1] == 1:
            return outputs[:-1]
    return outputs

def defect_dice_loss(logits, target, eps=1e-06):
    probs = torch.softmax(logits.float(), dim=1)[:, 1]
    target_fg = (target == 1).float()
    intersection = (probs * target_fg).sum(dim=(1, 2))
    denominator = probs.sum(dim=(1, 2)) + target_fg.sum(dim=(1, 2))
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()

def ce_dice_loss(outputs, target, ce_weight=1.0, dice_weight=1.0, class_weights=None):
    semantic_outputs = get_semantic_outputs(outputs)
    if len(semantic_outputs) == 2:
        aux_weights = [0.4, 1.0]
    else:
        aux_weights = [1.0 for _ in semantic_outputs]
    total = target.new_tensor(0.0, dtype=torch.float32)
    size = target.shape[-2:]
    for (weight, logits) in zip(aux_weights, semantic_outputs):
        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=True)
        ce_class_weights = None
        if class_weights is not None:
            ce_class_weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
        ce = F.cross_entropy(logits, target, weight=ce_class_weights)
        dice = defect_dice_loss(logits, target)
        total = total + float(weight) * (float(ce_weight) * ce + float(dice_weight) * dice)
    return total

def poly_lr(base_lr, current_iter, max_iter, power):
    if max_iter <= 0:
        return float(base_lr)
    factor = max(0.0, 1.0 - float(current_iter) / float(max_iter))
    return float(base_lr) * factor ** float(power)

def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

def update_confusion_matrix(confusion_matrix, pred, target, num_classes):
    pred = pred.detach().view(-1).to(torch.int64)
    target = target.detach().view(-1).to(torch.int64)
    valid = (target >= 0) & (target < int(num_classes))
    if valid.sum().item() == 0:
        return confusion_matrix
    index = target[valid] * int(num_classes) + pred[valid]
    bins = torch.bincount(index.cpu(), minlength=int(num_classes) ** 2)
    confusion_matrix += bins.numpy().reshape(int(num_classes), int(num_classes))
    return confusion_matrix

def metrics_from_confusion_matrix(confusion_matrix, eps=1e-06):
    cm = confusion_matrix.astype(np.float64)
    tp0 = cm[0, 0]
    tp1 = cm[1, 1]
    fp1 = cm[0, 1]
    fn1 = cm[1, 0]
    tn = tp0
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    iou = np.diag(cm) / (row_sum + col_sum - np.diag(cm) + eps)
    precision = tp1 / (tp1 + fp1 + eps)
    recall = tp1 / (tp1 + fn1 + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    dice = 2.0 * tp1 / (2.0 * tp1 + fp1 + fn1 + eps)
    pixel_acc = (tp0 + tp1) / (cm.sum() + eps)
    return {'mIoU': float((iou[0] + iou[1]) / 2.0), 'background_IoU': float(iou[0]), 'defect_IoU': float(iou[1]), 'Dice': float(dice), 'F1': float(f1), 'Precision': float(precision), 'Recall': float(recall), 'Pixel_Accuracy': float(pixel_acc), 'TP': float(tp1), 'FP': float(fp1), 'TN': float(tn), 'FN': float(fn1)}

@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes, amp=False):
    model.eval()
    confusion_matrix = np.zeros((int(num_classes), int(num_classes)), dtype=np.float64)
    use_amp = bool(amp) and device.type == 'cuda'
    for (images, masks, _) in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            logits = get_main_output(outputs)
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=True)
        pred = torch.argmax(logits, dim=1)
        confusion_matrix = update_confusion_matrix(confusion_matrix, pred, masks, num_classes)
    return metrics_from_confusion_matrix(confusion_matrix)


def save_yaml_config(cfg,path):
    import yaml
    from pathlib import Path
    Path(path).write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
