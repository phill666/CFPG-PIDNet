import csv, hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch
from utils.common import *
from utils.data import datasets,make_loader
from models import build_model

def build_neu_seg_model(cfg,augment):return build_model(cfg['variant'],augment)


def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)

def fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def save_epoch(path, model, optimizer, scaler, epoch, best, cfg, loaders):
    atomic_save(dict(model=state_dict_to_cpu(model.state_dict()), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), epoch=epoch, best=best, fingerprint=fingerprint(cfg), random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], generators=[loader.generator.get_state() for loader in loaders]), path)

def restore_epoch(path, model, optimizer, scaler, cfg, loaders):
    if not path.exists():
        return (0, -1.0)
    checkpoint = torch.load(path, map_location='cpu')
    if checkpoint['fingerprint'] != fingerprint(cfg):
        raise ValueError('Checkpoint protocol/configuration mismatch; use a fresh output directory.')
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    random.setstate(checkpoint['random'])
    np.random.set_state(checkpoint['numpy'])
    torch.set_rng_state(checkpoint['torch'])
    if checkpoint['cuda']:
        torch.cuda.set_rng_state_all(checkpoint['cuda'])
    for (loader, state) in zip(loaders, checkpoint['generators']):
        loader.generator.set_state(state)
    return (checkpoint['epoch'], checkpoint['best'])

def run(cfg,dataset):
    train, loss_cfg = cfg['train'], cfg['loss']
    if train['pretrained'] or train['optimizer'].lower() != 'sgd':
        raise ValueError('Industrial experiments require scratch SGD training.')
    if int(train['num_workers']) != 0:
        raise ValueError('The recoverable trainer currently uses num_workers=0 for reproducible RNG restoration.')
    set_seed(train['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Match the existing training environment; FP32 benchmark disables TF32 separately.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(cfg['output']['save_dir'])
    output.mkdir(parents=True, exist_ok=True)
    save_yaml_config(cfg, output/'config_used.yaml')
    _, data = datasets(cfg, dataset)
    train_loader = make_loader(data['train'], train['batch_size'], 0, True, device, train['seed'])
    val_loader = make_loader(data['val'], train['batch_size'], 0, False, device, train['seed'])
    model = build_neu_seg_model(cfg, augment=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=train['lr'], momentum=train['momentum'],
                                weight_decay=train['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=train['amp'] and device.type=='cuda')
    last = output/'last.pth'
    completed, best = restore_epoch(last, model, optimizer, scaler, cfg, [train_loader,val_loader])
    log_path, csv_path = output/'train_log.txt', output/'val_metrics.csv'
    # If interrupted between CSV append and checkpoint replace, discard only the
    # uncommitted CSV epoch; the checkpoint remains the source of truth.
    keys = ['epoch','lr','train_loss','mIoU','defect_IoU','Dice','Precision','Recall','best_defect_IoU']
    old_rows = []
    if csv_path.exists():
        with csv_path.open(encoding='utf-8',newline='') as f:
            old_rows = [r for r in csv.DictReader(f) if int(r['epoch']) <= completed]
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(old_rows)

    def log(value):
        print(value,flush=True)
        with log_path.open('a',encoding='utf-8') as f:
            f.write(value+'\n')

    log(f"dataset={dataset} variant={cfg['model']} seed={train['seed']} epoch_start={completed+1} "
        f"epochs={train['epochs']} batch={train['batch_size']} size={train['input_size']} workers=0 scratch=True")
    iterations = train['epochs'] * len(train_loader)
    for epoch in range(completed+1, train['epochs']+1):
        model.train()
        total_loss, count = 0., 0
        for i, (images, masks, _) in enumerate(train_loader):
            lr = poly_lr(train['lr'], (epoch-1)*len(train_loader)+i, iterations, train['power'])
            set_optimizer_lr(optimizer,lr)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(images.to(device,non_blocking=True))
                loss = ce_dice_loss(outputs,masks.to(device,non_blocking=True),
                                    loss_cfg['ce_weight'],loss_cfg['dice_weight'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch={epoch}, batch={i}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())*len(images)
            count += len(images)
        metrics = evaluate_model(model,val_loader,device,num_classes=2,amp=False)
        if metrics['defect_IoU'] > best:
            best = metrics['defect_IoU']
            atomic_save(state_dict_to_cpu(model.state_dict()),output/cfg['output']['save_best_name'])
        row = dict(epoch=epoch,lr=lr,train_loss=total_loss/count,
                   **{k:metrics[k] for k in ['mIoU','defect_IoU','Dice','Precision','Recall']},best_defect_IoU=best)
        with csv_path.open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=keys).writerow(row)
        save_epoch(last,model,optimizer,scaler,epoch,best,cfg,[train_loader,val_loader])
        log(' '.join(f'{k}={v:.6f}' if isinstance(v,float) else f'{k}={v}' for k,v in row.items()))
    log(f"Training done. {train['epochs']} epochs completed before validation calibration/test.")
