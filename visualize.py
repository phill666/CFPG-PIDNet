"""Publication layouts only: fixed specimens, checkpoints and full-image metrics."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from utils.visualization_data import entry_for,records_for,native_image,prediction,rotate,load,folder_for,prepare_manifest,upsample,normalize
from metrics.evaluation import measurement_backend
from utils.io import atomic_text,sha256

SOURCE=Path('configs')
DEST=Path('results/figures')
QA=DEST/'qa'
W,H=360,270
DPI=300
CELL_W,CELL_H=2.16,1.62
COLORS=['#00C853','#FF9100','#2979FF','#FF1744']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'pdf.fonttype':42,
                     'ps.fonttype':42,'figure.facecolor':'white','savefig.facecolor':'white'})
SUCCESS_LABELS=['KSDD2\n(#20884)','KSDD2\n(#20236)','MTD\n(Uneven)',
                'MTD\n(Blowhole)','NEU Seg\n(#001624)','NEU Seg\n(#003359)']
FAILURE_LABELS=['Mode A:\nLow-Contrast Miss\nKSDD2 (#20744)',
                'Mode B:\nBoundary Outlier\nMTD (Break)',
                'Mode C:\nTexture-Isomorph FP\nNEU Seg (#003732)']


def framing(gt,predictions):
    union=gt.astype(bool).copy()
    for pred in predictions: union|=pred
    yy,xx=np.where(union)
    height,width=gt.shape
    if len(xx):
        center=np.array([(xx.min()+xx.max()+1)/2,(yy.min()+yy.max()+1)/2],float)
        desired_w=max((xx.max()-xx.min()+1)*1.35,(yy.max()-yy.min()+1)*1.35*W/H)
    else:
        center=np.array([width/2,height/2]);desired_w=width
    # Near-square specimens retain almost their entire field. Extreme images
    # use a context-preserving 4:3 view centred as closely as feasible on union.
    maximum_w=min(width,height*W/H)
    minimum_w=.72*maximum_w
    crop_w=int(round(np.clip(desired_w,minimum_w,maximum_w)/4))*4
    crop_w=max(4,min(crop_w,int(maximum_w)//4*4))
    crop_h=crop_w*3//4
    x0=int(round(center[0]-crop_w/2));y0=int(round(center[1]-crop_h/2))
    x0=int(np.clip(x0,0,width-crop_w));y0=int(np.clip(y0,0,height-crop_h))
    original=[x0,y0,x0+crop_w,y0+crop_h]
    # Keep GT in the display if an outlying FP pulls the union centre away.
    # This is the nearest feasible crop to the union anchor, not a metric crop.
    gy,gx=np.where(gt)
    protected=False
    if len(gx):
        previous=(x0,y0)
        if gx.max()-gx.min()+1<=crop_w:
            low=max(0,int(gx.max()+1-crop_w));high=min(width-crop_w,int(gx.min()))
            x0=int(np.clip(x0,low,high))
        if gy.max()-gy.min()+1<=crop_h:
            low=max(0,int(gy.max()+1-crop_h));high=min(height-crop_h,int(gy.min()))
            y0=int(np.clip(y0,low,high))
        protected=(x0,y0)!=previous
    if len(gx) and not gt[y0:y0+crop_h,x0:x0+crop_w].any():
        cx=(gx.min()+gx.max()+1)/2;cy=(gy.min()+gy.max()+1)/2
        x0=int(np.clip(x0,max(0,int(cx-crop_w+1)),min(width-crop_w,int(cx))))
        y0=int(np.clip(y0,max(0,int(cy-crop_h+1)),min(height-crop_h,int(cy))))
        protected=True
    box=[x0,y0,x0+crop_w,y0+crop_h]
    assert (box[2]-box[0])*3==(box[3]-box[1])*4
    return box,dict(union_center=center.tolist(),nominal_box=original,display_box=box,
                    GT_visibility_guard=protected,retained_GT_fraction=float(gt[y0:y0+crop_h,x0:x0+crop_w].sum()/max(gt.sum(),1)))


def crop(value,box,mask=False):
    x0,y0,x1,y1=box
    result=cv2.resize(np.ascontiguousarray(value[y0:y1,x0:x1]),(W,H),
                      interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR)
    assert result.shape[:2]==(H,W)
    return result


def prepare_rows(entries,rows):
    prepared=[]
    for row in rows:
        image,gt=native_image(row)
        names=[row['competitor'],'PIDNet','CFPG-PIDNet']
        predictions=[prediction(entries,row,name) for name in names]
        displayed_gt=rotate(gt,row)
        displayed_predictions=[rotate(value,row) for value in predictions]
        box,audit=framing(displayed_gt,displayed_predictions)
        prepared.append(dict(row=row,native_image=image,native_gt=gt,
            image=rotate(np.asarray(image),row),gt=displayed_gt,predictions=displayed_predictions,
            crop=box,crop_audit=audit,
            metrics={name:records_for(entries,row['dataset'],name)[row['sample']] for name in names}))
    return prepared


def canvas(count,mechanism=False):
    gap=.14 if mechanism else .22
    left,right,top,bottom=1.85,.20,.53,.68
    grid_w=6*CELL_W+5*CELL_W*.05
    grid_h=count*CELL_H+(count-1)*CELL_H*gap
    fw,fh=left+grid_w+right,top+grid_h+bottom
    fig=plt.figure(figsize=(fw,fh),dpi=DPI,facecolor='white')
    axes=[]
    for i in range(count):
        row=[]
        for j in range(6):
            x=left+j*CELL_W*1.05;y=fh-top-CELL_H-i*CELL_H*(1+gap)
            ax=fig.add_axes([x/fw,y/fh,CELL_W/fw,CELL_H/fh])
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True);spine.set_color('#333333');spine.set_linewidth(.7)
            row.append(ax)
        axes.append(row)
    return fig,axes


def headings(fig,axes,titles,labels):
    for j,title in enumerate(titles):
        box=axes[0][j].get_position()
        fig.text((box.x0+box.x1)/2,box.y1+.012,title,ha='center',va='bottom',fontsize=10.5,fontweight='bold')
    for i,label in enumerate(labels):
        box=axes[i][0].get_position()
        fig.text(.91/fig.get_figwidth(),(box.y0+box.y1)/2,label,ha='center',va='center',
                 fontsize=9.5,fontweight='bold',linespacing=1.35)


def annotate(ax,text,bold=False):
    ax.text(.5,-.035,text,transform=ax.transAxes,ha='center',va='top',fontsize=8.5,
            fontweight='bold' if bold else 'normal',color='#66101C' if bold else '#111111',linespacing=1.15)


def outlines(ax,mask,color):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points=contour[:,0,:]
        if len(points)>1:
            points=np.concatenate([points,points[:1]])
            ax.plot(points[:,0],points[:,1],color=color,lw=1.8*72/DPI,antialiased=True)


def save(fig,name):
    png=DEST/(name+'.png');pdf=DEST/(name+'.pdf')
    fig.savefig(png,dpi=DPI,facecolor='white')
    fig.savefig(pdf,dpi=DPI,facecolor='white')
    fig.canvas.draw()
    boxes=[ax.get_window_extent() for ax in fig.axes if getattr(ax,'is_data_panel',False)]
    assert len({(round(b.width,4),round(b.height,4)) for b in boxes})==1,'Unequal publication cell sizes'
    plt.close(fig)
    return dict(png=str(png),pdf=str(pdf),cell_pixels=[round(boxes[0].width),round(boxes[0].height)])


def qualitative(prepared,labels,name):
    fig,axes=canvas(len(prepared))
    headings(fig,axes,['(a) Input & ROI','(b) Ground Truth','(c) Top Competitor',
                       '(d) PIDNet (Baseline)','(e) CFPG-PIDNet (Ours)','(f) CFPG Error Map'],labels)
    for i,item in enumerate(prepared):
        row=item['row'];box=item['crop']
        rgb=crop(item['image'],box)
        gt=crop(item['gt'],box,True).astype(bool)
        predictions=[crop(value.astype(np.uint8),box,True).astype(bool) for value in item['predictions']]
        for j in range(5): axes[i][j].imshow(rgb)
        outlines(axes[i][1],gt,COLORS[0])
        for j,(model,pred,color) in enumerate(zip([row['competitor'],'PIDNet','CFPG-PIDNet'],predictions,COLORS[1:]),2):
            outlines(axes[i][j],pred,color)
            metric=item['metrics'][model]
            annotate(axes[i][j],f"Dice: {metric['Dice']:.3f} | RAE: {metric['RAE']:.3f}\nBDE: {metric['BDE']:.2f} px",bold=j==4)
        badge={'U-Net':'U-Net','STDC-Seg':'STDC-Seg','PP-LiteSeg-STDC1':'PP-LiteSeg'}[row['competitor']]
        axes[i][2].text(.025,.975,badge,transform=axes[i][2].transAxes,va='top',ha='left',
            fontsize=8,fontweight='bold',bbox=dict(boxstyle='round,pad=.24',fc='white',ec='none',alpha=.82))
        error=np.zeros((*gt.shape,3),dtype=np.uint8);ours=predictions[-1]
        error[ours&gt]=255;error[ours&~gt]=[255,0,0];error[~ours&gt]=[0,0,255]
        axes[i][5].imshow(error,interpolation='nearest')
        if row['sample'] in ['20236','MT_Blowhole/exp4_num_3724']:
            yy,xx=np.where(gt)
            if len(xx):
                cx,cy=(xx.min()+xx.max())/2,(yy.min()+yy.max())/2
                width=max(20,(xx.max()-xx.min()+1)*2);height=max(15,(yy.max()-yy.min()+1)*2)
                width=max(width,height*4/3);height=width*3/4
                x0=int(np.clip(cx-width/2,0,max(W-width,0)));y0=int(np.clip(cy-height/2,0,max(H-height,0)))
                x1=min(W,int(x0+width));y1=min(H,int(y0+height))
                axes[i][0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='#FFD600',lw=1.2))
                inset=axes[i][0].inset_axes([.64,.025,.34,.34])
                inset.imshow(rgb[y0:y1,x0:x1]);inset.set_xticks([]);inset.set_yticks([])
                for spine in inset.spines.values(): spine.set_color('#FFD600');spine.set_linewidth(1.2)
        for ax in axes[i]: ax.is_data_panel=True
    handles=[Line2D([],[],linestyle='none',label='Contours:')]
    handles += [Line2D([],[],color=color,lw=1.5,label=label) for color,label in zip(COLORS,['GT','Top Competitor','PIDNet','CFPG-PIDNet'])]
    handles += [Line2D([],[],linestyle='none',label='|  Error Map:')]
    handles += [Patch(facecolor=color,edgecolor='#333333',label=label) for color,label in [('white','TP'),('red','FP'),('blue','FN')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.007),ncol=9,frameon=False,fontsize=8.5,
               handlelength=1.7,columnspacing=1.0)
    return save(fig,name)


def fetch_mechanisms(entries,prepared):
    path=DEST/'locked_mechanism_tensors.npz'
    identity=DEST/'mechanism_tensor_identity.json'
    manifest=[dict(dataset=item['row']['dataset'],sample=item['row']['sample'],checkpoint_sha256=
        entry_for(entries,item['row']['dataset'],'CFPG-PIDNet')['checkpoint_sha256']) for item in prepared]
    if path.exists():
        assert json.loads(identity.read_text())==manifest
        with np.load(path) as cache: return {key:cache[key] for key in cache.files}
    result={};backend=measurement_backend()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    for i,item in enumerate(prepared):
        row=item['row'];model,_=load(entry_for(entries,row['dataset'],'CFPG-PIDNet'))
        model.cuda().eval();core=model.cfpg.cfpg;captured={}
        hooks=[core.coarse_gate.register_forward_pre_hook(lambda m,a:captured.update(freq=a[0].detach())),
            core.coarse_gate.register_forward_hook(lambda m,a,o:captured.update(coarse=o.detach())),
            core.refine_gate.register_forward_hook(lambda m,a,o:captured.update(refine=o.detach())),
            core.confidence_head.register_forward_pre_hook(lambda m,a:captured.update(entropy=a[0].detach())),
            core.confidence_head.register_forward_hook(lambda m,a,o:captured.update(ratio=o.detach()))]
        with torch.inference_mode():
            model(backend.preprocess_image(item['native_image']).cuda())
            coarse=captured['coarse'].float().softmax(1);ratio=captured['ratio']
            alpha=(1-ratio)*coarse+ratio*captured['refine'].float().softmax(1)
            alpha=alpha/(alpha.sum(1,keepdim=True)+1e-8)
            for key,value in [('freq',captured['freq']),('entropy',captured['entropy']),('ratio',ratio),('alpha',alpha)]:
                native=upsample(value,item['native_gt'].shape)
                result[f'{i}_{key}']=np.stack([rotate(channel,row) for channel in native])
        for hook in hooks: hook.remove()
        del model;torch.cuda.empty_cache()
    np.savez_compressed(path,**result);atomic_text(identity,json.dumps(manifest,indent=2))
    return result


def mechanism(entries,prepared):
    tensors=fetch_mechanisms(entries,prepared)
    fig,axes=canvas(6,mechanism=True)
    headings(fig,axes,['(a) Input RGB','(b) Ground Truth','(c) Frequency Prior (F)',
                       '(d) Shannon Entropy H(伪c)','(e) Feedback Ratio (r)','(f) Adaptive Scale (伪)'],SUCCESS_LABELS)
    for i,item in enumerate(prepared):
        box=item['crop']
        axes[i][0].imshow(crop(item['image'],box))
        axes[i][1].imshow(crop(item['gt'],box,True),cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        frequency=tensors[f'{i}_freq']
        raw_prior=np.mean([normalize(channel**2) for channel in frequency],axis=0)
        prior=crop(raw_prior,box)
        lo,hi=np.percentile(prior,[1,99])
        axes[i][2].imshow(prior,cmap='inferno',vmin=lo,vmax=max(hi,lo+1e-8))
        for j,key in [(3,'entropy'),(4,'ratio')]:
            value=crop(tensors[f'{i}_{key}'][0],box)
            lo,hi=np.percentile(value,[2,98])
            axes[i][j].imshow(value,cmap='viridis',vmin=lo,vmax=max(hi,lo+1e-8))
            annotate(axes[i][j],f'[{value.min():.3f}, {value.max():.3f}]')
        alpha=tensors[f'{i}_alpha']
        display=np.stack([crop(channel,box) for channel in alpha],axis=-1)
        display=display/np.maximum(display.max(2,keepdims=True),1e-8)
        axes[i][5].imshow(display)
        mean=alpha.mean(axis=(1,2))
        annotate(axes[i][5],'mean: ('+', '.join(f'{v:.2f}' for v in mean)+')')
        for ax in axes[i]: ax.is_data_panel=True
    for j,cmap in [(2,'inferno'),(3,'viridis'),(4,'viridis')]:
        box=axes[-1][j].get_position()
        cax=fig.add_axes([box.x0+.06*box.width,.025,box.width*.88,.0045])
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,1),cmap=cmap),cax=cax,orientation='horizontal')
        bar.set_ticks([0,1]);bar.set_ticklabels(['Low','High']);bar.ax.tick_params(labelsize=8,length=2,pad=2)
        bar.outline.set_linewidth(.5)
    box=axes[-1][5].get_position()
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [('red','R: k=7'),('green','G: k=15'),('blue','B: k=31')]],
        bbox_to_anchor=((box.x0+box.x1)/2,.014),loc='lower center',ncol=3,frameon=False,
        fontsize=8,handlelength=.8,columnspacing=.7)
    return save(fig,'fig4_cfpg_mechanism_pub')


def audit(entries,prepared):
    backend=measurement_backend()
    rows=[]
    for item in prepared:
        row=item['row']
        for name,metric in item['metrics'].items():
            entry=entry_for(entries,row['dataset'],name)
            folder=folder_for(entry)
            result=json.loads((folder/'locked_result.json').read_text())
            # Display predictions and reported numbers must identify the same
            # unchanged full-image sample and locked threshold.
            pred=prediction(entries,row,name);gt=item['native_gt'].astype(bool)
            tp=int((pred&gt).sum())
            dice=2*tp/max(int(pred.sum())+int(gt.sum()),1)
            rae=abs(int(pred.sum())-int(gt.sum()))/int(gt.sum())
            assert abs(dice-metric['Dice'])<1e-12 and abs(rae-metric['RAE'])<1e-12
            bde=backend.boundary_distance_error(pred.astype(np.uint8),gt.astype(np.uint8),np.hypot(*gt.shape))
            assert abs(bde-metric['BDE'])<1e-6,(row['sample'],name,bde,metric['BDE'])
            rows.append(dict(dataset=row['dataset'],sample=row['sample'],model=name,
                threshold=result['thresholds']['MeasurementOpt'],Dice=metric['Dice'],RAE=metric['RAE'],BDE=metric['BDE'],
                full_image_shape=list(gt.shape),crop_audit=item['crop_audit'],
                checkpoint_sha256=entry['checkpoint_sha256'],record_sha256=sha256(folder/'locked_per_image.json')))
    return rows


def main():
    global DEST,QA
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=Path('configs/visualization_manifest.json'));parser.add_argument('--output',type=Path,default=DEST);args=parser.parse_args()
    DEST=args.output;QA=DEST/'qa'
    DEST.mkdir(parents=True,exist_ok=True);QA.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    entries=prepare_manifest(args.manifest,DEST)
    selection_path=SOURCE/'figure_samples.json'
    before=sha256(selection_path)
    selection=json.loads(selection_path.read_text())
    expected=['20884','20236','MT_Uneven/exp1_num_129031','MT_Blowhole/exp4_num_3724','001624','003359']
    assert [r['sample'] for r in selection['successes']]==expected
    assert [r['sample'] for r in selection['failures']]==['20744','MT_Break/exp1_num_98080','003732']
    successes=prepare_rows(entries,selection['successes']);failures=prepare_rows(entries,selection['failures'])
    metrics=audit(entries,successes+failures)
    atomic_text(DEST/'full_image_metrics_locked.json',json.dumps(metrics,indent=2))
    outputs=[mechanism(entries,successes),qualitative(successes,SUCCESS_LABELS,'fig5_qualitative_success_pub'),
             qualitative(failures,FAILURE_LABELS,'fig6_failure_cases_pub')]
    assert sha256(selection_path)==before
    from PIL import Image
    for output in outputs:
        image=Image.open(output['png']);output['pixels']=list(image.size);output['dpi']=list(image.info['dpi'])
        assert image.info['dpi'][0]>299
        print(Path(output['png']).name,'pixels',image.size,'cell',output['cell_pixels'],'dpi',image.info['dpi'])
    print('Original full-image Dice / RAE / BDE, all 9 specimens and 3 models:')
    for r in metrics:
        print(f"{r['dataset']} {r['sample']} {r['model']} Dice={r['Dice']:.12f} RAE={r['RAE']:.12f} BDE={r['BDE']:.12f}")
    atomic_text(DEST/'publication_checks.json',json.dumps(dict(outputs=outputs,sample_list_unchanged=True,
        locked_full_image_metrics=metrics,display_size=[W,H],common_ratio='4:3',
        corrected_typo='exp1_num_58080 is absent; retained locked exp1_num_98080',
        display_crop_policy='nearest feasible 4:3 crop to union anchor; guard against GT completely disappearing',
        cropped_metrics_recomputed=False),indent=2))


if __name__=='__main__': main()
