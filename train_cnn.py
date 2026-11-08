"""AI02 CNN pretrained transfer learning.

Holdout: Reuses V4 split_manifest.csv if present; validation never used for
hyperparameter selection, early stopping, or temperature tuning.
Usage: python train_cnn.py --images data/dataset-resized --arch mobilenet_v3_large
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from collections import Counter

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, log_loss

from cnn_utils import (LABELS, ARCHES, build_model, train_transform,
                       val_transform, pil_rgb, checkpoint_name, save_checkpoint)

SEED = 2026
EXTS = {'.jpg','.jpeg','.png','.webp','.bmp'}


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_images(directory):
    """Decode/pixel-hash images and remove exact duplicates, label conflicts.

    Duplicate handling matches the V4 baseline: do NOT use final-validation
    labels to select or tune the model.
    """
    from src.quality import audit_images
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Chưa tìm thấy thư mục TrashNet: {root}')
    paths, labels = [], []
    for label in LABELS:
        folder = root / label
        if not folder.is_dir():
            raise FileNotFoundError(f'Thiếu thư mục lớp {label}: {folder}')
        for item in sorted(folder.rglob('*')):
            if item.is_file() and item.suffix.lower() in EXTS:
                paths.append(item); labels.append(label)
    kept, kept_labels, summary = audit_images(paths, labels, root)
    items = [(path, LABELS.index(label), path.relative_to(root).as_posix())
             for path,label in zip(kept,kept_labels)]
    duplicates = [(d['path'], d['same_as']) for d in summary['same_label_duplicates']]
    invalid = [(d['path'], d['reason']) for d in summary['invalid']]
    invalid.extend((d['path'], 'Xung đột nhãn: '+str(d['same_as']))
                   for d in summary['label_conflicts'])
    return items, duplicates, invalid

def train_val_split(items, manifest):
    """Outer holdout must be shared with V4 for fair comparison."""
    if manifest.is_file():
        with manifest.open(encoding='utf-8-sig', newline='') as f:
            rows=list(csv.DictReader(f))
        table={row['path'].replace('\\','/').lstrip('./'):row for row in rows}
        # Ignore extra images absent from manifest, which may include V4-deduped files.
        selected=[item for item in items if item[2] in table]
        absent=[k for k in table if k not in {z[2] for z in items}]
        if absent: raise ValueError(f'Split manifest có {len(absent)} ảnh không còn tìm thấy, ví dụ {absent[0]}')
        if len(selected) < 30: raise ValueError('Manifest không đủ dữ liệu hợp lệ')
        train=[]; holdout=[]
        for p,y,rel in selected:
            row=table[rel]
            if row['true_label'] != LABELS[y]: raise ValueError(f'Nhãn sai trong split_manifest: {rel}')
            if row['subset'] == 'train': train.append((p,y,rel))
            elif row['subset'] == 'validation': holdout.append((p,y,rel))
            else: raise ValueError(f'Giá trị subset không hợp lệ: {row["subset"]}')
        if len(set(z[2] for z in train).intersection(z[2] for z in holdout)):
            raise ValueError('Trùng file giữa train và validation')
        print(f'Đã tái sử dụng split V4: train={len(train)} validation={len(holdout)}')
        return train, holdout, 'V4 split_manifest.csv (shared holdout)'
    y=[z[1] for z in items]
    if min(Counter(y).values()) < 8: raise ValueError('Mỗi lớp cần ít nhất 8 ảnh hợp lệ')
    tr,va=train_test_split(np.arange(len(items)),test_size=.2,stratify=y,random_state=SEED)
    return [items[i] for i in tr], [items[i] for i in va], 'stratified 80/20 seed 2026 (new split)'


class TrashDataset(Dataset):
    def __init__(self, items, augment=False, arch=None):
        self.items=list(items)
        self.transform=train_transform(arch) if augment else val_transform(arch)
    def __len__(self): return len(self.items)
    def __getitem__(self, index):
        path,cls,_=self.items[index]
        return self.transform(pil_rgb(path)), int(cls)


def make_loader(items, augment, batch_size, workers, shuffle=False, arch=None):
    return DataLoader(TrashDataset(items,augment,arch=arch), batch_size=batch_size,
                      shuffle=shuffle, num_workers=workers, pin_memory=False)


def epoch_train(model, loader, optimizer, criterion, device, freeze_body):
    model.train()
    if freeze_body:
        model.features.eval()  # avoids modifying frozen batchnorm statistics
    else:
        for block in list(model.features)[:-2]:
            block.eval()  # keep frozen earlier BatchNorm stats fixed
    total=0.; n=0
    for images, target in loader:
        images,target=images.to(device),target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output=model(images)
        loss=criterion(output,target)
        loss.backward()
        optimizer.step()
        total += loss.item()*len(target); n+=len(target)
    return total/max(n,1)


@torch.inference_mode()
def infer(model, loader, device):
    model.eval()
    ys=[]; logits=[]
    for imgs, truth in loader:
        out=model(imgs.to(device)).float().cpu().numpy()
        logits.append(out);ys.extend(truth.numpy().tolist())
    return np.asarray(ys),np.concatenate(logits,axis=0)


def softmax(x):
    z=x-np.max(x,axis=1,keepdims=True)
    e=np.exp(z); return e/np.sum(e,axis=1,keepdims=True)


def fit_temperature(y, logits):
    grid=[.65,.75,.85,1.0,1.15,1.3,1.5,1.75,2.0,2.5,3.0]
    scores=[log_loss(y,softmax(logits/t),labels=list(range(6))) for t in grid]
    return float(grid[int(np.argmin(scores))])


@torch.inference_mode()
def infer_tta(model, loader, device):
    model.eval()
    ys=[]; logits=[]
    for imgs, truth in loader:
        batch=imgs.to(device)
        out=(model(batch).float()+model(batch.flip(dims=[3])).float())/2.
        logits.append(out.cpu().numpy()); ys.extend(truth.numpy().tolist())
    return np.asarray(ys), np.concatenate(logits,axis=0)


def evaluate(model, loader, device, temp, tta=False):
    y,logits=infer_tta(model,loader,device) if tta else infer(model,loader,device)
    prob=softmax(logits/temp)
    pred=np.argmax(prob,axis=1)
    return y,pred,prob


def write_png_cm(cm, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.5,6.5))
    ax.imshow(cm,cmap='Blues')
    ax.set_xticks(range(6),LABELS,rotation=35,ha='right');ax.set_yticks(range(6),LABELS)
    ax.set_xlabel('Dự đoán');ax.set_ylabel('Nhãn thật');ax.set_title('CNN - Ma trận nhầm lẫn (validation)')
    for i in range(6):
        for j in range(6): ax.text(j,i,str(int(cm[i,j])),ha='center',va='center')
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


def safe_write_csv(path, fieldnames, rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader();writer.writerows(rows)


def train(args):
    seed_everything(SEED)
    t_start=time.time()
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Máy chưa có PyTorch CUDA; dùng --device auto hoặc --device cpu')
    device=torch.device('cuda' if (args.device=='auto' and torch.cuda.is_available()) else
                        args.device if args.device != 'auto' else 'cpu')
    print('Thiết bị huấn luyện:', device, '| Kiến trúc:',args.arch)
    if device.type=='cpu' and args.arch=='convnext_tiny':
        print('CẢNH BÁO: ConvNeXt-Tiny khá nặng trên CPU; EfficientNet-B2 phù hợp hơn nếu máy yếu.')
    out=Path(args.results);out.mkdir(parents=True,exist_ok=True)
    model_dir=Path(args.model_dir);model_dir.mkdir(parents=True,exist_ok=True)
    images, duplicates, invalid=collect_images(args.images)
    if len(images)<40 or len(set(y for _,y,_ in images)) !=6: raise ValueError('Thiếu dữ liệu 6 lớp')
    train_outer, val_outer, source = train_val_split(images,Path(args.split_manifest))
    if args.audit_near_duplicates:
        try:
            from src.quality import find_cross_split_near_duplicates
            all_items = train_outer + val_outer
            audit = find_cross_split_near_duplicates(
                [z[0] for z in all_items], [LABELS[z[1]] for z in all_items],
                list(range(len(train_outer))),
                list(range(len(train_outer),len(all_items))), max_examples=30)
            (out/'near_duplicate_review.json').write_text(
                json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
            print('Ảnh gần trùng cần xem xét thủ công:', audit['candidate_pairs'])
        except Exception as ex:
            print('[CẢNH BÁO] Chưa kiểm tra được near-duplicates:',ex)
    if min(Counter(y for _,y,_ in train_outer).values())<5:
        raise ValueError('Mỗi lớp cần đủ ảnh train để tách tập dev')
    tr_idx,dv_idx=train_test_split(np.arange(len(train_outer)),test_size=.15,
                                   stratify=[x[1] for x in train_outer],random_state=SEED)
    tr=[train_outer[i] for i in tr_idx];dv=[train_outer[i] for i in dv_idx]
    print(f'Split: fit={len(tr)}, internal-dev={len(dv)}, final-validation={len(val_outer)}')
    train_ld=make_loader(tr,True,args.batch_size,args.workers,shuffle=True,arch=args.arch)
    dev_ld=make_loader(dv,False,args.batch_size,args.workers,arch=args.arch)
    val_ld=make_loader(val_outer,False,args.batch_size,args.workers,arch=args.arch)
    if args.no_pretrained and not args.smoke:
        raise ValueError('--no-pretrained chỉ cho phép cùng --smoke (không phải mô hình dự thi)')
    model=build_model(args.arch,pretrained=not args.no_pretrained).to(device)
    # Balanced loss is preferable to over-sampling plus class weights simultaneously.
    counts=Counter(y for _,y,_ in tr)
    class_w=np.array([len(tr)/(6*counts[i]) for i in range(6)],dtype=np.float32)
    weights=torch.from_numpy(class_w).to(device)
    criterion=nn.CrossEntropyLoss(weight=weights,label_smoothing=.04)
    best_f1=-1.;best_state=None;best_stage='';best_epoch=0;history=[]

    def phase(stage, epochs, lr_head, lr_body=0.):
        nonlocal best_f1,best_state,best_stage,best_epoch
        if epochs<=0:return
        for param in model.features.parameters():param.requires_grad_(False)
        for param in model.classifier.parameters():param.requires_grad_(True)
        if stage=='fine_tune':
            for block in list(model.features)[-2:]:
                for param in block.parameters(): param.requires_grad_(True)
            optimizer=torch.optim.AdamW([
                {'params':model.classifier.parameters(),'lr':lr_head},
                {'params':(p for p in model.features.parameters() if p.requires_grad),'lr':lr_body}
            ],weight_decay=1e-4)
        else:
            optimizer=torch.optim.AdamW(model.classifier.parameters(),lr=lr_head,weight_decay=1e-4)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max(epochs,1),eta_min=1e-7)
        patience_count=0
        for ep in range(1,epochs+1):
            loss=epoch_train(model,train_ld,optimizer,criterion,device,stage=='head')
            scheduler.step()
            y,logits=infer(model,dev_ld,device)
            pred=logits.argmax(1)
            f1=float(f1_score(y,pred,labels=list(range(6)),average='macro',zero_division=0))
            history.append({'phase':stage,'epoch':ep,'train_loss':round(loss,6),
                            'dev_macro_f1':round(f1,6)})
            print(f'{stage}: epoch {ep}/{epochs}, train_loss={loss:.4f}, dev_Macro-F1={f1:.4f}',flush=True)
            if f1 > best_f1+0.0001:
                best_f1=f1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                best_stage=stage;best_epoch=ep;patience_count=0
            else:
                patience_count+=1
                if patience_count>=args.patience:
                    print('Early stopping theo INTERNAL-DEV');break

    phase('head',args.head_epochs,args.lr_head)
    if args.finetune_epochs>0:
        # Re-start from best dev checkpoint before partial backbone fine-tune.
        model.load_state_dict(best_state)
        phase('fine_tune',args.finetune_epochs,args.lr_ft_head,args.lr_ft_body)
    if best_state is None: raise ValueError('Không có checkpoint; cần ít nhất 1 epoch')
    model.load_state_dict(best_state)
    y_dev,log_dev=infer(model,dev_ld,device)
    y_dev_tta,log_dev_tta=infer_tta(model,dev_ld,device)
    if not np.array_equal(y_dev,y_dev_tta):
        raise ValueError('Internal dev TTA không khớp thứ tự ảnh')
    f1_no=float(f1_score(y_dev,log_dev.argmax(axis=1),labels=list(range(6)),average='macro',zero_division=0))
    f1_tta=float(f1_score(y_dev_tta,log_dev_tta.argmax(axis=1),labels=list(range(6)),average='macro',zero_division=0))
    # TTA selection uses internal DEV only. Validation is never used to decide.
    use_tta=bool(f1_tta > f1_no + .002)
    temp=fit_temperature(y_dev, log_dev_tta if use_tta else log_dev)
    print('TTA hflip:',use_tta,'| dev F1 normal:',round(f1_no,4),'| TTA:',round(f1_tta,4))
    y_true,y_pred,proba=evaluate(model,val_ld,device,temp,tta=use_tta)
    macro=float(f1_score(y_true,y_pred,labels=list(range(6)),average='macro',zero_division=0))
    acc=float(accuracy_score(y_true,y_pred))
    cm=confusion_matrix(y_true,y_pred,labels=list(range(6)))
    metrics={
        'task':'AI02','architecture':args.arch,
        'pretrained':not args.no_pretrained,
        'experimental_mode':'SMOKE_NOT_FOR_SUBMISSION' if args.smoke else 'REAL_DATA_CHECK_SOURCE',
        'dataset_root':str(Path(args.images).resolve()),'number_images_valid':len(images),
        'number_exact_duplicates_removed':len(duplicates),'number_invalid_removed':len(invalid),
        'split_source':source,'split_counts':{'train_fit':len(tr),'dev_internal':len(dv),'val_final':len(val_outer)},
        'seed':SEED,'best_internal_dev_macro_f1':best_f1,'best_stage':best_stage,'best_epoch':best_epoch,
        'temperature_fitted_on_internal_dev':temp,
        'tta_selected_on_internal_dev': 'hflip' if use_tta else 'none',
        'dev_macro_f1_no_tta':f1_no,'dev_macro_f1_hflip_tta':f1_tta,
        'final_val_accuracy':acc,'final_val_macro_f1':macro,
        'final_val_classification_report':classification_report(y_true,y_pred,labels=list(range(6)),
                      target_names=list(LABELS),output_dict=True,zero_division=0),
        'final_val_confusion_matrix':cm.tolist(),
        'metal_as_glass':int(cm[LABELS.index('metal'), LABELS.index('glass')]),
        'glass_as_metal':int(cm[LABELS.index('glass'), LABELS.index('metal')]),
        'history':history,'runtime_seconds':round(time.time()-t_start,1),
        'notes':['Pretrained CNN. Select early stopping/temperature using INTERNAL-DEV only.',
                 'Final validation used exactly once for reporting.',
                 'If split_manifest.csv from V4 is absent, comparison with V4 is NOT paired.',
                 'Per-class performance and risk should be checked, not assumed improved.']
    }
    checkpoint=model_dir/checkpoint_name(args.arch)
    save_checkpoint(checkpoint,model,args.arch,temperature=temp,
                    source='ImageNet pretrained torchvision + TrashNet transfer learning',
                    training_mode=metrics['experimental_mode'],
                    tta='hflip' if use_tta else 'none')
    with (out/f'metrics_{args.arch}.json').open('w',encoding='utf-8') as f:
        json.dump(metrics,f,indent=2,ensure_ascii=False)
    write_png_cm(cm,out/f'confusion_{args.arch}.png')
    safe_write_csv(out/f'predictions_{args.arch}.csv',
                   ['path','true_label','predicted_label','confidence','prob_glass','prob_metal'],
                   [{'path':item[2],'true_label':LABELS[y], 'predicted_label':LABELS[p],
                     'confidence':round(float(pr.max()),6),
                     'prob_glass':round(float(pr[1]),6),'prob_metal':round(float(pr[2]),6)}
                    for item,y,p,pr in zip(val_outer,y_true,y_pred,proba)])
    safe_write_csv(out/f'errors_{args.arch}.csv', ['path','true_label','predicted_label','confidence'],
                   [{'path':item[2],'true_label':LABELS[y],'predicted_label':LABELS[p],
                     'confidence':round(float(pr.max()),6)}
                    for item,y,p,pr in zip(val_outer,y_true,y_pred,proba) if y!=p])
    safe_write_csv(out/'dataset_exceptions.csv',['type','file_1','file_2_or_error'],
                   [{'type':'duplicate','file_1':a,'file_2_or_error':b} for a,b in duplicates]+
                   [{'type':'invalid','file_1':a,'file_2_or_error':b} for a,b in invalid])
    print(f'HOÀN THÀNH: Macro-F1 validation={macro:.4f} Accuracy={acc:.4f} '
          f'| metal→glass={metrics["metal_as_glass"]}; glass→metal={metrics["glass_as_metal"]}')
    print('Mô hình:',checkpoint,'\nBáo cáo:',out/f'metrics_{args.arch}.json')
    if not args.smoke and 'V4 split_manifest' not in source:
        print('CẢNH BÁO: Không có split_manifest V4; không so sánh trực tiếp như cùng tập!')
    return metrics


def main():
    ap=argparse.ArgumentParser(description='Nâng cấp AI02: pretrained CNN transfer learning')
    ap.add_argument('--images',default='data/dataset-resized')
    ap.add_argument('--arch',choices=ARCHES,default='mobilenet_v3_large')
    ap.add_argument('--split-manifest',default='results/split_manifest.csv')
    ap.add_argument('--model-dir',default='model')
    ap.add_argument('--results',default='results_cnn')
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--workers',type=int,default=0,help='Windows: đặt 0 để ít lỗi multiprocessing')
    ap.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    ap.add_argument('--audit-near-duplicates',action='store_true',help='Xuất cảnh báo ảnh gần trùng train/validation, không tự loại ảnh')
    ap.add_argument('--head-epochs',type=int,default=9)
    ap.add_argument('--finetune-epochs',type=int,default=12)
    ap.add_argument('--patience',type=int,default=4)
    ap.add_argument('--lr-head',type=float,default=0.0008)
    ap.add_argument('--lr-ft-head',type=float,default=0.00015)
    ap.add_argument('--lr-ft-body',type=float,default=0.000015)
    ap.add_argument('--smoke',action='store_true',help='Chỉ kiểm thử mã, KHÔNG dùng kết quả dự thi')
    ap.add_argument('--no-pretrained',action='store_true',help='Chỉ cùng --smoke')
    args=ap.parse_args()
    train(args)

if __name__=='__main__':main()
