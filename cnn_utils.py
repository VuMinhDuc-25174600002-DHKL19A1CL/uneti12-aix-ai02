"""Shared CNN models, deterministic preprocessing and inference for AI02.

The pretrained weights are downloaded only during training. Inference loads our
own state_dict, never unpickles arbitrary data from users.
"""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps

LABELS = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
VI = {'cardboard':'Bìa carton', 'glass':'Thủy tinh', 'metal':'Kim loại',
      'paper':'Giấy', 'plastic':'Nhựa', 'trash':'Rác khác'}
GUIDANCE = {
    'cardboard':'Giữ bìa carton sạch và khô; phân loại theo hướng dẫn tại nơi thu gom.',
    'glass':'Thận trọng với mảnh vỡ; thu gom thủy tinh theo quy định tại nơi tiếp nhận.',
    'metal':'Thu gom riêng kim loại nếu phù hợp quy định và không còn chất nguy hại.',
    'paper':'Giữ giấy sạch và khô; kiểm tra quy định đối với giấy bẩn.',
    'plastic':'Kiểm tra loại nhựa và quy định của hệ thống thu gom địa phương.',
    'trash':'Kiểm tra thủ công vì vật thể có thể không thuộc các nhóm còn lại.'
}
ARCHES = ('mobilenet_v3_large', 'efficientnet_b0', 'efficientnet_b2', 'convnext_tiny')
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(arch, pretrained=False):
    if arch == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(LABELS))
    elif arch == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(LABELS))
    elif arch == 'efficientnet_b2':
        weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b2(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(LABELS))
    elif arch == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        net = models.convnext_tiny(weights=weights)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, len(LABELS))
    else:
        raise ValueError(f'Kiến trúc không được hỗ trợ: {arch}')
    return net


def input_spec(arch=None):
    """Match size/interpolation used during fine-tuning AND inference.

    Older V5 checkpoints preserve their 224px transformations.
    """
    if arch == 'efficientnet_b2':
        return 288, 288, InterpolationMode.BICUBIC
    if arch == 'convnext_tiny':
        return 224, 236, InterpolationMode.BILINEAR
    return 224, 256, InterpolationMode.BILINEAR


def train_transform(arch=None):
    size, _, interpolation = input_spec(arch)
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.80, 1.0),
                                     interpolation=interpolation),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=.12, contrast=.13, saturation=.08, hue=.015),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])


def val_transform(arch=None):
    size, resize, interpolation = input_spec(arch)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=interpolation),
        transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])

def pil_rgb(path_or_pil):
    if isinstance(path_or_pil, Image.Image):
        return ImageOps.exif_transpose(path_or_pil).convert('RGB')
    with Image.open(path_or_pil) as image:
        return ImageOps.exif_transpose(image).convert('RGB')


def checkpoint_name(arch):
    return f'ai02_{arch}.pth'


def save_checkpoint(path, model, arch, temperature=1.0, **meta):
    payload = {'arch':arch, 'labels':list(LABELS),
               'temperature':float(temperature),
               'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
               'metadata':meta}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@lru_cache(maxsize=3)
def load_checkpoint(path_string):
    path = Path(path_string).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Chưa tìm thấy mô hình CNN: {path}')
    payload = torch.load(str(path), map_location='cpu', weights_only=True)
    if payload.get('arch') not in ARCHES or tuple(payload.get('labels', [])) != LABELS:
        raise ValueError('Checkpoint không đúng định dạng 6 lớp AI02')
    if (payload.get('metadata') or {}).get('training_mode') == 'SMOKE_NOT_FOR_SUBMISSION':
        raise ValueError('Checkpoint kiểm thử giả lập. Không sử dụng để dự đoán hoặc trình diễn AI02.')
    model = build_model(payload['arch'], pretrained=False)
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return model, float(payload.get('temperature', 1.0)), payload


def available_models(model_dir='model'):
    root = Path(model_dir)
    roots = [root]
    if not root.is_absolute():
        roots.append(Path(__file__).resolve().parent / root)
        if getattr(sys, 'frozen', False):
            roots.append(Path(sys.executable).resolve().parent / root)
            roots.append(Path(getattr(sys, '_MEIPASS', '.')).resolve() / root)
    found = []
    for arch in ARCHES:
        for candidate in roots:
            ckpt = candidate / checkpoint_name(arch)
            if ckpt.is_file():
                found.append(ckpt.resolve())
                break
    return found


@torch.inference_mode()
def predict_image(image, model_paths, threshold=.60, margin=.15):
    """Classify with exact trained preprocessing; optional dev-chosen hflip TTA.

    Ensemble uses an equal-weight average declared in advance, not tuned on
    the final validation set. Large ensembles will be slower on CPU.
    """
    if isinstance(model_paths, (str, Path)):
        model_paths = [model_paths]
    if not model_paths:
        raise FileNotFoundError('Chưa có CNN đã huấn luyện. Hãy chạy train_cnn.py.')
    img = pil_rgb(image)
    probs = []
    names = []
    tta_used = []
    for item in model_paths:
        model, temp, payload = load_checkpoint(str(Path(item).resolve()))
        arch = payload['arch']
        # Older V5 metadata has no tta, and remains single-view 224 px.
        meta = payload.get('metadata') or {}
        tta = (meta.get('tta') == 'hflip')
        tens = val_transform(arch)(img).unsqueeze(0)
        logits = model(tens).float()
        if tta:
            logits = (logits + model(tens.flip(dims=[3])).float()) / 2.
        p = torch.softmax(logits.squeeze(0) / max(temp, 1e-4), dim=0).numpy()
        probs.append(p)
        names.append(arch)
        tta_used.append(tta)
    averaged = np.stack(probs).mean(axis=0)
    ranking = np.argsort(-averaged)
    first, second = int(ranking[0]), int(ranking[1])
    label = LABELS[first]
    conf = float(averaged[first]); gap = conf - float(averaged[second])
    uncertain = bool(conf < threshold or gap < margin)
    msg = ('Kết quả chưa chắc chắn: ' + VI[LABELS[first]] +
           ' và ' + VI[LABELS[second]] + ' khá gần nhau. Hãy kiểm tra bằng mắt.') if uncertain else ''
    return {'label':label, 'label_vi':VI[label], 'confidence':conf,
            'probabilities':{lbl:float(averaged[i]) for i,lbl in enumerate(LABELS)},
            'top2':[{'label':LABELS[i], 'label_vi':VI[LABELS[i]],
                     'probability':float(averaged[i])} for i in (first,second)],
            'margin':gap, 'uncertain':uncertain, 'warning':msg,
            'recommendation':GUIDANCE[label], 'architectures':names,
            'tta_used':tta_used,
            'note':'Xác suất tham khảo, không phải cam kết đúng; ảnh ngoài TrashNet có thể sai.'}
