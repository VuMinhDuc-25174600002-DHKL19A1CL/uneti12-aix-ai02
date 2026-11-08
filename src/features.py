"""HOG+RGB baseline and a richer CPU-friendly image descriptor for AI02.

The baseline deliberately reproduces the features supplied by the organizer.
All transforms operate on a single image; none is fitted on validation data.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2hsv

LABELS = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect(folder):
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f'Không tìm thấy thư mục dữ liệu: {root}')
    paths, labels = [], []
    for label in LABELS:
        sub = root / label
        if not sub.is_dir():
            raise FileNotFoundError(f'Thiếu thư mục của lớp {label}: {sub}')
        for p in sorted(sub.rglob('*')):
            if p.is_file() and p.suffix.lower() in EXTS:
                paths.append(p)
                labels.append(label)
    if not paths:
        raise ValueError('Không tìm được ảnh hợp lệ (jpg/png/bmp/webp).')
    return paths, np.asarray(labels)


def _open_rgb_gray(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(im.convert('RGB').resize((64, 64)), dtype=np.float32) / 255.
        gray = np.asarray(im.convert('L').resize((64, 64)), dtype=np.float32) / 255.
    return rgb, gray


def _hist(x, bins, rng=(0., 1.)):
    # Same density=True convention as the supplied BTC feature code.
    return np.histogram(x, bins=bins, range=rng, density=True)[0].astype('float32')


def image_features(path, kind='baseline'):
    rgb, gray = _open_rgb_gray(path)
    h = hog(gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), feature_vector=True).astype('float32')
    base_color = np.hstack([_hist(rgb[:, :, c], 8) for c in range(3)])
    baseline = np.hstack([h, base_color]).astype('float32')
    if kind == 'baseline':
        return baseline
    if kind != 'improved':
        raise ValueError(f'Loại đặc trưng không hợp lệ: {kind}')

    # Improvements: HSV material-color cues + local spatial color + LBP texture.
    hsv = rgb2hsv(rgb)
    hsv_hist = np.hstack([_hist(hsv[:, :, c], 12) for c in range(3)])
    spatial = np.hstack([
        _hist(rgb[ry:ry+32, rx:rx+32, c], 8)
        for ry in (0, 32) for rx in (0, 32) for c in range(3)
    ])
    # LBP expects integer pixels; use uniform patterns and small 2x2 spatial cells.
    gray_u8 = np.asarray(np.rint(gray * 255), dtype=np.uint8)
    lbp = local_binary_pattern(gray_u8, P=8, R=1, method='uniform')
    texture = np.hstack([
        np.histogram(lbp[ry:ry+32, rx:rx+32], bins=np.arange(12), density=True)[0]
        for ry in (0, 32) for rx in (0, 32)
    ])
    color_moments = np.hstack([rgb.mean((0, 1)), rgb.std((0, 1))])
    return np.nan_to_num(np.hstack([baseline, hsv_hist, spatial, texture, color_moments]),
                         nan=0., posinf=0., neginf=0.).astype('float32')
