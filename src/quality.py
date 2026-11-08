"""Validate real images and remove exact duplicates before splitting.

Does not claim to detect every near-duplicate of the same object/session.
"""
from collections import Counter
import hashlib
from pathlib import Path
from PIL import Image, ImageOps


def audit_images(paths, labels, dataset_root):
    kept, kept_labels, invalid, duplicates, conflicts = [], [], [], [], []
    lookup = {}
    for path, label in zip(paths, labels):
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                width, height = im.size
                if width < 32 or height < 32:
                    raise ValueError(f'Ảnh quá nhỏ: {width}x{height}')
                content = f'{width}x{height}|'.encode() + im.tobytes()
                digest = hashlib.sha256(content).hexdigest()
        except Exception as e:
            invalid.append({'path': str(path), 'reason': str(e)})
            continue
        if digest in lookup:
            prev_path, prev_label = lookup[digest]
            d = {'path': str(path), 'label': str(label),
                 'same_as': str(prev_path), 'same_as_label': str(prev_label)}
            if prev_label == label:
                duplicates.append(d)
            else:
                conflicts.append(d)
            continue
        lookup[digest] = (path, label)
        kept.append(path)
        kept_labels.append(label)
    # Exclude conflicting hashes from both labels, rather than choosing a truth arbitrarily.
    bad_hashes = set()
    for d in conflicts:
        # The first copy's byte hash is looked up via saved path.
        for h, (p, lab) in lookup.items():
            if str(p) == d['same_as']:
                bad_hashes.add(h)
                break
    if bad_hashes:
        unwanted = {str(lookup[h][0]) for h in bad_hashes}
        kept_labels = [l for p, l in zip(kept, kept_labels) if str(p) not in unwanted]
        kept = [p for p in kept if str(p) not in unwanted]
    summary = {
        'n_input': len(paths), 'n_after_cleanup': len(kept),
        'by_class_original': dict(sorted(Counter(map(str, labels)).items())),
        'by_class_used': dict(sorted(Counter(map(str, kept_labels)).items())),
        'n_invalid': len(invalid), 'n_exact_duplicate_same_label': len(duplicates),
        'n_conflicting_duplicate': len(conflicts), 'invalid': invalid,
        'same_label_duplicates': duplicates, 'label_conflicts': conflicts,
        'duplicate_scope': 'SHA256 trên pixels RGB gốc đã giải mã, không kiểm tra gần trùng'
    }
    return kept, kept_labels, summary


def find_cross_split_near_duplicates(paths, labels, train_ix, val_ix,
                                     max_hamming=3, max_pixel_mae=0.035,
                                     max_examples=40):
    """Advisory dHash scan for near-identical train/validation images.

    Does not automatically remove pairs: perceptual hashing may flag unrelated
    items on similar white backgrounds. Review the files manually if found.
    """
    import numpy as np

    def dhash(path):
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert('L').resize((9, 8))
            array = np.asarray(im, dtype=np.uint8)
        mask = array[:, 1:] > array[:, :-1]
        bits = 0
        for val in mask.flat:
            bits = (bits << 1) | int(val)
        return bits

    values = [dhash(path) for path in paths]
    # Color difference guard: dHash alone confuses many different images on
    # uniform backgrounds. Compute a compact 32x32 RGB preview only once.
    previews = []
    for path in paths:
        with Image.open(path) as im:
            small = ImageOps.exif_transpose(im).convert('RGB').resize((32, 32))
            previews.append(np.asarray(small, dtype=np.uint8))
    examples = []
    count = 0
    same_class_count = 0
    for t in train_ix:
        for v in val_ix:
            dist = (values[int(t)] ^ values[int(v)]).bit_count()
            if dist <= max_hamming:
                pixel_mae = float(np.abs(previews[int(t)].astype(np.float32) -
                                         previews[int(v)].astype(np.float32)).mean() / 255.)
                if pixel_mae > max_pixel_mae:
                    continue
                count += 1
                same_class_count += int(labels[int(t)] == labels[int(v)])
                if len(examples) < max_examples:
                    examples.append({
                        'train_path': str(paths[int(t)]), 'validation_path': str(paths[int(v)]),
                        'train_label': str(labels[int(t)]), 'validation_label': str(labels[int(v)]),
                        'hamming_distance': int(dist),
                        'pixel_mae': round(pixel_mae, 6)
                    })
    return {
        'method': 'dHash 64-bit + 32x32 RGB pixel MAE; advisory cross-split only',
        'hamming_threshold': max_hamming, 'pixel_mae_threshold': max_pixel_mae,
        'candidate_pairs': count,
        'same_class_candidate_pairs': same_class_count,
        'cross_class_candidate_pairs': count - same_class_count,
        'examples_truncated': count > max_examples, 'examples': examples,
        'warning': 'Similarity candidates are NOT confirmed duplicate objects; inspect originals manually. '
                   'This step never uses labels to train or select the model.'
    }
