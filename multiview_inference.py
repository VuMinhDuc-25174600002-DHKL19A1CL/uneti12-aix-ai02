"""Multi-photo inference for ONE object from multiple REAL viewpoints.

Each view runs through the SAME selected, actually trained classifier. Average
probabilities and report disagreement. This method doesn't train a new model,
create real extra 3-D information, or guarantee an accuracy improvement.
"""
from __future__ import annotations

from hashlib import sha256
from math import isfinite
from typing import Callable, Iterable

LABELS = ('cardboard','glass','metal','paper','plastic','trash')
VI = dict(cardboard='Bìa carton', glass='Thủy tinh', metal='Kim loại',
          paper='Giấy', plastic='Nhựa', trash='Rác khác')
GUIDES = {
    'cardboard':'Để carton khô, gấp phẳng; kiểm tra lớp phủ và nơi thu gom.',
    'glass':'Xác nhận chai/lọ thủy tinh được nhận, không bỏ mảnh vỡ hoặc bóng đèn vào thùng thông thường.',
    'metal':'Làm sạch lon/hộp kim loại, phân loại theo điểm thu gom.',
    'paper':'Giữ giấy khô, sạch; kiểm tra lớp phủ và nơi tiếp nhận.',
    'plastic':'Kiểm tra loại nhựa, làm sạch; không phải mọi nhựa đều được tiếp nhận.',
    'trash':'Chưa xác định được thành phần; cần quan sát vật phẩm thực tế.',
}


def _normalized_probabilities(result: dict) -> dict:
    raw = result.get('probabilities')
    if not isinstance(raw, dict):
        raise ValueError('Mô hình không trả về xác suất sáu nhóm rác.')
    values = {}
    for label in LABELS:
        try:
            n = float(raw.get(label, 0))
        except (TypeError, ValueError) as error:
            raise ValueError(f'Xác suất không hợp lệ: {label}') from error
        if not isfinite(n) or n < 0:
            raise ValueError(f'Xác suất không hợp lệ: {label}')
        values[label] = n
    total = sum(values.values())
    if total <= 0:
        raise ValueError('Mô hình không cung cấp phân bố xác suất hợp lệ.')
    return {k:v/total for k,v in values.items()}


def fuse_predictions(results: list[dict], *, filenames: list[str] | None = None) -> dict:
    """Equal-weight soft vote with per-view consistency; no artificial certainty."""
    if not 2 <= len(results) <= 6:
        raise ValueError('Cần từ 2 đến 6 ảnh chụp khác góc của cùng một vật.')
    if filenames is not None and len(filenames) != len(results):
        raise ValueError('Tên file phải khớp số ảnh.')
    probs = [_normalized_probabilities(result) for result in results]
    means = {label:sum(p[label] for p in probs)/len(probs) for label in LABELS}
    ranks = sorted(LABELS,key=lambda label:means[label],reverse=True)
    top,runner_up = ranks[:2]
    confid = means[top]; gap = confid - means[runner_up]
    votes = [max(LABELS,key=lambda label: p[label]) for p in probs]
    agree_count = sum(v == top for v in votes)
    agreement = agree_count / len(votes)
    # With two conflicting viewpoints, warn rather than pretending fusion solved it.
    uncertain = bool(confid < .60 or gap < .15 or agreement < .75)
    per_view = []
    for idx,(label,p) in enumerate(zip(votes,probs)):
        per_view.append({'goc': idx+1,
                         'anh':filenames[idx] if filenames else f'Góc {idx+1}',
                         'nhan':label, 'nhan_tieng_viet':VI[label],
                         'xac_suat_%':round(100*p[label],1),
                         'phieu_cho_ket_qua_cuoi':(label == top)})
    return {'label':top, 'label_vi':VI[top], 'confidence':confid,
            'probabilities':means, 'uncertain':uncertain,
            'warning':('Các góc ảnh cho kết quả khác nhau hoặc xác suất gần nhau; hãy kiểm tra vật liệu bằng mắt.' if uncertain else ''),
            'recommendation':GUIDES[top],
            'note':'Trung bình xác suất nhiều ảnh thật, không phải bảo đảm đúng. Chỉ chụp CÙNG MỘT vật.',
            'architectures':['Multi-view soft voting'],
            'view_count':len(results), 'view_agreement':agreement,
            'top2_margin':gap, 'views':per_view,
            'fusion_method':'Trung bình xác suất 2–6 góc chụp thật của cùng một vật',
            'disagreement':agreement < .75}


def dedupe_files_by_contents(files: Iterable):
    """Avoid counting identical byte-for-byte images as independent viewpoints."""
    unique, duplicate_names, seen = [], [], set()
    for obj in files:
        digest = sha256(obj.getvalue()).digest()
        if digest in seen:
            duplicate_names.append(getattr(obj,'name','Ảnh trùng'))
        else:
            seen.add(digest)
            unique.append(obj)
    return unique, duplicate_names


def image_exposure_hint(image):
    """Coarse brightness warning, never used as evidence for a waste class."""
    from PIL import ImageStat
    if image is None: return ''
    thumb = image.copy(); thumb.thumbnail((160,160))
    lum = ImageStat.Stat(thumb.convert('L')).mean[0]
    if lum < 35: return 'Ảnh rất tối; hãy chụp thêm góc đủ sáng.'
    if lum > 230: return 'Ảnh rất sáng; tránh ánh sáng chiếu thẳng gây cháy ảnh.'
    return ''
