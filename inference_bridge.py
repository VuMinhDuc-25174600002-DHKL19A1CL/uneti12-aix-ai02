"""Interface adapter for TrashNet AI02 V4 classic models and V5 CNN checkpoints.
All checkpoints must be locally trained and trusted. Never load user-supplied joblib.
"""
from __future__ import annotations
from pathlib import Path
import tempfile

LABELS = ('cardboard','glass','metal','paper','plastic','trash')
VI = {'cardboard':'Bìa carton','glass':'Thủy tinh','metal':'Kim loại',
      'paper':'Giấy','plastic':'Nhựa','trash':'Rác khác'}

def discover_models(model_dir='model'):
    root=Path(model_dir)
    found=[]
    for stem, title in [
        ('ai02_efficientnet_b2','CNN V9 · EfficientNet-B2'),
        ('ai02_convnext_tiny','CNN V9 · ConvNeXt-Tiny'),
        ('ai02_mobilenet_v3_large','CNN V5 · MobileNetV3-Large'),
        ('ai02_efficientnet_b0','CNN V5 · EfficientNet-B0'),
    ]:
        p=root/(stem+'.pth')
        if p.is_file(): found.append({'name':title,'kind':'cnn','paths':[p]})
    cnns=[row for row in found if row['kind']=='cnn']
    if len(cnns)>1:
        # Fixed equal weights; not chosen after inspecting final validation labels.
        found.insert(0,{'name':'CNN · Ensemble 2 mô hình (trung bình đều)',
                        'kind':'cnn','paths':[cnns[0]['paths'][0],cnns[1]['paths'][0]]})
    for stem,title in [('improved','SVM cải tiến V4'),
                       ('baseline','Baseline · HOG + RGB + Logistic Regression')]:
        p=root/(stem+'.joblib')
        if p.is_file(): found.append({'name':title,'kind':'classic','paths':[p]})
    return found

def predict_pil(image, selection):
    if selection['kind']=='cnn':
        # Lazy import so V4 can run before torch is installed.
        from cnn_utils import predict_image
        return predict_image(image, selection['paths'])

    from src.predict import predict_image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as t:
        temporary_path=Path(t.name)
    try:
        image.save(temporary_path,format='PNG')
        raw=predict_image(selection['paths'][0],temporary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    prob={lbl:float(raw.get('probabilities',{}).get(lbl) or 0.0) for lbl in LABELS}
    label=raw['label']
    rank=sorted(prob, key=prob.get,reverse=True)
    confidence=raw.get('confidence')
    if confidence is None: confidence=prob.get(label,0.)
    uncertainty = (confidence < .55 or (len(rank)>1 and prob[rank[0]]-prob[rank[1]] < .15))
    return {'label':label,'label_vi':VI.get(label,label), 'confidence':float(confidence),
            'probabilities':prob, 'uncertain':bool(uncertainty),
            'warning':('Hai nhóm vật liệu có kết quả khá gần nhau; hãy kiểm tra bằng mắt.'
                       if uncertainty else ''),
            'recommendation':raw.get('guide',''),
            'architectures':[selection['name']],
            'note':raw.get('caution','Kết quả tham khảo, có thể dự đoán sai.')}
