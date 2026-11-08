"""Prediction utilities shared by command-line and user interfaces."""
import joblib
import numpy as np
from pathlib import Path
from .features import image_features

GUIDANCE = {
 'cardboard': 'Bìa carton: thu gom sạch/khô theo hướng dẫn của điểm tiếp nhận.',
 'glass': 'Thủy tinh: tránh mảnh sắc; thu gom riêng theo quy định tại nơi tiếp nhận.',
 'metal': 'Kim loại: thu gom riêng khi sạch và phù hợp quy định.',
 'paper': 'Giấy: ưu tiên giữ khô; giấy bẩn có thể cần xử lý khác.',
 'plastic': 'Nhựa: kiểm tra loại nhựa và hệ thống phân loại tại địa phương.',
 'trash': 'Rác khác: cần kiểm tra thủ công trước khi đưa vào luồng xử lý.'
}

def load_bundle(model_file):
    # Security: joblib pickle must be trusted; never accept model upload from strangers.
    bundle = joblib.load(model_file)
    if not isinstance(bundle, dict) or not {'pipeline', 'feature_kind'} <= set(bundle):
        raise ValueError('File mô hình không đúng định dạng dự án AI02')
    return bundle

def predict_image(model_file, image_file):
    b = load_bundle(model_file)
    vec = image_features(image_file, b['feature_kind']).reshape(1, -1)
    pipeline = b['pipeline']
    label = str(pipeline.predict(vec)[0])
    if hasattr(pipeline, 'predict_proba'):
        proba = pipeline.predict_proba(vec)[0]
        prob = {str(k): float(v) for k, v in zip(pipeline.classes_, proba)}
    else:
        prob = {label: None}
    confidence = prob.get(label)
    return {'label': label, 'confidence': confidence, 'probabilities': prob,
            'guide': GUIDANCE.get(label, ''),
            'caution': 'Đây chỉ là dự đoán từ ảnh. Xác suất không đảm bảo đúng; TrashNet chủ yếu có nền đơn giản.'}
