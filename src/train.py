"""Fair AI02 baseline/improved comparison. Run: python -m src.train --images data/dataset-resized.

Important: select hyperparameters on TRAIN folds only, evaluate validation once.
No validation labels are fed into training/model selection.
"""
import argparse
import csv
import json
import time
import platform
import sklearn
import skimage
import PIL
from collections import Counter
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import LABELS, collect, image_features
from .quality import audit_images, find_cross_split_near_duplicates

SEED = 2026


def save_json(filename, obj):
    Path(filename).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def score(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'classification_report': classification_report(
            y_true, y_pred, labels=list(LABELS), output_dict=True, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()
    }


def bootstrap_deltas(y, pred_a, pred_b, n_boot=800):
    y, pred_a, pred_b = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(SEED + 1)
    n = len(y)
    deltas, a_scores, b_scores = [], [], []
    for _ in range(n_boot):
        s = rng.integers(0, n, size=n)
        va = f1_score(y[s], pred_a[s], labels=list(LABELS), average='macro', zero_division=0)
        vb = f1_score(y[s], pred_b[s], labels=list(LABELS), average='macro', zero_division=0)
        a_scores.append(va); b_scores.append(vb); deltas.append(vb-va)
    return {
        'method': 'Paired bootstrap on one held-out validation set; exploratory, not external test',
        'n_resamples': n_boot,
        'baseline_macro_f1_ci95': list(map(float, np.quantile(a_scores, [0.025, 0.975]))),
        'improved_macro_f1_ci95': list(map(float, np.quantile(b_scores, [0.025, 0.975]))),
        'difference_macro_f1_ci95': list(map(float, np.quantile(deltas, [0.025, 0.975]))),
        'difference_ci_excludes_zero': bool(np.quantile(deltas, 0.025) > 0 or np.quantile(deltas, 0.975) < 0)
    }


def probability_diagnostics(y_true, classes, proba, n_bins=10):
    """Multiclass Brier score and ECE; descriptive only, not proof of calibration."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=np.float64)
    classes = np.asarray(classes)
    if proba.shape != (len(y_true), len(classes)):
        raise ValueError('Xác suất không khớp số ảnh hoặc số lớp')
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise ValueError('Xác suất có NaN/Inf hoặc ngoài [0, 1]')
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError('Tổng xác suất từng ảnh phải bằng 1')
    targets = (y_true[:, None] == classes[None, :]).astype(float)
    confidence = proba.max(axis=1)
    correct = (classes[proba.argmax(axis=1)] == y_true).astype(float)
    ece = 0.0
    bins = []
    for j in range(n_bins):
        low, high = j / n_bins, (j + 1) / n_bins
        mask = (confidence >= low) & ((confidence < high) if j < n_bins - 1 else (confidence <= high))
        count = int(mask.sum())
        if not count:
            continue
        accuracy = float(correct[mask].mean())
        avg_conf = float(confidence[mask].mean())
        ece += count / len(y_true) * abs(accuracy - avg_conf)
        bins.append({'range': [low, high], 'n': count, 'accuracy': accuracy, 'mean_confidence': avg_conf})
    return {'multiclass_brier': float(np.square(proba - targets).sum(axis=1).mean()),
            'ece_10bins': float(ece), 'reliability_bins': bins,
            'note': 'ECE/Brier tính trên validation; không bảo đảm xác suất đúng cho ảnh thực địa'}


def pairwise_confusion_report(cm_before, cm_after):
    rows = []
    for i in range(len(LABELS)):
        for j in range(i + 1, len(LABELS)):
            before = int(cm_before[i, j] + cm_before[j, i])
            after = int(cm_after[i, j] + cm_after[j, i])
            rows.append({'label_a': LABELS[i], 'label_b': LABELS[j],
                         'baseline_errors': before, 'improved_errors': after,
                         'errors_reduced': before - after})
    return sorted(rows, key=lambda x: (-x['errors_reduced'], -x['baseline_errors'],
                                       x['label_a'], x['label_b']))


def save_cm(matrix, filename, title):
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    im = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(6), LABELS, rotation=35, ha='right')
    ax.set_yticks(range(6), LABELS)
    ax.set_xlabel('Nhãn mô hình dự đoán')
    ax.set_ylabel('Nhãn thật')
    ax.set_title(title)
    for r in range(6):
        for c in range(6):
            ax.text(c, r, int(matrix[r][c]), ha='center', va='center',
                    color='white' if matrix[r][c] > matrix.max() * .58 else 'black')
    fig.colorbar(im, ax=ax, fraction=.045, pad=.02)
    fig.tight_layout()
    fig.savefig(filename, dpi=190)
    plt.close(fig)


def plot_distribution(counts, filename):
    fig, ax = plt.subplots(figsize=(8.4, 4.7))
    vals = [counts.get(x, 0) for x in LABELS]
    bars = ax.bar(LABELS, vals, color='#197a8c')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v+3, str(v), ha='center', fontsize=9)
    ax.set_title('Phân phối 6 lớp - ảnh hợp lệ sau loại ảnh trùng chính xác')
    ax.set_ylabel('Số ảnh')
    ax.set_ylim(0, max(vals)*1.13)
    fig.tight_layout(); fig.savefig(filename, dpi=190); plt.close(fig)


def plot_per_class(before, after, filename):
    cls = list(LABELS)
    b = [before['classification_report'][x]['f1-score'] for x in cls]
    a = [after['classification_report'][x]['f1-score'] for x in cls]
    pos = np.arange(6)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(pos-.2, b, .4, label='Baseline', color='#456e95')
    ax.bar(pos+.2, a, .4, label='Cải tiến', color='#20a39e')
    ax.set_xticks(pos, cls)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel('F1-score trên validation')
    ax.set_title('So sánh F1 theo từng nhóm rác')
    ax.legend()
    fig.tight_layout(); fig.savefig(filename, dpi=190); plt.close(fig)


def write_csv(filename, header, rows):
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def write_result_md(out, results):
    a, b = results['baseline'], results['improved']
    delta = results['comparison']['delta_macro_f1']
    pair = results['comparison']['most_confused_pair_baseline']
    ci = results['bootstrap']['difference_macro_f1_ci95']
    msg = 'Tăng' if delta > 0 else ('Giảm' if delta < 0 else 'Không đổi')
    lines = [
        ('# KIỂM TRA MÃ TRÊN DỮ LIỆU THỬ - KHÔNG DÙNG ĐỂ NỘP' if
         results.get('experimental_mode') == 'SMOKE_SYNTHETIC_DO_NOT_REPORT' else
         '# BÁO CÁO VALIDATION AI02 - CẦN XÁC MINH NGUỒN ẢNH'), '',
        f"- Bộ dữ liệu sau lọc: **{results['dataset']['n_after_cleanup']} ảnh**.",
        f"- Phân chia: train **{results['split']['n_train']}**, validation **{results['split']['n_validation']}**; stratified, seed 2026.",
        f"- Baseline Macro-F1: **{a['macro_f1']:.4f}**; Accuracy: **{a['accuracy']:.4f}**.",
        f"- Cải tiến Macro-F1: **{b['macro_f1']:.4f}**; Accuracy: **{b['accuracy']:.4f}**.",
        f"- {msg} Macro-F1: **{delta:+.4f}** (điểm tuyệt đối; không phải % tương đối).",
        f"- Khoảng tin cậy bootstrap 95% cho hiệu số: **[{ci[0]:+.4f}, {ci[1]:+.4f}]**.",
        f"- Cặp dễ nhầm nhất ở baseline: **{pair['label_a']} / {pair['label_b']}**, tổng {pair['baseline_n']} lỗi hai chiều; sau cải tiến {pair['improved_n']} lỗi.",
        '', '## Giải thích thiết kế',
        '- Baseline giữ HOG 64x64 + RGB histogram + Logistic Regression, theo mã BTC.',
        '- Cải tiến thêm HSV, histogram màu theo vùng, LBP; dùng RBF SVM được hiệu chỉnh xác suất kiểu sigmoid trên train.',
        '- Siêu tham số SVM được chọn bằng cross-validation **chỉ trên train**; validation để so sánh cuối.',
        '- Đã rà ảnh hỏng/trùng nội dung chính xác; kiểm tra gần trùng theo ngữ nghĩa cần bổ sung nếu có điều kiện.',
        '', '## Diễn giải thận trọng',
        '- Cải tiến không được coi là vượt baseline nếu Macro-F1 không tăng trên cùng validation.',
        '- Khoảng tin cậy chứa số 0 => bằng chứng chênh lệch chưa đủ chắc chắn.',
        '- Xác suất trên giao diện chỉ tham khảo; TrashNet chụp trên nền đơn giản.',
        '- Kết quả không đại diện mọi điểm thu gom ở Việt Nam và không đảm bảo khả năng tái chế.',
        '- Kiểm tra Brier score/ECE có ý nghĩa mô tả, độ tin cậy xác suất không được bảo đảm.',
        '', '## Hình trong thư mục results',
        '- `01_class_distribution.png`: phân phối lớp.',
        '- `02_confusion_baseline.png`, `03_confusion_improved.png`: ma trận nhầm lẫn.',
        '- `04_per_class_f1.png`: so sánh F1 theo lớp.',
        '- `split_manifest.csv`: nhãn và danh sách từng ảnh trong train/validation.',
        '- `near_duplicate_audit.json`, `near_duplicate_candidates.csv`: gợi ý kiểm tra ảnh gần trùng qua train/validation bằng dHash (KHÔNG tự động loại).',
        '- `pairwise_confusions.csv`: mọi cặp lớp và số lỗi giảm/tăng.',
        '- `metrics.json`: thêm Brier score và ECE 10 bins (phân tích tham khảo).',
    ]
    (out / 'BAO_CAO_KET_QUA.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    started = time.perf_counter()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir); model_dir.mkdir(parents=True, exist_ok=True)
    paths_all, y_all = collect(args.images)
    paths, labels, audit = audit_images(paths_all, y_all, args.images)
    y = np.asarray(labels)
    if len(y) < 40 or len(np.unique(y)) != 6 or min(Counter(y).values()) < 8:
        raise ValueError('Thiếu ảnh hoặc lớp sau lọc; cần ít nhất 8 ảnh mỗi lớp (tối thiểu 40 ảnh).')
    print('[1/5] Đã kiểm tra', len(paths_all), 'ảnh; còn', len(paths), 'ảnh hợp lệ không trùng chính xác.')
    save_json(out/'dataset_audit.json', audit)
    plot_distribution(audit['by_class_used'], out/'01_class_distribution.png')

    ix = np.arange(len(y))
    train_ix, val_ix = train_test_split(ix, test_size=.20, stratify=y, random_state=SEED)
    near_dup = find_cross_split_near_duplicates(paths, y, train_ix, val_ix)
    save_json(out/'near_duplicate_audit.json', near_dup)
    if near_dup['candidate_pairs']:
        print('CHÚ Ý:', near_dup['candidate_pairs'],
              'cặp ảnh gần giống train/validation (dHash). Xem near_duplicate_candidates.csv;'
              ' chỉ là gợi ý kiểm tra thủ công.')
    print('[2/5] Trích đặc trưng từ' , len(paths), 'ảnh; thời gian phụ thuộc CPU.')
    # Single complete extraction before models; no fit on validation for descriptor.
    X_base = np.vstack([image_features(p, 'baseline') for p in paths])
    X_imp = np.vstack([image_features(p, 'improved') for p in paths])
    save_json(out/'feature_dimensions.json', {'baseline': X_base.shape[1], 'improved': X_imp.shape[1]})

    print('[3/5] Huấn luyện baseline đúng tham số BTC...')
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, class_weight='balanced'))
    base.fit(X_base[train_ix], y[train_ix])
    p_base = base.predict(X_base[val_ix])
    met_base = score(y[val_ix], p_base)
    met_base['probability_diagnostics'] = probability_diagnostics(
        y[val_ix], base.classes_, base.predict_proba(X_base[val_ix]))

    print('[4/5] Tối ưu SVM bằng train-only cross-validation...')
    search_params = [0.5, 2.0, 8.0]
    if args.fast:
        best_c, cv_info = 2.0, {'mode': 'fast smoke mode (NO hyperparameter search)', 'best_C': 2.0}
    else:
        grid = GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', class_weight='balanced')),
            {'svc__C': search_params}, scoring='f1_macro',
            cv=StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED),
            n_jobs=1, refit=True, return_train_score=False,
        )
        grid.fit(X_imp[train_ix], y[train_ix])
        best_c = float(grid.best_params_['svc__C'])
        cv_info = {
            'method': f'{args.cv_folds}-fold Stratified CV train-only',
            'candidate_C': search_params, 'best_C': best_c,
            'best_cv_macro_f1': float(grid.best_score_),
            'all_candidate_results': [
                {'C': float(p['svc__C']), 'mean_macro_f1': float(m), 'std_macro_f1': float(s)}
                for p, m, s in zip(grid.cv_results_['params'],
                                   grid.cv_results_['mean_test_score'],
                                   grid.cv_results_['std_test_score'])]
        }
    # Calibrate on TRAIN folds only, so predict() is the argmax of predict_proba().
    # This avoids scikit-learn SVC(probability=True)'s possible mismatch between
    # decision_function class and maximum-probability class during a live demo.
    imp = CalibratedClassifierCV(
        estimator=make_pipeline(StandardScaler(), SVC(
            C=best_c, kernel='rbf', gamma='scale', class_weight='balanced', random_state=SEED)),
        method='sigmoid', cv=3)
    imp.fit(X_imp[train_ix], y[train_ix])
    p_imp = imp.predict(X_imp[val_ix])
    met_imp = score(y[val_ix], p_imp)
    met_imp['probability_diagnostics'] = probability_diagnostics(
        y[val_ix], imp.classes_, imp.predict_proba(X_imp[val_ix]))
    bootstrap = bootstrap_deltas(y[val_ix], p_base, p_imp,
                                  n_boot=100 if args.fast else 800)
    cm_b = np.asarray(met_base['confusion_matrix'])
    cm_i = np.asarray(met_imp['confusion_matrix'])
    wrong = cm_b + cm_b.T
    np.fill_diagonal(wrong, 0)
    aa, bb = np.unravel_index(np.argmax(wrong), wrong.shape)
    pair_report = pairwise_confusion_report(cm_b, cm_i)
    confusion_pair = {'label_a': LABELS[aa], 'label_b': LABELS[bb],
                      'baseline_n': int(cm_b[aa, bb]+cm_b[bb, aa]),
                      'improved_n': int(cm_i[aa, bb]+cm_i[bb, aa])}

    results = {
        'task': 'AI02', 'label_order': list(LABELS), 'dataset': audit,
        'near_duplicate_split_audit': {k:v for k,v in near_dup.items() if k != 'examples'},
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
                        'scikit_image': skimage.__version__, 'pillow': PIL.__version__},
        'split': {'type': 'stratified 80:20 holdout AFTER exact dedup', 'random_state': SEED,
                  'n_train': len(train_ix), 'n_validation': len(val_ix),
                  'train_class_counts': dict(Counter(y[train_ix])),
                  'validation_class_counts': dict(Counter(y[val_ix]))},
        'baseline_definition': 'HOG 64x64 + RGB hist(8 x3) + StandardScaler + balanced LogisticRegression max_iter=1200',
        'improvement_definition': 'Baseline descriptor + HSV hist + spatial RGB + LBP + color moments + RBF SVM with train-only 3-fold sigmoid probability calibration',
        'baseline': met_base, 'improved': met_imp,
        'hyperparameter_search_train_only': cv_info, 'bootstrap': bootstrap,
        'comparison': {'delta_macro_f1': float(met_imp['macro_f1']-met_base['macro_f1']),
                       'delta_accuracy': float(met_imp['accuracy']-met_base['accuracy']),
                       'improved_beats_baseline_on_validation': bool(met_imp['macro_f1'] > met_base['macro_f1']),
                       'most_confused_pair_baseline': confusion_pair,
                       'pairwise_error_reductions': pair_report,
                       'n_pairs_with_fewer_errors': sum(row['errors_reduced'] > 0 for row in pair_report)},
        'limitations': ['Không đánh giá trên bộ ảnh ngoài phạm vi TrashNet.',
                        'Chưa kiểm tra ảnh gần trùng do chụp liên tiếp; chỉ phát hiện bản trùng pixel tuyệt đối.',
                        'SVM sigmoid calibration là xác suất ước lượng, không đảm bảo đúng trong mọi bối cảnh.'],
        'runtime_seconds': time.perf_counter()-started,
        'experimental_mode': 'SMOKE_SYNTHETIC_DO_NOT_REPORT' if args.fast else 'ACTUAL_DATA_IF_INPUT_IS_ORIGINAL_TRASHNET'
    }
    save_json(out/'metrics.json', results)
    save_cm(cm_b, out/'02_confusion_baseline.png', 'Ma trận nhầm lẫn - Baseline')
    save_cm(cm_i, out/'03_confusion_improved.png', 'Ma trận nhầm lẫn - Cải tiến')
    plot_per_class(met_base, met_imp, out/'04_per_class_f1.png')

    def rel_path(index):
        try:
            return str(paths[index].relative_to(Path(args.images)))
        except ValueError:
            return str(paths[index])
    train_membership = set(map(int, train_ix))
    write_csv(out/'split_manifest.csv', ['path', 'true_label', 'subset'],
              [[rel_path(i), y[i], 'train' if i in train_membership else 'validation']
               for i in range(len(paths))])
    write_csv(out/'near_duplicate_candidates.csv',
              ['train_path', 'validation_path', 'train_label', 'validation_label', 'hamming_distance', 'pixel_mae'],
              [[e[k] for k in ('train_path', 'validation_path', 'train_label',
                               'validation_label', 'hamming_distance', 'pixel_mae')]
               for e in near_dup['examples']])
    write_csv(out/'pairwise_confusions.csv',
              ['label_a', 'label_b', 'baseline_errors', 'improved_errors', 'errors_reduced'],
              [[row[k] for k in ('label_a', 'label_b', 'baseline_errors',
                                  'improved_errors', 'errors_reduced')] for row in pair_report])
    write_csv(out/'validation_predictions.csv',
              ['path', 'true_label', 'baseline_prediction', 'improved_prediction', 'baseline_correct', 'improved_correct'],
              [[rel_path(i), y[i], pb, pi, int(y[i] == pb), int(y[i] == pi)]
               for i, pb, pi in zip(val_ix, p_base, p_imp)])
    write_csv(out/'error_examples.csv', ['path','true','baseline','improved','improved_fixed_error'],
              [[rel_path(i), y[i], pb, pi, int(pb != y[i] and pi == y[i])]
               for i, pb, pi in zip(val_ix, p_base, p_imp) if pi != y[i] or pb != y[i]])
    joblib.dump({'pipeline': base, 'feature_kind': 'baseline', 'labels': list(LABELS)},
                model_dir/'baseline.joblib', compress=3)
    joblib.dump({'pipeline': imp, 'feature_kind': 'improved', 'labels': list(LABELS),
                 'best_C': best_c}, model_dir/'improved.joblib', compress=3)
    write_result_md(out, results)
    if not any(x['errors_reduced'] > 0 and x['baseline_errors'] > 0 for x in pair_report):
        print('CẢNH BÁO: Chưa có cặp lớp nào giảm số lỗi hai chiều trên validation.')
    print('[5/5] Hoàn thành. Baseline Macro-F1 =', round(met_base['macro_f1'], 4),
          '| Improved Macro-F1 =', round(met_imp['macro_f1'], 4),
          '| Delta =', round(results['comparison']['delta_macro_f1'], 4))
    if results['comparison']['delta_macro_f1'] <= 0:
        print('CẢNH BÁO: Cải tiến chưa đạt Macro-F1 cao hơn baseline trên validation. Không tuyên bố vượt baseline.')
    return results


def main():
    ap = argparse.ArgumentParser(description='AI02 train/evaluate baseline and improvement fairly')
    ap.add_argument('--images', default='data/dataset-resized', help='thư mục có 6 thư mục con TrashNet')
    ap.add_argument('--out', default='results', help='thư mục kết quả')
    ap.add_argument('--model-dir', default='model', help='thư mục lưu mô hình')
    ap.add_argument('--cv-folds', type=int, default=3, choices=[3,4,5])
    ap.add_argument('--fast', action='store_true', help='Chỉ kiểm tra chạy thử; KHÔNG sử dụng kết quả làm minh chứng')
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()
