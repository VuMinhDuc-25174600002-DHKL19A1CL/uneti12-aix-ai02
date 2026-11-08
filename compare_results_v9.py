"""Compare V4 baseline/SVM to CNN on the exact same labeled validation images.
Run after CNN training, requiring results/validation_predictions.csv and the V5 CSV.
"""
from pathlib import Path
from collections import Counter
import json,csv,argparse
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from cnn_utils import LABELS


def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def analyze(old_csv,new_csv,out):
    old={r['path'].replace('\\','/'):r for r in read_csv(old_csv)}
    new={r['path'].replace('\\','/'):r for r in read_csv(new_csv)}
    if set(old)!=set(new):
        raise ValueError('Hai validation khác nhau! Không được so sánh. Cần dùng V4 split_manifest.csv.')
    keys=sorted(old)
    truth=[old[k]['true_label'] for k in keys]
    if any(new[k]['true_label']!=t for k,t in zip(keys,truth)):
        raise ValueError('Nhãn thật khác nhau; dừng so sánh')
    pred_base=[old[k]['baseline_prediction'] for k in keys]
    pred_svm=[old[k]['improved_prediction'] for k in keys]
    pred_cnn=[new[k]['predicted_label'] for k in keys]
    scores={n:float(f1_score(truth,p,labels=list(LABELS),average='macro',zero_division=0))
            for n,p in [('baseline',pred_base),('svm_v4',pred_svm),('cnn_v9',pred_cnn)]}
    cms={n:confusion_matrix(truth,p,labels=list(LABELS)) for n,p in
         [('baseline',pred_base),('svm_v4',pred_svm),('cnn_v9',pred_cnn)]}
    i,j=LABELS.index('metal'),LABELS.index('glass')
    metal_glass={n:{'metal_pred_glass':int(m[i,j]),'glass_pred_metal':int(m[j,i])} for n,m in cms.items()}
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    reports={n:classification_report(truth,p,labels=list(LABELS),output_dict=True,
              zero_division=0) for n,p in [('baseline',pred_base),('svm_v4',pred_svm),('cnn_v9',pred_cnn)]}
    with (out/'CLASS_F1_V4_V9.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['class','baseline_f1','svm_v4_f1','cnn_v9_f1','delta_cnn_baseline'])
        for label in LABELS:
            b=reports['baseline'][label]['f1-score'];c=reports['cnn_v9'][label]['f1-score']
            w.writerow([label,round(b,5),round(reports['svm_v4'][label]['f1-score'],5),
                        round(c,5),round(c-b,5)])
    pair_rows=[]
    for i in range(6):
        for j in range(i+1,6):
            counts={n:int(m[i,j]+m[j,i]) for n,m in cms.items()}
            pair_rows.append([LABELS[i],LABELS[j],counts['baseline'],counts['svm_v4'],
                              counts['cnn_v9'],counts['baseline']-counts['cnn_v9']])
    with (out/'CAP_LOP_DE_NHAM.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['class_a','class_b','baseline_errors','svm_v4_errors',
                                   'cnn_v9_errors','errors_reduced_vs_baseline'])
        w.writerows(sorted(pair_rows,key=lambda r:-r[-1]))
    # Paired uncertainty: same validation rows for all models.
    rng=np.random.default_rng(2026)
    deltas=[]
    for _ in range(400):
        picks=rng.integers(0,len(truth),size=len(truth))
        ya=np.asarray(truth)[picks];ba=np.asarray(pred_base)[picks];ca=np.asarray(pred_cnn)[picks]
        deltas.append(f1_score(ya,ca,labels=list(LABELS),average='macro',zero_division=0)-
                      f1_score(ya,ba,labels=list(LABELS),average='macro',zero_division=0))
    detail={'n_validation':len(keys),'macro_f1':scores,'metal_glass_confusions':metal_glass,
            'cnn_minus_baseline_macro_f1':scores['cnn_v9']-scores['baseline'],
            'paired_bootstrap_delta_ci95':np.quantile(deltas,[.025,.975]).tolist(),
            'per_class_f1':{label:{n:float(rep[label]['f1-score']) for n,rep in reports.items()}
                               for label in LABELS},
            'num_pairs_improved_vs_baseline':sum(r[-1]>0 for r in pair_rows),
            'warning':'F1 tăng/giảm cần dựa số liệu. Không chọn/tune V5 nhiều lần dựa validation.'}
    (out/'SO_SANH_V4_V9.json').write_text(json.dumps(detail,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Kết quả trên CÙNG validation:')
    for n,s in scores.items():print(f'  {n}: Macro-F1={s:.4f}, metal→glass={metal_glass[n]["metal_pred_glass"]}')
    print('CNN - Baseline:',round(detail['cnn_minus_baseline_macro_f1'],4))
    print('Lưu:',out/'SO_SANH_V4_V9.json')
    return detail

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--old',default='results/validation_predictions.csv')
    p.add_argument('--new',default='results_cnn/predictions_efficientnet_b2.csv')
    p.add_argument('--out',default='results_cnn')
    a=p.parse_args();analyze(a.old,a.new,a.out)
