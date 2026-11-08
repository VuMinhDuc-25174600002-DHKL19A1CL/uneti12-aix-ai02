"""Pure helpers for AI02 V10 presentation. No model training or checkpoint loading."""
from __future__ import annotations
import csv
import hashlib
import html
import io
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

LABELS = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
VI = {"cardboard":"Bìa carton", "glass":"Thủy tinh", "metal":"Kim loại", "paper":"Giấy", "plastic":"Nhựa", "trash":"Rác khác"}


def safe_text(value):
    return html.escape(str(value), quote=True)


def model_fingerprint(selection):
    """Change result identity if model file is replaced and browser session stays open."""
    payload = [selection.get("name", "")]
    for part in selection.get("paths", []):
        path = Path(part)
        try:
            s = path.stat()
            payload.append(f"{path}:{s.st_size}:{s.st_mtime_ns}")
        except OSError:
            payload.append(f"{path}:MISSING")
    return "|".join(payload)


def image_fingerprint(blob, selection):
    return hashlib.sha256(blob).hexdigest() + ":" + model_fingerprint(selection)


def top_two(result):
    scores = result.get("probabilities", {})
    valid = [(key, max(0.0, min(1.0, float(scores.get(key, 0) or 0)))) for key in LABELS]
    valid.sort(key=lambda pair: pair[1], reverse=True)
    return valid[:2]


def decision_message(result, minimum=0.60, min_margin=0.15):
    top = top_two(result)
    if not top:
        return ("warning", "Chưa có xác suất hợp lệ để đánh giá.")
    gap = top[0][1] - (top[1][1] if len(top)>1 else 0.0)
    if result.get("uncertain") or top[0][1] < minimum or gap < min_margin:
        return ("warning", "Dự đoán chưa chắc chắn. Hãy kiểm tra thủ công trước khi phân loại.")
    return ("ok", "Mô hình có độ tự tin tương đối cao; kết quả vẫn chỉ mang tính tham khảo.")


def probability_bars_html(result):
    top = {lab for lab, _ in top_two(result)[:1]}
    parts = []
    for label, prob in sorted(
        ((lbl, max(0.0, min(1.0, float(result.get("probabilities", {}).get(lbl, 0) or 0)))) for lbl in LABELS),
        key=lambda pair: pair[1], reverse=True):
        name = safe_text(VI[label])
        width = f"{prob*100:.1f}%"
        highlight = " primary" if label in top else ""
        parts.append(f'<div class="prob-row"><div class="prob-caption"><span>{name}</span><strong>{prob*100:.1f}%</strong></div>'
                     f'<div class="prob-track"><div class="prob-fill{highlight}" style="width:{width}"></div></div></div>')
    return '<div class="probabilities">' + "".join(parts) + '</div>'


def time_vn():
    return datetime.now(timezone(timedelta(hours=7))).strftime("%Y-%m-%d %H:%M:%S +07:00")


def record_prediction(file_name, model_name, result, source="Ảnh đơn"):
    top = top_two(result)
    second = top[1] if len(top)>1 else ("", 0.)
    return {"thoi_gian": time_vn(), "nguon": str(source), "anh": str(file_name),
            "mo_hinh": str(model_name), "nhan": str(result.get("label", "")),
            "nhan_tieng_viet": str(result.get("label_vi", "")),
            "xac_suat_%": round(100*float(result.get("confidence", 0) or 0), 2),
            "nhan_thu_hai": str(VI.get(second[0], second[0])),
            "xac_suat_thu_hai_%": round(100*second[1], 2),
            "can_kiem_tra": bool(decision_message(result)[0] == "warning")}


def csv_export(rows):
    if not rows:
        return "".encode("utf-8-sig")
    buf = io.StringIO(newline="")
    columns = []
    for row in rows:
        for name in row.keys():
            if name not in columns:
                columns.append(name)
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: sanitize_spreadsheet_value(row.get(key, "")) for key in columns})
    return buf.getvalue().encode("utf-8-sig")


def sanitize_spreadsheet_value(value):
    """Avoid formula injection when CSV opened in Excel or Google Sheets."""
    if isinstance(value, str) and value and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def summarize_metrics(path):
    """Return numbers only when the user's real results file has them."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    output = {"file": str(p), "raw": doc}
    for name in ("macro_f1", "accuracy", "validation_macro_f1", "val_macro_f1"):
        val=doc.get(name)
        if isinstance(val, (int,float)) and not isinstance(val, bool):
            output[name] = round(float(val), 4)
    return output
