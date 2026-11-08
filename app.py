"""SMART CAMPUS AI02 | Web V13 (recyclability + multi-view).

Run from the AI02 project root: python -m streamlit run app.py
Uses existing trusted models from model/ via inference_bridge.py.
Does not retrain, fabricate predictions, or save uploaded images.
"""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from inference_bridge import VI, discover_models, predict_pil
from recycling_advice import CONDITIONS, COLLECTION, assess_recycling
from multiview_inference import dedupe_files_by_contents, fuse_predictions, image_exposure_hint
from ui_v10_utils import (
    csv_export, decision_message, image_fingerprint, model_fingerprint,
    probability_bars_html, record_prediction, safe_text,
    summarize_metrics, top_two,
)

st.set_page_config(
    page_title="SMART CAMPUS AI | UNETI 2026 · AI02",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
PAGES = (
    "Nhận diện ảnh", "Nhận diện đa góc", "So sánh mô hình", "Phân tích hàng loạt",
    "Lịch sử & Xuất CSV", "Dữ liệu & Phương pháp",
)
MAX_IMAGE_MB = 12
MAX_PIXELS = 25_000_000
MAX_BATCH = 20
MAX_VIEWS = 6
MAX_HISTORY = 150

st.markdown("""
<style>
:root {--blue:#116bb8; --navy:#143754; --orange:#f18a1c; --ink:#263e53;
       --muted:#5d7183; --line:#dbe5ed; --background:#f6f9fc;}
.stApp {background:var(--background);color:var(--ink);}
[data-testid="stHeader"] {background:var(--background);}
.block-container {max-width:1260px;padding-top:.85rem;padding-bottom:2.4rem;}
[data-testid="stSidebar"] {background:#ffffff;border-right:1px solid var(--line);}
[data-testid="stSidebar"] [data-testid="stRadio"] label {font-size:.91rem;}
.stButton > button[kind="primary"] {background:var(--blue);color:#fff;border-radius:11px;
    min-height:50px;font-size:1rem;font-weight:750;border:0;}
.stButton > button[kind="primary"]:hover {background:#0c528d;color:#fff;}
.stButton > button:not([kind="primary"]), [data-testid="stDownloadButton"] button {
    border-radius:10px;min-height:42px;border-color:#ccdbe7;}
[data-testid="stFileUploader"] section {border:2px dashed #a8cbe7;
    border-radius:13px;background:#f3f8fc;}
[data-testid="stFileUploader"] section:hover {border-color:#3184ce;background:#eef6ff;}
[data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--line)!important;border-radius:16px;}
[data-testid="stAlert"] {border-radius:11px;}
.stImage img {border-radius:12px;max-height:430px;object-fit:contain;}
.brand {background:var(--blue);border-radius:13px;color:#fff;padding:14px 15px;margin:0 0 14px;}
.brand strong {font-size:1.04rem;letter-spacing:.025em;display:block;color:#fff;}
.brand small {font-size:.73rem;display:block;color:#edf7ff;margin-top:4px;}
.hero {background:#fff;border:1px solid var(--line);border-radius:17px;
    padding:20px 24px;margin:0 0 15px;}
.hero .tag {color:#b36811;font-size:.73rem;letter-spacing:.07em;font-weight:800;}
.hero h1 {font-size:clamp(1.65rem,2.4vw,2.45rem);color:var(--navy);font-weight:850;
    margin:7px 0 8px;line-height:1.2;letter-spacing:-.025em;}
.hero h1 span {color:var(--blue);}
.hero p {margin:0;color:var(--muted);font-size:.98rem;line-height:1.5;}
.status {background:#eaf4fd;border:1px solid #cfe4f7;color:#234e70;border-radius:12px;
    padding:11px 15px;margin:0 0 17px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:.88rem;}
.dot {height:9px;width:9px;display:inline-block;border-radius:50%;background:#219c82;margin-right:2px;}
.dot.missing {background:var(--orange);}
.section-heading {font-weight:850;color:var(--navy);font-size:1.3rem;margin:0 0 4px;}
.section-subtitle {color:var(--muted);font-size:.91rem;margin:0 0 14px;}
.card-title {font-size:1.02rem;font-weight:800;color:var(--navy);margin-bottom:8px;}
.prediction {background:#fff8f0;border:1px solid #f5d6b4;border-left:5px solid var(--orange);
    border-radius:14px;padding:17px 20px;margin:10px 0 13px;}
.prediction .overline {font-size:.77rem;text-transform:uppercase;letter-spacing:.06em;color:#926014;font-weight:800;}
.prediction .name {font-size:clamp(1.8rem,3vw,2.4rem);color:#1365a8;font-weight:850;line-height:1.25;margin:6px 0;}
.prediction .confidence {font-size:1.11rem;font-weight:800;color:#94530d;}
.prediction .model {font-size:.87rem;color:#586c7d;margin-top:8px;}
.message {padding:12px 14px;margin:9px 0 14px;border:1px solid #b9e5dc;
    background:#eefaf7;color:#286351;border-radius:11px;line-height:1.55;font-size:.92rem;}
.message.warning {background:#fff5e9;border-color:#f1d2ac;color:#88530f;}
.probabilities {margin:8px 0 13px;}
.prob-row {margin:0 0 11px;}
.prob-caption {display:flex;justify-content:space-between;gap:10px;margin-bottom:5px;font-size:.91rem;}
.prob-caption strong {font-variant-numeric:tabular-nums;font-weight:800;}
.prob-track {height:11px;background:#e7eef5;border-radius:30px;overflow:hidden;}
.prob-fill {background:#237fce;border-radius:30px;height:100%;}
.prob-fill.primary {background:var(--orange);}
.guidance {border-radius:11px;background:#eff6fd;border:1px solid #d6e6f4;
    padding:13px 15px;line-height:1.6;color:#244d70;font-size:.92rem;}
.empty {border:1px dashed #bfcfdd;border-radius:13px;min-height:170px;
    display:flex;align-items:center;justify-content:center;flex-direction:column;
    text-align:center;padding:22px;color:#6b8092;background:#f9fcfe;}
.empty strong {font-size:1.03rem;color:#3e5a6f;margin-top:8px;}
.empty .mark {font-size:2.2rem;line-height:1;opacity:.8;}
.recycle-box {border:1px solid #bcdacb;border-left:5px solid #219c73;border-radius:12px;
    background:#eff9f4;padding:15px 18px;margin:9px 0 13px;color:#224e3c;}
.recycle-box.unknown,.recycle-box.conditional {border-color:#ead09e;border-left-color:#de8c25;background:#fff9ed;color:#755013;}
.recycle-box.separate,.recycle-box.no_for_collection {border-color:#eac7c7;border-left-color:#c76565;background:#fff4f4;color:#803f3f;}
.recycle-box .recycle-title {font-weight:850;font-size:.9rem;letter-spacing:.02em;}
.recycle-box .recycle-answer {font-size:1.07rem;font-weight:750;margin:5px 0 6px;}
.recycle-box .recycle-desc {font-size:.9rem;line-height:1.55;}
.foot {border-top:1px solid var(--line);margin-top:25px;padding-top:12px;
    font-size:.78rem;line-height:1.6;color:#64798c;}
@media(max-width:780px){.block-container{padding-left:.85rem;padding-right:.85rem;}
    .hero{padding:16px 17px}.prediction{padding:14px}.empty{min-height:125px}}
</style>
""", unsafe_allow_html=True)


def read_uploaded_image(file_obj):
    """Validate bytes and EXIF orientation; never read user-supplied server paths."""
    if file_obj is None:
        return None, None, None
    raw = file_obj.getvalue()
    if not raw:
        raise ValueError("Ảnh trống. Bạn hãy chọn lại một file ảnh.")
    if len(raw) > MAX_IMAGE_MB * 1024 * 1024:
        raise ValueError(f"Ảnh lớn hơn {MAX_IMAGE_MB} MB. Hãy giảm dung lượng ảnh.")
    try:
        with Image.open(BytesIO(raw)) as opened:
            if opened.width * opened.height > MAX_PIXELS:
                raise ValueError("Ảnh vượt quá 25 megapixel; hãy giảm kích thước ảnh.")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("File không phải ảnh hợp lệ hoặc ảnh đã bị hỏng.") from error
    return image, raw, getattr(file_obj, "name", "anh_camera.jpg")


def technical_error_hint(error):
    """Give practical recovery steps without claiming the prediction succeeded."""
    message = str(error)
    low = message.lower()
    if isinstance(error, (ModuleNotFoundError, ImportError)):
        return "Thiếu thư viện. Trong Terminal dự án, chạy: `..\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt -r requirements_cnn.txt`."
    if isinstance(error, FileNotFoundError) or "no such file" in low:
        return "Không tìm thấy file mô hình. Kiểm tra thư mục `model/`, sau đó khởi động lại ứng dụng."
    if "out of memory" in low or "allocate memory" in low or "defaultcpuallocator" in low:
        return "Máy thiếu RAM để nạp CNN. Hãy đóng ứng dụng nặng khác, thử SVM hoặc MobileNetV3 và khởi động lại web."
    if "state_dict" in low or "checkpoint" in low or "invalid load key" in low:
        return "Checkpoint không tương thích hoặc chưa huấn luyện hoàn chỉnh. Kiểm tra lại file `.pth` của mô hình đã chọn."
    if "size mismatch" in low or "architecture" in low:
        return "Kiến trúc trong checkpoint khác với code dự đoán. Cần dùng đúng bộ huấn luyện V9 đi kèm."
    return "Hãy kiểm tra file mô hình và thông báo kỹ thuật ở phần 'Chi tiết lỗi'."


def predict_safe(image, selected_model):
    """Run the real inference adapter; never return a made-up prediction."""
    return predict_pil(image, selected_model)


def render_prediction(result, selected_model_name):
    label = safe_text(result.get("label_vi") or VI.get(result.get("label", ""), "Chưa xác định"))
    score = max(0.0, min(1.0, float(result.get("confidence", 0) or 0)))
    st.markdown(
        '<div class="prediction"><div class="overline">KẾT QUẢ NHẬN DIỆN</div>'
        f'<div class="name">{label}</div><div class="confidence">{100*score:.1f}% · Mức tin cậy tham khảo</div>'
        f'<div class="model">Mô hình: {safe_text(selected_model_name)}</div></div>',
        unsafe_allow_html=True,
    )
    status, message = decision_message(result)
    st.markdown(f'<div class="message {"warning" if status == "warning" else ""}">'
                f'{"⚠" if status == "warning" else "✓"} {safe_text(message)}</div>',
                unsafe_allow_html=True)
    ranked = top_two(result)
    if len(ranked) >= 2:
        second, second_score = ranked[1]
        st.caption(f"Nhóm đứng thứ hai: {VI.get(second,second)} ({100*second_score:.1f}%) · "
                   f"Chênh lệch {(ranked[0][1]-second_score)*100:.1f} điểm %")
    render_recycling(result)
    st.markdown("**Xác suất 6 nhóm rác**")
    st.markdown(probability_bars_html(result), unsafe_allow_html=True)
    st.markdown("**Gợi ý phân loại**")
    recommendation = result.get("recommendation") or "Vui lòng kiểm tra vật liệu thực tế và quy định thu gom địa phương."
    st.markdown(f'<div class="guidance">{safe_text(recommendation)}</div>', unsafe_allow_html=True)
    st.caption(result.get("note") or "Xác suất không phải cam kết kết quả đúng tuyệt đối.")


def recycle_config():
    """User-supplied information; never guessed from an image."""
    return {
        "condition": st.session_state.get("ai02v13_condition", CONDITIONS[0]),
        "collection": st.session_state.get("ai02v13_collection", COLLECTION[0]),
        "special": st.session_state.get("ai02v13_special", False),
    }


def recycle_for(result):
    return assess_recycling(result.get("label", "trash"),
                            prediction_uncertain=decision_message(result)[0] == "warning",
                            **recycle_config())


def render_recycling(result):
    info = recycle_for(result)
    st.markdown("**♻️ Rác này có thể tái chế không?**")
    st.markdown(
        '<div class="recycle-box ' + safe_text(info["status"]) + '">'
        '<div class="recycle-title">' + safe_text(info["title"]) + '</div>'
        '<div class="recycle-answer">' + safe_text(info["answer"]) + '</div>'
        '<div class="recycle-desc">' + safe_text(info["explanation"]) + '</div></div>',
        unsafe_allow_html=True)
    st.markdown("**Bạn nên làm gì?**")
    for step in info["steps"]:
        st.markdown("- " + step)
    st.caption("Độ sạch, vật phẩm đặc biệt và nơi thu gom được chỉnh ở mục ♻️ Điều kiện tái chế bên trái. "
               "AI chỉ nhận diện nhóm vật liệu, không xác nhận chắc chắn khả năng tái chế.")


def empty_view(icon, title, subtitle):
    st.markdown(f'<div class="empty"><div class="mark">{safe_text(icon)}</div>'
                f'<strong>{safe_text(title)}</strong><div>{safe_text(subtitle)}</div></div>',
                unsafe_allow_html=True)


def add_history(filename, selected_model, result, source):
    row = record_prediction(filename, selected_model["name"], result, source)
    advisory = recycle_for(result)
    row["tinh_trang_tai_che"] = advisory["title"]
    row["goi_y_tai_che"] = advisory["answer"]
    row["so_goc_chup"] = int(result.get("view_count", 1))
    st.session_state["ai02v12_history"] = ([row] + st.session_state["ai02v12_history"])[:MAX_HISTORY]


def current_result(storage_key, token):
    entry = st.session_state.get(storage_key)
    return entry.get("value") if entry and entry.get("token") == token else None


if "ai02v12_history" not in st.session_state:
    st.session_state["ai02v12_history"] = []

try:
    models = discover_models(MODEL_DIR)
    model_listing_error = None
except Exception as exc:
    models = []
    model_listing_error = str(exc)

st.sidebar.markdown('<div class="brand"><strong>♻ SMART CAMPUS AI</strong>'
                    '<small>UNETI 2026 · AI02 · TRƯỜNG HỌC XANH</small></div>', unsafe_allow_html=True)
page = st.sidebar.radio("Chức năng", PAGES, key="ai02v12_page")
st.sidebar.divider()
st.sidebar.markdown("**Mô hình dự đoán**")
if models:
    index = st.sidebar.selectbox(
        "Chọn mô hình", options=list(range(len(models))),
        format_func=lambda i: models[i]["name"], key="ai02v12_model_index",
    )
    selected_model = models[index]
    st.sidebar.caption("CNN (PyTorch)" if selected_model["kind"] == "cnn" else "Baseline / SVM (scikit-learn)")
    if selected_model["kind"] == "classic":
        st.sidebar.info("Mô hình đang chọn thuộc nhóm truyền thống. Để thử CNN V9, cần có checkpoint `.pth` đã huấn luyện.")
else:
    selected_model = None
    st.sidebar.error("Chưa tìm thấy model. Hãy đặt checkpoint `.pth` hoặc `.joblib` vào thư mục `model/`.")
    if model_listing_error:
        st.sidebar.caption("Lỗi phát hiện mô hình: " + model_listing_error)

with st.sidebar.expander("♻️ Điều kiện tái chế", expanded=True):
    st.caption("Ảnh chỉ gợi ý nhóm vật liệu. Hãy bổ sung thông tin về vật thật.")
    st.selectbox("Tình trạng vật phẩm", CONDITIONS, key="ai02v13_condition")
    st.selectbox("Điểm thu gom gần bạn", COLLECTION, key="ai02v13_collection")
    st.checkbox("Là pin, điện tử, bóng đèn, hóa chất hoặc vật sắc nhọn", key="ai02v13_special")
    st.caption("Chưa biết? Giữ lựa chọn mặc định. Không mặc định tất cả rác là tái chế được.")

with st.sidebar.expander("Hướng dẫn dùng camera", expanded=False):
    st.write("**Trên chính máy chạy web:** mở `http://localhost:8501` rồi cho phép camera trong trình duyệt.")
    st.write("**Trên điện thoại/máy khác:** địa chỉ HTTP dạng `172.20.x.x:8501` thường bị chặn camera. "
             "Hãy tải ảnh lên hoặc triển khai qua HTTPS.")
st.sidebar.caption("Ảnh chỉ được dùng để dự đoán trong phiên. Ứng dụng không tự thêm ảnh vào bộ dữ liệu huấn luyện.")

st.markdown('''<div class="hero"><div class="tag">OLYMPIC AI · UNETI 2026 · AI02</div>
<h1>Nhận diện rác thải <span>đa góc & tái chế</span></h1>
<p>Tải ảnh hoặc nhiều góc chụp → AI phân loại → tư vấn khả năng tái chế. Giữ nguyên kết quả thực nghiệm của bạn. "Trường học xanh" – đội UNETI12 · ODD3.</p>
</div>''', unsafe_allow_html=True)
status_name = safe_text(selected_model["name"]) if selected_model else "Chưa có checkpoint"
st.markdown(f'<div class="status"><span class="dot {"" if selected_model else "missing"}"></span>'
            f'<strong>Mô hình đang chọn:</strong> {status_name} &nbsp;·&nbsp; '
            f'{len(models)} lựa chọn &nbsp;·&nbsp; 6 nhóm rác &nbsp;·&nbsp; Đa góc 2–6 ảnh</div>', unsafe_allow_html=True)

if page == "Nhận diện ảnh":
    st.markdown('<div class="section-heading">Nhận diện một ảnh</div>'
                '<div class="section-subtitle">Bước 1: tải ảnh lên. Bước 2: nhấn nhận diện. Bạn chỉ cần mở camera nếu thực sự muốn chụp ảnh mới.</div>',
                unsafe_allow_html=True)
    left, right = st.columns([1.0, 1.03], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="card-title">01 · Ảnh đầu vào</div>', unsafe_allow_html=True)
            source = st.radio("Chọn nguồn ảnh", ["Tải ảnh từ máy", "Dùng camera"],
                              index=0, horizontal=True, key="ai02v12_source")
            upload = None
            if source == "Tải ảnh từ máy":
                upload = st.file_uploader(
                    "Chọn ảnh JPG, PNG, WEBP hoặc BMP", type=["jpg", "jpeg", "png", "webp", "bmp"],
                    key="ai02v12_upload",
                )
                st.caption("Mặc định chọn tải ảnh để hoạt động ổn định trên mọi địa chỉ truy cập.")
            else:
                st.info("Camera chỉ hoạt động ổn định khi mở `http://localhost:8501` trên chính máy chạy app, "
                        "hoặc dùng trang HTTPS. Nếu đang mở IP LAN `172.20.x.x:8501`, hãy chọn **Tải ảnh từ máy**.")
                confirmed = st.checkbox(
                    "Tôi đang truy cập bằng localhost hoặc HTTPS và muốn bật camera",
                    key="ai02v12_enable_camera",
                    value=False,
                )
                if confirmed:
                    upload = st.camera_input("Chụp ảnh rác", key="ai02v12_camera")
                else:
                    st.caption("Camera đang tắt để tránh lỗi xin quyền hoặc khung chụp ảnh bị treo.")

            photo = blob = filename = None
            if upload is not None:
                try:
                    photo, blob, filename = read_uploaded_image(upload)
                    st.image(ImageOps.contain(photo, (1100, 650)), use_container_width=True)
                    st.caption(f"{filename} · {photo.width} × {photo.height} px · {len(blob)/1024:.1f} KB")
                except ValueError as exc:
                    st.error(str(exc))
            else:
                empty_view("▧", "Chưa có ảnh", "Tải lên một ảnh rác để bắt đầu")
            st.divider()
            st.markdown('<div class="card-title">02 · Phân tích</div>', unsafe_allow_html=True)
            clicked = st.button("PHÂN TÍCH HÌNH ẢNH", type="primary", use_container_width=True,
                                disabled=(photo is None or selected_model is None), key="ai02v12_predict")
    with right:
        with st.container(border=True):
            st.markdown('<div class="card-title">Kết quả nhận diện</div>', unsafe_allow_html=True)
            token = image_fingerprint(blob, selected_model) if blob is not None and selected_model else None
            if clicked and photo is not None and selected_model is not None:
                try:
                    with st.spinner("Mô hình đang phân tích. CNN trên CPU có thể chậm ở lượt đầu..."):
                        outcome = predict_safe(photo, selected_model)
                    st.session_state["ai02v12_result"] = {"token": token, "value": outcome}
                    add_history(filename, selected_model, outcome, "Ảnh đơn/camera")
                except Exception as exc:
                    st.session_state.pop("ai02v12_result", None)
                    st.error("Không thể dự đoán bằng mô hình đang chọn.")
                    st.info(technical_error_hint(exc))
                    with st.expander("Chi tiết lỗi để sửa trong VS Code"):
                        st.code(f"{type(exc).__name__}: {exc}")
            outcome = current_result("ai02v12_result", token)
            if outcome:
                render_prediction(outcome, selected_model["name"])
                with st.expander("Dữ liệu kỹ thuật"):
                    st.json(outcome)
                st.download_button("Tải kết quả JSON", data=json.dumps(outcome, indent=2, ensure_ascii=False),
                                   file_name="AI02_ket_qua.json", mime="application/json",
                                   use_container_width=True, key="ai02v12_json")
            else:
                empty_view("◎", "Chưa có kết quả", "Ảnh và mô hình phải được chọn trước khi phân tích")

elif page == "Nhận diện đa góc":
    st.markdown('<div class="section-heading">Nhận diện một vật từ nhiều góc chụp</div>'
                '<div class="section-subtitle">Chọn 2–6 ảnh KHÁC GÓC của CÙNG MỘT vật. AI chạy cùng một mô hình trên từng ảnh, rồi tổng hợp xác suất.</div>',
                unsafe_allow_html=True)
    st.info("Mẹo chụp: chụp trực diện, nghiêng trái/phải và mặt trên; đủ sáng, vật thể chiếm phần lớn khung hình. "
            "Không tải 6 vật khác nhau hay 6 bản sao của cùng ảnh. Chụp nhiều góc không bảo đảm AI luôn đúng.")
    left, right = st.columns([1.0, 1.03], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="card-title">01 · Tải từ 2–6 ảnh</div>', unsafe_allow_html=True)
            multi_files = st.file_uploader(
                "Chọn ảnh một vật từ các góc khác nhau", accept_multiple_files=True,
                type=["jpg", "jpeg", "png", "webp", "bmp"], key="ai02v13_multi_files")
            if len(multi_files) > MAX_VIEWS:
                st.warning(f"Chỉ dùng tối đa {MAX_VIEWS} ảnh đầu tiên trong một lượt.")
            unique_files, duplicate_names = dedupe_files_by_contents(multi_files[:MAX_VIEWS])
            if duplicate_names:
                st.warning("Đã bỏ ảnh trùng hoàn toàn: " + ", ".join(duplicate_names))
            photos, names, errors = [], [], []
            for obj in unique_files:
                try:
                    image, _, name = read_uploaded_image(obj)
                    # Inference uses 224–260px crops; cap multi-view working RAM.
                    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
                    photos.append(image)
                    names.append(name)
                    hint = image_exposure_hint(image)
                    if hint: st.caption(name + ": " + hint)
                except ValueError as exc:
                    errors.append(f"{getattr(obj,'name','Ảnh')}: {exc}")
            if errors:
                st.error("; ".join(errors))
            if photos:
                for i in range(0,len(photos),3):
                    cols = st.columns(min(3, len(photos)-i), gap="small")
                    for j,col in enumerate(cols):
                        k=i+j
                        with col:
                            st.image(photos[k], use_container_width=True)
                            st.caption(f"Góc {k+1}: {names[k]}")
            else:
                empty_view("▧", "Chưa có ảnh đa góc", "Hãy tải 2–6 ảnh của cùng một vật")
            confirmed = st.checkbox("Tôi xác nhận các ảnh là cùng MỘT vật thể", key="ai02v13_same_object")
            enough = 2 <= len(photos) <= MAX_VIEWS and not errors and confirmed
            clicked = st.button("NHẬN DIỆN ĐA GÓC", type="primary", use_container_width=True,
                                disabled=(not enough or selected_model is None), key="ai02v13_multi_run")
            if photos and len(photos) < 2:
                st.caption("Cần ít nhất 2 ảnh KHÁC góc, không tính ảnh trùng.")
    with right:
        with st.container(border=True):
            st.markdown('<div class="card-title">02 · Tổng hợp từ các góc chụp</div>', unsafe_allow_html=True)
            token = None
            if photos and selected_model and not errors:
                individual_hashes = [hashlib.sha256(f.getvalue()).hexdigest() for f in unique_files]
                token = model_fingerprint(selected_model) + "|" + "|".join(individual_hashes)
            if clicked:
                try:
                    with st.spinner("Đang dự đoán từng góc ảnh (CNN trên CPU có thể chậm)..."):
                        partial = [predict_safe(img, selected_model) for img in photos]
                        outcome = fuse_predictions(partial, filenames=names)
                    st.session_state["ai02v13_multiview_result"] = {"token":token,"value":outcome}
                    add_history("; ".join(names), selected_model, outcome, "Nhiều góc")
                except Exception as exc:
                    st.session_state.pop("ai02v13_multiview_result", None)
                    st.error("Không thể tổng hợp các góc ảnh")
                    st.info(technical_error_hint(exc))
                    with st.expander("Chi tiết lỗi"):
                        st.code(f"{type(exc).__name__}: {exc}")
            outcome = current_result("ai02v13_multiview_result", token)
            if outcome:
                render_prediction(outcome, selected_model["name"] + " · đa góc")
                st.metric("Mức đồng thuận giữa các góc", f"{outcome['view_agreement']*100:.0f}%")
                if outcome["disagreement"]:
                    st.warning("Các góc ảnh bất đồng về nhãn. Đừng coi kết quả tổng hợp là chắc chắn; hãy kiểm tra vật thật hoặc chụp lại.")
                st.markdown("**Dự đoán từng góc**")
                st.dataframe(outcome["views"], use_container_width=True, hide_index=True)
                st.download_button("TẢI KẾT QUẢ ĐA GÓC (JSON)",
                                   data=json.dumps(outcome,ensure_ascii=False,indent=2),
                                   mime="application/json",file_name="AI02_da_goc.json",
                                   use_container_width=True,key="ai02v13_multiview_json")
            else:
                empty_view("◎", "Chưa có kết quả đa góc", "Chọn ít nhất 2 góc, xác nhận cùng một vật rồi nhấn Nhận diện đa góc")

elif page == "So sánh mô hình":
    st.markdown('<div class="section-heading">So sánh hai mô hình</div>'
                '<div class="section-subtitle">Đối chiếu dự đoán của Baseline, SVM và CNN trên cùng một ảnh.</div>',
                unsafe_allow_html=True)
    st.caption("Đây là so sánh trên một ảnh; không thay thế Macro-F1 và ma trận nhầm lẫn trên validation.")
    source_file = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png", "webp", "bmp"], key="ai02v12_comparison_upload")
    selected_indices = st.multiselect("Chọn tối đa 2 mô hình", list(range(len(models))),
                                      default=list(range(min(2, len(models)))),
                                      format_func=lambda i: models[i]["name"], max_selections=2,
                                      key="ai02v12_comparison_models")
    photo = blob = filename = None
    if source_file is not None:
        try:
            photo, blob, filename = read_uploaded_image(source_file)
            st.image(ImageOps.contain(photo, (360, 300)), width=340)
        except ValueError as exc:
            st.error(str(exc))
    token = None
    if blob is not None and selected_indices:
        token = hashlib.sha256(blob).hexdigest() + "|" + "|".join(model_fingerprint(models[i]) for i in selected_indices)
    compare_clicked = st.button("SO SÁNH", type="primary", disabled=(photo is None or not selected_indices),
                                key="ai02v12_compare_button")
    if compare_clicked:
        compared = []
        for ix in selected_indices:
            selection = models[ix]
            try:
                with st.spinner("Đang chạy " + selection["name"]):
                    result = predict_safe(photo, selection)
                compared.append((selection["name"], result))
                add_history(filename, selection, result, "So sánh")
            except Exception as exc:
                st.warning(f"Không thể chạy {selection['name']}: {technical_error_hint(exc)}")
                with st.expander("Lỗi kỹ thuật: " + selection["name"]):
                    st.code(f"{type(exc).__name__}: {exc}")
        st.session_state["ai02v12_comparison"] = {"token": token, "value": compared}
    compared = current_result("ai02v12_comparison", token)
    if compared:
        for column, (name, outcome) in zip(st.columns(len(compared)), compared):
            with column:
                with st.container(border=True):
                    render_prediction(outcome, name)

elif page == "Phân tích hàng loạt":
    st.markdown('<div class="section-heading">Phân tích hàng loạt</div>'
                '<div class="section-subtitle">Tối đa 20 ảnh trong một lượt; kết quả có thể xuất CSV cho Excel.</div>',
                unsafe_allow_html=True)
    batch_files = st.file_uploader("Chọn nhiều ảnh", accept_multiple_files=True,
                                   type=["jpg", "jpeg", "png", "webp", "bmp"], key="ai02v12_batch_upload")
    if batch_files and len(batch_files) > MAX_BATCH:
        st.warning(f"Đã chọn {len(batch_files)} ảnh; ứng dụng sẽ xử lý {MAX_BATCH} ảnh đầu tiên.")
    token = None
    if batch_files and selected_model:
        hashes = [hashlib.sha256(f.getvalue()).hexdigest() for f in batch_files[:MAX_BATCH]]
        token = model_fingerprint(selected_model) + "|" + "|".join(hashes)
    batch_clicked = st.button("PHÂN TÍCH HÀNG LOẠT", type="primary",
                              disabled=(not batch_files or selected_model is None), key="ai02v12_batch_button")
    if batch_clicked:
        rows = []
        total = min(len(batch_files), MAX_BATCH)
        indicator = st.progress(0, text="Đang chuẩn bị...")
        for idx, file_obj in enumerate(batch_files[:MAX_BATCH]):
            try:
                photo, _, filename = read_uploaded_image(file_obj)
                result = predict_safe(photo, selected_model)
                row = record_prediction(filename, selected_model["name"], result, "Hàng loạt")
                advice = recycle_for(result)
                row["tinh_trang_tai_che"] = advice["title"]
                row["goi_y_tai_che"] = advice["answer"]
                row["so_goc_chup"] = 1
                rows.append(row)
                add_history(filename, selected_model, result, "Hàng loạt")
            except Exception as exc:
                rows.append({"thoi_gian":"", "nguon":"Hàng loạt", "anh":file_obj.name,
                             "mo_hinh":selected_model["name"], "nhan":"ERROR",
                             "nhan_tieng_viet":"Lỗi xử lý ảnh", "xac_suat_%":"",
                             "nhan_thu_hai":"", "xac_suat_thu_hai_%":"",
                             "can_kiem_tra":True, "tinh_trang_tai_che":"Chưa xác định",
                             "goi_y_tai_che":"Ảnh lỗi, chưa phân loại", "so_goc_chup":0,
                             "loi":f"{type(exc).__name__}: {exc}"})
            indicator.progress((idx+1)/total, text=f"Đã xử lý {idx+1}/{total} ảnh")
        st.session_state["ai02v12_batch_result"] = {"token": token, "value": rows}
    rows = current_result("ai02v12_batch_result", token)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.download_button("TẢI BẢNG CSV", data=csv_export(rows), mime="text/csv",
                           file_name="AI02_hang_loat.csv", use_container_width=True,
                           key="ai02v12_batch_download")

elif page == "Lịch sử & Xuất CSV":
    st.markdown('<div class="section-heading">Lịch sử dự đoán</div>'
                '<div class="section-subtitle">Kết quả chỉ được lưu trong bộ nhớ phiên web hiện tại.</div>',
                unsafe_allow_html=True)
    records = st.session_state.get("ai02v12_history", [])
    if records:
        st.dataframe(records, use_container_width=True, hide_index=True)
        l, r = st.columns([2, 1], gap="medium")
        with l:
            st.download_button("TẢI LỊCH SỬ CSV", data=csv_export(records), mime="text/csv",
                               file_name="AI02_lich_su.csv", use_container_width=True,
                               key="ai02v12_history_download")
        with r:
            if st.button("XÓA LỊCH SỬ", use_container_width=True, key="ai02v12_history_clear"):
                st.session_state["ai02v12_history"] = []
                st.rerun()
    else:
        st.info("Chưa có lượt dự đoán nào trong phiên.")

elif page == "Dữ liệu & Phương pháp":
    st.markdown('<div class="section-heading">Dữ liệu & Phương pháp</div>'
                '<div class="section-subtitle">Thông tin trình bày cho hội đồng và kết quả đánh giá thực tế.</div>',
                unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Bài toán AI02 – Trường học xanh**")
        st.write("6 nhóm rác của bộ TrashNet: carton, thủy tinh, kim loại, giấy, nhựa, rác khác.")
        st.markdown("**Mở rộng V13**")
        st.write("Nhận diện đa góc: suy luận riêng trên 2–6 ảnh của cùng một vật, trung bình xác suất rồi đo tỷ lệ đồng thuận. Không phải mô hình được huấn luyện lại.")
        st.write("Đánh giá tiềm năng tái chế: quy tắc minh họa có điều kiện theo nhóm vật liệu + thông tin do người dùng khai báo; không tự xác định quy định thu gom địa phương.")
        st.markdown("**Baseline**")
        st.write("HOG + RGB histogram và Logistic Regression; đo Macro-F1, ma trận nhầm lẫn.")
        st.markdown("**Cải tiến**")
        st.write("SVM V4 sử dụng đặc trưng thủ công mở rộng; CNN V5/V9 có MobileNetV3, EfficientNet, ConvNeXt khi đã huấn luyện và lưu checkpoint thật.")
        st.markdown("**Nguyên tắc so sánh**")
        st.write("Đánh giá các mô hình trên cùng tập validation; không điều chỉnh mô hình theo nhãn validation cuối.")
        st.warning("Ứng dụng chỉ hỗ trợ học thuật và demo; không bảo đảm đúng mọi ảnh hoặc đánh giá chắc chắn khả năng tái chế.")
    st.markdown("**Số liệu đánh giá đã có trên máy**")
    result_paths = [ROOT / "results" / "metrics.json"]
    cnn_dir = ROOT / "results_cnn"
    if cnn_dir.is_dir():
        result_paths.extend(sorted(cnn_dir.glob("metrics_*.json")))
    n = 0
    for path in result_paths:
        values = summarize_metrics(path)
        if values is None:
            continue
        n += 1
        with st.expander(path.parent.name + "/" + path.name, expanded=(n == 1)):
            short = {k: v for k, v in values.items() if k not in ("raw", "file")}
            if short:
                st.json(short)
            st.json(values["raw"], expanded=False)
    if n == 0:
        st.info("Chưa có file metrics JSON trên máy. Cần đánh giá mô hình thật trước khi trình bày số điểm.")

st.markdown('<div class="foot">UNETI12 · ODD3 · Olympic AI UNETI 2026 · AI02 · Web V13<br>'
            'Dự đoán và xác suất chỉ mang tính tham khảo, không có mô hình đạt 100% trên mọi ảnh.</div>',
            unsafe_allow_html=True)
