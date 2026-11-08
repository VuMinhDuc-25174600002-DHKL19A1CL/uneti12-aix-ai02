# UNETI12 – AIX – AI02

Phân loại ảnh rác thành 6 nhóm bằng Baseline HOG + RGB/Logistic Regression, SVM cải tiến và CNN MobileNetV3-Large. Ứng dụng V13 chạy tại chỗ bằng Streamlit, có dự đoán từng ảnh, nhận diện đa góc 2–6 ảnh cùng một vật và tư vấn tái chế theo quy tắc tham khảo.

## Đội

- Mã đội: **UNETI12**
- Tên đội: **AIX**
- Mã đề: **AI02**
- Thành viên: **Bùi Trung Kiên; Trần Đức Quảng; Vũ Minh Đức**

## Cấu trúc gói nộp

```text
UNETI12_AIX_AI02/
├── slides.pdf
├── README.md
├── requirements.txt
├── app.py, predict_cli_v13.py, train_cnn.py, ...
├── src/                 # trích đặc trưng, huấn luyện, dự đoán, kiểm tra dữ liệu
├── model/               # baseline.joblib, improved.joblib, ai02_mobilenet_v3_large.pth
├── results/             # metrics, validation predictions, figures and tables
├── results_cnn/         # CNN metrics, validation predictions, confusion matrix and comparison
└── .streamlit/config.toml
```

Bộ ảnh không được chép vào gói; ứng dụng demo chỉ cần Python packages và các checkpoint trong `model/`. Dữ liệu cần khi huấn luyện lại; cách tải nằm ở mục “Huấn luyện lại”.

## Môi trường đã dùng để tạo kết quả

- Windows 11 x64, Python 3.14.0
- NumPy 2.5.4; SciPy 1.18.1; scikit-learn 1.9.1; scikit-image 0.26.0
- Pillow 12.3.0; Matplotlib 3.11.2; joblib 1.6.0
- Streamlit 1.67.0; PyTorch 2.15.0; Torchvision 0.29.2

Các phiên bản trực tiếp được ghim trong `requirements.txt` để phù hợp với môi trường đã tạo checkpoint. Cài package cần kết nối mạng; sau khi cài đặt xong, ứng dụng và suy luận bằng checkpoint cục bộ không cần Internet.

## Chạy demo local bằng Streamlit

Streamlit là công nghệ tạo giao diện web của sản phẩm; “local” là cách chạy giao diện đó ngay trên máy trình diễn. Đây là cùng một ứng dụng, không phải hai phiên bản. Lệnh dưới đây mở máy chủ chỉ trên máy hiện tại (`127.0.0.1`), sau đó truy cập bằng trình duyệt tại `http://localhost:8501`; không cần triển khai website lên Internet.

Từ thư mục gốc `UNETI12_AIX_AI02` trong PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Mở `http://localhost:8501`. Chọn `CNN V5 · MobileNetV3-Large` hoặc `SVM cải tiến V4` ở thanh bên. Dừng web bằng `Ctrl+C`. Có thể nhấp đúp `01_CHAY_LOCAL_V13.bat` sau khi môi trường đã cài đủ thư viện; BAT này chạy cùng lệnh Streamlit ở trên.

Các lệnh huấn luyện lại và `predict_cli_v13.py` là công cụ riêng cho tái lập thí nghiệm hoặc dự đoán từ PowerShell; không cần chạy chúng để mở giao diện demo.

### Đưa demo lên Streamlit Community Cloud

GitHub dùng để lưu mã nguồn; Streamlit Community Cloud chạy ứng dụng và cấp đường link web. Đẩy nội dung thư mục này lên GitHub sao cho `app.py`, `requirements.txt`, `model/`, `src/` và `.streamlit/` nằm ở thư mục gốc repository. Trong `share.streamlit.io`, kết nối GitHub, chọn **Create app**, chọn repository và nhánh, đặt **Main file path** là `app.py`; trong **Advanced settings**, chọn Python 3.14 rồi deploy. Cloud sẽ tạo môi trường và cài các gói từ `requirements.txt`; lần dựng đầu tiên sẽ tải PyTorch và có thể mất vài phút. Sau khi deploy xong, dùng URL `*.streamlit.app` để mở demo từ máy BTC mà không cần cài Python/thư viện trên máy đó.

Ứng dụng xử lý ảnh tải lên trên máy chủ Cloud để suy luận và không lưu ảnh tải lên thành tệp. Chỉ dùng ảnh demo phù hợp, không đưa dữ liệu cá nhân hoặc ảnh nhạy cảm lên dịch vụ công khai.

### Demo dự phòng khi mạng BTC gặp sự cố

Ứng dụng chạy local và không gọi dịch vụ web bên ngoài khi dự đoán; checkpoint đã có trong `model/`. Chuẩn bị thư viện trên máy demo trước buổi thi để chạy được hoàn toàn ngoại tuyến. Nếu máy BTC không có Python/thư viện cài sẵn và không có mạng để cài, mở `slides.pdf` cùng các hình/báo cáo đã lưu trong `results/` và `results_cnn/` để trình bày số liệu; không trình bày ảnh kết quả lưu sẵn như một suy luận live.

## Lệnh dự đoán một ảnh

```powershell
python predict_cli_v13.py --image "C:\duong-dan\anh-rac.jpg" --model mobilenet_v3_large
```

Các lựa chọn `--model`: `mobilenet_v3_large`, `improved`, `baseline`. Ảnh phải thuộc một trong sáu nhóm đã học; mô hình luôn trả về một trong sáu nhãn và có thể nhầm với vật ngoài phạm vi.

## Huấn luyện lại và tái lập đánh giá

Cần kết nối Internet để tải TrashNet và lần đầu lấy pretrained ImageNet weights của Torchvision. Dataset không nằm trong gói nộp.

```powershell
python download_dataset.py
python -m src.train_advanced --images "data/dataset-resized" --out "results" --model-dir "model"
python train_cnn.py --arch mobilenet_v3_large --images "data/dataset-resized" --split-manifest "results/split_manifest.csv" --model-dir "model" --results "results_cnn" --batch-size 8
python compare_results.py
```

Chương trình CNN tái sử dụng `results/split_manifest.csv`, giữ nguyên tập validation ngoài 505 ảnh của thí nghiệm này. Nếu thay đổi dữ liệu, cần tạo lại split trước khi huấn luyện và báo cáo kết quả mới; không so sánh với kết quả cũ như cùng một holdout.

## Kết quả thực nghiệm đã lưu

- Dữ liệu gốc: TrashNet, 2.527 ảnh; sau kiểm tra dữ liệu còn 2.521 ảnh hợp lệ. Chia stratified 80:20, seed 2026: train 2.016, validation 505.
- Baseline HOG + RGB histogram + Logistic Regression: Macro-F1 **0,5483**, accuracy **0,5802**.
- SVM cải tiến (đặc trưng màu/texture + RBF SVM): Macro-F1 **0,7206**, accuracy **0,7406**.
- CNN MobileNetV3-Large pretrained ImageNet: Macro-F1 **0,8663**, accuracy **0,8812**. Huấn luyện CNN dùng 1.713 ảnh fit, 303 ảnh internal-dev và giữ chung final validation 505 ảnh.
- Trên một tập validation, chênh lệch CNN − Baseline Macro-F1 là **+0,3180**; paired bootstrap CI 95% **[+0,2663; +0,3654]** (800 lần lấy mẫu). Đây là kết quả nội bộ trên TrashNet, không phải đánh giá độc lập ngoài thực địa.

Báo cáo và chứng cứ: `results/metrics.json`, `results/BAO_CAO_KET_QUA.md`, `results/02_confusion_baseline.png`, `results/03_confusion_improved.png`, `results/04_per_class_f1.png`, `results_cnn/metrics_mobilenet_v3_large.json`, `results_cnn/confusion_mobilenet_v3_large.png`, `results_cnn/SO_SANH_V4_V5.json`.

## Giới hạn và demo V13

TrashNet có nền chụp tương đối đơn giản và số ảnh lớp `trash` thấp hơn các lớp khác. Macro-F1 trên validation không đảm bảo đúng tương tự với ảnh điện thoại hoặc rác chụp trong môi trường khác. Bộ nhãn không có điện tử/pin. Nhận diện đa góc chỉ trung bình xác suất dự đoán của 2–6 góc; không bảo đảm tăng độ chính xác và không tạo thêm mô hình. Tư vấn tái chế là lớp quy tắc tham khảo, dựa thêm vào thông tin người dùng khai báo; không phải đầu ra của classifier và không thay thế quy định của điểm thu gom địa phương.

## Nguồn tham khảo

- TrashNet dataset: https://github.com/garythung/trashnet
- Torchvision MobileNetV3-Large weights/documentation: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html
- PyTorch installation: https://pytorch.org/get-started/locally/
- Scikit-learn: https://scikit-learn.org/stable/
- Streamlit: https://docs.streamlit.io/
- Tư vấn xử lý vật liệu trong mã V13 (hướng dẫn chung, không phải quy định Việt Nam): https://www.epa.gov/recycle/how-do-i-recycle-common-recyclables ; https://www.epa.gov/recycle/frequent-questions-recycling

## Công cụ AI và phần kế thừa

- Công cụ hỗ trợ lập trình và chuẩn bị hồ sơ: OpenAI Codex (mô hình GPT-6). Công cụ được dùng để hỗ trợ soạn README, tạo tiện ích dòng lệnh dự đoán `predict_cli_v13.py`, biên tập slide và rà soát cấu trúc gói nộp. Nhóm chịu trách nhiệm kiểm tra và giải thích toàn bộ nội dung trước khi trình bày.
- Phần kế thừa và nguồn bên ngoài: bộ dữ liệu TrashNet; trọng số pretrained ImageNet của MobileNetV3-Large do Torchvision cung cấp; các thư viện mã nguồn mở và phương pháp baseline HOG + histogram RGB + Logistic Regression theo yêu cầu AI02. Nguồn và phiên bản thư viện được liệt kê ở trên.
- Phần nhóm thực hiện: cung cấp mã nguồn sản phẩm AI02 V13; chạy các lệnh huấn luyện baseline, SVM cải tiến và CNN; lưu checkpoint, metrics, dự đoán validation và hình/bảng kết quả trong `model/`, `results/` và `results_cnn/`. Các số liệu báo cáo lấy từ những tệp kết quả này; nhóm chịu trách nhiệm đối chiếu chúng với lần chạy thực tế và trả lời câu hỏi về thiết kế thực nghiệm.
