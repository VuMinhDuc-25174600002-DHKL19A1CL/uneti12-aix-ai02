# BÁO CÁO VALIDATION AI02 - CẦN XÁC MINH NGUỒN ẢNH

- Bộ dữ liệu sau lọc: **2521 ảnh**.
- Phân chia: train **2016**, validation **505**; stratified, seed 2026.
- Baseline Macro-F1: **0.5483**; Accuracy: **0.5802**.
- Cải tiến Macro-F1: **0.7206**; Accuracy: **0.7406**.
- Tăng Macro-F1: **+0.1723** (điểm tuyệt đối; không phải % tương đối).
- Khoảng tin cậy bootstrap 95% cho hiệu số: **[+0.1236, +0.2231]**.
- Cặp dễ nhầm nhất ở baseline: **glass / metal**, tổng 30 lỗi hai chiều; sau cải tiến 21 lỗi.

## Giải thích thiết kế
- Baseline giữ HOG 64x64 + RGB histogram + Logistic Regression, theo mã BTC.
- Cải tiến thêm HSV, histogram màu theo vùng, LBP; dùng RBF SVM được hiệu chỉnh xác suất kiểu sigmoid trên train.
- Siêu tham số SVM được chọn bằng cross-validation **chỉ trên train**; validation để so sánh cuối.
- Đã rà ảnh hỏng/trùng nội dung chính xác; kiểm tra gần trùng theo ngữ nghĩa cần bổ sung nếu có điều kiện.

## Diễn giải thận trọng
- Cải tiến không được coi là vượt baseline nếu Macro-F1 không tăng trên cùng validation.
- Khoảng tin cậy chứa số 0 => bằng chứng chênh lệch chưa đủ chắc chắn.
- Xác suất trên giao diện chỉ tham khảo; TrashNet chụp trên nền đơn giản.
- Kết quả không đại diện mọi điểm thu gom ở Việt Nam và không đảm bảo khả năng tái chế.
- Kiểm tra Brier score/ECE có ý nghĩa mô tả, độ tin cậy xác suất không được bảo đảm.

## Hình trong thư mục results
- `01_class_distribution.png`: phân phối lớp.
- `02_confusion_baseline.png`, `03_confusion_improved.png`: ma trận nhầm lẫn.
- `04_per_class_f1.png`: so sánh F1 theo lớp.
- `split_manifest.csv`: nhãn và danh sách từng ảnh trong train/validation.
- `near_duplicate_audit.json`, `near_duplicate_candidates.csv`: gợi ý kiểm tra ảnh gần trùng qua train/validation bằng dHash (KHÔNG tự động loại).
- `pairwise_confusions.csv`: mọi cặp lớp và số lỗi giảm/tăng.
- `metrics.json`: thêm Brier score và ECE 10 bins (phân tích tham khảo).