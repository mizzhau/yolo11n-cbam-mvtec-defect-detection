# TỔNG QUAN ĐỀ TÀI & NHẬT KÝ THỰC NGHIỆM DỰ ÁN

---

## 1. Thông Tin Chung & Tên Đề Tài

- **Tên đề tài:** Ứng dụng YOLO kết hợp cơ chế Attention nâng cao trong Phát hiện khuyết tật sản phẩm công nghiệp.
- **Mô hình cốt lõi:** YOLO11n (bản nano gọn nhẹ làm baseline) kết hợp cơ chế chú ý CBAM (Convolutional Block Attention Module).
- **Bộ dữ liệu nghiên cứu:** MVTec Anomaly Detection (MVTec AD) — gồm 15 nhóm sản phẩm công nghiệp (5 nhóm bề mặt texture, 10 nhóm đối tượng object), 5.354 ảnh chất lượng cao với 48 dạng khuyết tật cụ thể.
- **Ý nghĩa thực tiễn & học thuật:** Tự động hóa kiểm tra quang học (AOI) trong dây chuyền sản xuất công nghiệp, hạn chế phụ thuộc vào kiểm tra thủ công. Đóng góp minh chứng thực nghiệm có hệ thống về hiệu quả của cơ chế chú ý đa tỷ lệ trên mô hình object detection one-stage.
- **Thành viên/Nhân sự:** (3 người)

1. Lê Thị Trà My.
2. Phan Minh Trí.
3. Phạm Nguyễn Minh Hậu.

---

## 2. Yêu Cầu Gốc và Đầu Tiên Của Đề Tài Sau Khi Nhận Được Từ Mentor.

Đề tài đặt ra các yêu cầu tiên quyết mang tính bản lề cần giải quyết ngay từ đầu:

1. Khảo sát các mô hình phát hiện đối tượng và cơ chế chú ý trong bài toán phát hiện khuyết tật sản phẩm công nghiệp.
2. Tiền xử lý bộ dữ liệu MVTec AD, chuyển mặt nạ khuyết tật thành bounding box theo định dạng YOLO.
3. Xây dựng mô hình YOLO11n làm baseline và đề xuất tích hợp cơ chế CBAM nhằm nâng cao khả năng nhận diện vùng khuyết tật.
4. Huấn luyện, đánh giá và so sánh mô hình cải tiến với YOLO11n nguyên bản bằng Precision, Recall, mAP@0.5, mAP@0.5:0.95 và F1-score.
5. Phân tích hiệu quả phát hiện, chi phí tính toán và các trường hợp dự đoán sai của mô hình.
6. Xây dựng ứng dụng web minh họa cho phép tải ảnh và hiển thị kết quả phát hiện khuyết tật.
7. Thực hiện kỹ thuật/phương pháp ablation để đánh giá cải tiến của mô hình đề xuất và phân tích mô hình đề xuất tốt hơn, mạnh hơn ở những điểm nào

---

## 3. Tóm Tắt Đề Cương & Các Task Chính Theo Thang Điểm

### 3.1. Bốn Mục Tiêu Cốt Lõi

- **Mục tiêu 1:** Xây dựng hệ thống dữ liệu hoàn chỉnh từ MVTec AD (chuyển mask sang bbox, làm sạch box nhiễu, cân bằng phân bố khuyết tật bằng kỹ thuật data augmentation).
- **Mục tiêu 2:** Đề xuất cải tiến YOLO11n bằng cách tích hợp CBAM đồng thời trên 3 mức đặc trưng đa tỷ lệ P3, P4, P5; triển khai thực nghiệm Ablation Study có hệ thống trên 4 khối kiến trúc: Backbone, Neck, Backbone + Neck và Head.
- **Mục tiêu 3:** Đánh giá định lượng toàn diện qua các chỉ số chuẩn (mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1, GFLOPs, FPS), kết hợp trực quan hóa bản đồ chú ý Grad-CAM và ma trận nhầm lẫn.
- **Mục tiêu 4:** Phát triển ứng dụng Web Demo (Streamlit) cho phép tải ảnh kiểm tra, chọn mô hình và quan sát trực quan kết quả phát hiện khuyết tật kèm độ tin cậy.

### 3.2. Khung Công Việc Chính Theo Thang Điểm Đánh Giá (Thang 10.0)

|   STT   | Nội Dung Công Việc Chính Theo Đề Cương                 |    Trọng Số     | Yêu Cầu Đầu Ra                                                                                                                     |
| :-----: | :----------------------------------------------------- | :-------------: | :--------------------------------------------------------------------------------------------------------------------------------- |
|  **1**  | **Tổng quan đề tài & Khảo sát công trình liên quan**   |   **0.75 đ**    | Phân tích các nghiên cứu trong 3–5 năm gần đây về YOLO và Attention; chỉ rõ khoảng trống nghiên cứu.                               |
|  **2**  | **Kế hoạch thực hiện & Phân công công việc**           |   **0.50 đ**    | Bảng kế hoạch chi tiết, các mốc bàn giao sản phẩm, minh chứng phân công rõ ràng.                                                   |
|  **3**  | **Phương pháp nghiên cứu & Cơ sở lý thuyết**           |   **1.00 đ**    | Trình bày nguyên lý YOLO11n, module CBAM (Channel + Spatial), giải thích lý do chèn đa tỷ lệ P3/P4/P5.                             |
|  **4**  | **Tiền xử lý dữ liệu & Nâng cao chất lượng mô hình**   |   **1.75 đ**    | Chuyển mask sang bbox, làm sạch nhãn qua CVAT, phân chia Supervised, tăng cường dữ liệu bù mất cân bằng.                           |
|  **5**  | **Thực nghiệm Ablation Study & Đánh giá mô hình**      |   **2.00 đ**    | Huấn luyện công bằng (cùng seed, epochs, optimizer) giữa Baseline và 4 cấu hình CBAM; đo đạc đầy đủ các chỉ số mAP, P, R, F1, FPS. |
|  **6**  | **Phân tích kết quả, Báo cáo hạn chế & Web Demo**      |   **2.00 đ**    | Lý giải nguyên nhân kết quả, minh chứng Grad-CAM, xây dựng ứng dụng web demo hoàn chỉnh.                                           |
| **7-9** | **Quyển báo cáo, định dạng & Tác phong, Slide bảo vệ** |   **1.50 đ**    | Báo cáo đúng quy chuẩn khoa học, slide trình bày trực quan và tác phong bảo vệ tốt.                                                |
| **10**  | **Điểm thưởng khuyến khích NCKH / Bài báo khoa học**   | **0.5 - 1.0 đ** | Có bài báo công bố hội thảo/tạp chí hoặc tham gia cuộc thi NCKH sinh viên.                                                         |

---

## 4. Nhật Ký Tiến Độ: Các Task & Thực Nghiệm Đã Hoàn Thành

Toàn bộ khung dữ liệu, kiến trúc thuật toán và các kiểm thử nền tảng đã được hoàn thiện:

### 4.1. Dữ Liệu & Tiền Xử Lý Đã Được Thử Nghiệm (Đã hoàn thành 100%)

- **Tổ chức dữ liệu:** Tải trọn vẹn 15 danh mục MVTec AD (5.354 ảnh) vào môi trường lưu trữ dự án.
- **Chuyển đổi nhãn (Mask-to-BBox):** Viết script tự động quét contour từ mask nhị phân để sinh bounding box chuẩn YOLO cho toàn bộ 48 dạng khuyết tật.
- **Kiểm định chất lượng qua CVAT:** Rà soát thủ công, sửa đổi 255 nhãn bounding box bị lệch hoặc rỗng do khuyết tật dạng sợi mảnh, chuẩn hóa 10 ảnh sai phân loại.
- **Phân chia dữ liệu có giám sát:** Triển khai phân chia tập dữ liệu chuẩn tỷ lệ 70% Train / 15% Val / 15% Test theo phương pháp phân tầng (stratified) bảo đảm đủ 48 lớp ở từng tập.
- **Tăng cường dữ liệu có mục tiêu:** Xây dựng module Offline Augmentation (xoay, lật, chỉnh HSV, nhiễu hạt) nâng số lượng mẫu các lớp hiếm lên tối thiểu 40 ảnh/lớp, tạo thành bộ dữ liệu huấn luyện cân bằng.

### 4.2. Kiến Trúc Mô Hình & Kiểm Thử Unit Test Đã Được Thử Nghiệm (Đã hoàn thành 100%)

- **Hiện thực hóa Module CBAM:** Đóng gói module PyTorch gồm Channel Attention và Spatial Attention tuần tự, bảo đảm tuyệt đối tính chất **Shape-Invariant** $(B, C, H, W)$.
- **Unit Testing & Gradient Flow:** Chạy kiểm thử tự động kiểm tra kích thước đầu ra, forward/backward pass trên cả CPU và GPU. Kết quả **100% PASS**, gradient truyền mượt mà không đứt mạch.
- **Tích hợp kiến trúc YOLO11n:** Thiết lập cấu hình YAML cho đầy đủ 5 mô hình thực nghiệm: Baseline (Exp 0), CBAM@Backbone (Exp 1), CBAM@Neck (Exp 2), CBAM@Head (Exp 4) và CBAM@Neck+Head (Exp 3).

### 4.3. Nhật Ký Các Thực Nghiệm & Testing Đã Được Thử Nghiệm Và Đã Hoàn Thành

Dự án đã tiến hành 3 đợt huấn luyện và kiểm thử thực nghiệm có hệ thống:

1. **Đợt 1 (Pilot 20 Epochs):** Đánh giá độ ổn định của pipeline trên Google Colab T4, kiểm thử lưu checkpoint định kỳ (`save_period=1`) cho Baseline và CBAM@Backbone. Kết quả xác nhận mã nguồn chạy mượt mà, không lỗi VRAM/OOM.
2. **Đợt 2 (Scale-up 100 Epochs):** Huấn luyện đối đầu giữa Baseline và CBAM@Backbone qua 100 epochs để theo dõi xu hướng hội tụ dài hạn.
3. **Đợt 3 (Full Ablation Study 100 Epochs — 5 Biến thể):** Hoàn thành huấn luyện trọn vẹn cả 5 cấu hình trên tập dữ liệu MVTec AD (split 70/15/15 augmented) và đo đạc chi tiết trên 48 classes khuyết tật:

| Cấu Hình Thực Nghiệm   | Precision  |   Recall   |  mAP@0.5   | Đánh Giá & Nhận Xét                                     |
| :--------------------- | :--------: | :--------: | :--------: | :------------------------------------------------------ |
| **YOLO11n (Baseline)** |   0.6346   |   0.6068   |   0.6198   | Chuẩn cơ sở so sánh                                     |
| **+ CBAM @ Backbone**  |   0.5279   |   0.6000   |   0.5866   | Giảm nhẹ do lọc đặc trưng quá sớm ở tầng thô            |
| **+ CBAM @ Neck**      | **0.6794** |   0.5477   |   0.6164   | Precision cao nhất, tối ưu khi kết hợp đa tỷ lệ         |
| **+ CBAM @ Head**      |   0.6572   |   0.5955   |   0.6205   | Cải thiện nhẹ trước tầng dự đoán (+0.07%)               |
| **+ CBAM @ Neck+Head** |   0.6415   | **0.6455** | **0.6701** | **Tốt nhất toàn diện (mAP tăng +5.03%, Recall +3.87%)** |

- **Đánh giá trực quan hóa & Phân tích chuyên sâu:** Đã xuất và tổng hợp toàn bộ 9 nhóm biểu đồ khoa học: Loss/Metrics curve, Confusion Matrix, PR curve, F1-confidence, và bảng đánh giá chi tiết từng lớp trong 48 classes (`per_class_evaluation_full.md`).

---
