# KHUNG KẾ HOẠCH DỰ PHÒNG & VẬN HÀNH (RUNBOOK & BACKUP PLAN TEMPLATE)
# Đề tài: YOLO11n + CBAM — MVTec AD Defect Detection (KLCN-2026)

> **Mục đích:** Tài liệu này là khung sườn chuẩn hóa dành cho mỗi Deliverable (sản phẩm bàn giao) trong dự án. 
> - **Đối với thành viên:** Điền trong 3–5 phút. Khi có rủi ro (thành viên bận, mất mạng, lỗi máy, hết quota Colab), thành viên dự phòng có thể mở ra tiếp quản và chạy ngay trong 1 phút.
> - **Đối với AI Coding Agent:** Đọc hiểu cấu trúc này để nắm rõ Input/Output, chạy chính xác lệnh thực thi và kiểm chứng kết quả tự động.

---

## PHẦN 1: KHUNG MẪU RUNBOOK (DÙNG ĐỂ COPY & ĐIỀN)

```markdown
# RUNBOOK: [MÃ-TASK] - [TÊN DELIVERABLE]

## 1. Định danh Deliverable & Chuỗi phụ thuộc
- **Tên Deliverable:** <Ví dụ: Data Pipeline chuyển đổi MVTec mask sang YOLO bbox>
- **Người phụ trách chính (Owner):** <Tên thành viên>
- **Người dự phòng (Backup):** <Tên thành viên tiếp quản khi có sự cố>
- **Trạng thái:** [ ] Todo  |  [ ] In Progress  |  [ ] Done
- **Đầu vào cần nhận (Input):**
  - Cần: `<Tên file / thư mục input>`
  - Từ ai: `<Thành viên bàn giao>`
  - Vị trí: `<Đường dẫn local hoặc link Google Drive>`
- **Đầu ra bàn giao (Output):**
  - Sản phẩm: `<Tên file / thư mục output>`
  - Bàn giao cho: `<Thành viên sử dụng tiếp theo>`
  - Vị trí lưu: `<Đường dẫn local hoặc Drive>`

## 2. Môi trường & Lệnh thực thi 1 chạm (Runscript)
- **Môi trường yêu cầu:** <Colab Pro (GPU T4/A100) / Laptop GPU (RTX 3040) / CPU>
- **Thư mục làm việc (Working Directory):** `<KLCN-2026/>`
- **Các biến môi trường / Cấu hình phụ:** `<Nếu có, ví dụ: export CUDA_VISIBLE_DEVICES=0>`
- **Lệnh thực thi trực tiếp (Copy-Paste chạy ngay):**
  ```bash
  <Lệnh dòng lệnh CLI hoàn chỉnh, có cờ tham số cụ thể>
  ```

## 3. Tiêu chí nghiệm thu đầu ra (Verification Contract)
Làm sao để biết lệnh chạy thành công 100%?
- [ ] File output bắt buộc tồn tại: `<Đường dẫn file>`
- [ ] Định lượng tối thiểu: `<Ví dụ: Kích thước file > 5MB, đủ 5354 nhãn .txt, v.v.>`
- [ ] Lệnh kiểm tra nhanh (Sanity Check nếu có):
  ```bash
  <Lệnh kiểm tra hoặc assert dữ liệu>
  ```

## 4. Kế hoạch dự phòng & Xử lý sự cố (Emergency Fallback)
- **Sự cố 1: Tràn bộ nhớ GPU (CUDA Out Of Memory):**
  - Giải pháp: Hạ `--batch-size` từ `<16>` xuống `<8>`, thêm tham số `--amp`.
- **Sự cố 2: Hết quota / Mất kết nối Colab:**
  - Checkpoint gần nhất lưu tại: `<Link Google Drive thư mục checkpoints>`
  - Lệnh resume tiếp tục huấn luyện ngay lập tức:
    ```bash
    python src/training/train_cbam.py --resume <path_to_last.pt>
    ```
- **Sự cố 3: Người phụ trách vắng mặt đột xuất:**
  - Người dự phòng làm theo 3 bước:
    1. Kéo code mới nhất từ nhánh: `git checkout <ten-nhanh> && git pull`
    2. Tải dữ liệu/checkpoint tại: `<Link Drive>`
    3. Chạy chính xác lệnh ở **Mục 2** trên máy của mình.

## 5. Quyết định kỹ thuật & Nhật ký thay đổi (Changelog)
- **Quyết định thiết kế cốt lõi:**
  - `<Lý do chọn thông số hoặc thuật toán này thay vì phương án khác>`
- **Nhật ký thay đổi:**
  - `<Ngày/Tháng/Năm>`: `<Nội dung điều chỉnh và lý do>`
```

---

## PHẦN 2: CÁC VÍ DỤ MẪU THỰC TẾ TRONG DỰ ÁN

Dưới đây là 3 ví dụ thực tế đã được điền sẵn theo phân vai công việc của nhóm 3 thành viên:

### Ví dụ 1: Vai trò Người 2 (Data Engineer & Experiment Validation)

```markdown
# RUNBOOK: DELIV-DATA-01 - Chuẩn hóa nhãn MVTec sang format YOLO

## 1. Định danh Deliverable & Chuỗi phụ thuộc
- **Tên Deliverable:** Chuyển đổi toàn bộ mask segmentation sang Bounding Box YOLO (15 categories)
- **Người phụ trách chính (Owner):** Người 2 (Data Engineer)
- **Người dự phòng (Backup):** Người 1 (Leader)
- **Trạng thái:** [x] Done
- **Đầu vào cần nhận (Input):**
  - Cần: Dataset MVTec gốc 15 thư mục `.tar.xz` giải nén
  - Từ ai: Tải từ website MVTec
  - Vị trí: `data/raw/`
- **Đầu ra bàn giao (Output):**
  - Sản phẩm: File nhãn `.txt` chuẩn format YOLO (x_center, y_center, w, h)
  - Bàn giao cho: Người 1 và Người 3 dùng để train
  - Vị trí lưu: `data/processed/labels/` và Drive `KLCN-2026/dataset/labels/`

## 2. Môi trường & Lệnh thực thi 1 chạm (Runscript)
- **Môi trường yêu cầu:** Local CPU hoặc Google Colab (Python 3.10, OpenCV, Shapely)
- **Thư mục làm việc:** `d:/KLCN-2026/`
- **Lệnh thực thi:**
  ```bash
  python src/data/mvtec_mask_to_yolo.py --raw-dir data/raw --output-dir data/processed
  ```

## 3. Tiêu chí nghiệm thu đầu ra (Verification Contract)
- [x] File output bắt buộc: Đủ 15 thư mục con tương ứng 15 categories.
- [x] Định lượng: Tổng cộng đủ 5,354 tệp `.txt` (ảnh normal tạo file rỗng, ảnh lỗi có tọa độ $\in [0, 1]$).
- [x] Lệnh kiểm tra nhanh:
  ```bash
  python src/data/verify_yolo_boxes.py --labels-dir data/processed/labels
  # Kết quả mong đợi: In ra "ALL 5354 SAMPLES VALID - 0 ERRORS"
  ```

## 4. Kế hoạch dự phòng & Xử lý sự cố (Emergency Fallback)
- **Sự cố 1: Tọa độ bbox bị tràn ra ngoài biên (x > 1.0 hoặc y < 0):**
  - Chạy hàm kẹp biên `np.clip(val, 0.0, 1.0)` đã tích hợp sẵn trong script.
- **Sự cố 2: Người 2 bận hoặc máy lỗi:**
  - Người 1 chỉ cần vào Google Drive: `KLCN-2026/backup_labels/` tải file `mvtec_yolo_labels_backup.zip` về giải nén vào `data/processed/labels/` mà không cần chạy lại script.

## 5. Quyết định kỹ thuật & Changelog
- **Quyết định:** Sử dụng bounding box bao quanh vùng mask ngoài cùng (min_x, min_y, max_x, max_y) thay vì tách đa bbox khi lỗi bị phân mảnh, đảm bảo tương thích 100% với YOLO11.
- **Changelog:**
  - 10/09/2026: Sửa lỗi 255 nhãn bị lệch tọa độ qua kết quả review CVAT.
```

---

### Ví dụ 2: Vai trò Người 3 (Model & Engineering)

```markdown
# RUNBOOK: DELIV-MODEL-02 - Huấn luyện YOLO11n + CBAM Neck 100 Epochs

## 1. Định danh Deliverable & Chuỗi phụ thuộc
- **Tên Deliverable:** Trọng số mô hình YOLO11n tích hợp CBAM tại Neck (Exp 2)
- **Người phụ trách chính (Owner):** Người 3 (Model Engineer)
- **Người dự phòng (Backup):** Người 1 (Leader)
- **Trạng thái:** [x] Done
- **Đầu vào cần nhận (Input):**
  - Cần: Cấu hình `configs/models/yolo11n_cbam_neck.yaml` và dataset đã split 70/15/15
  - Từ ai: Người 2 bàn giao
  - Vị trí: `data/processed/` và `configs/data/mvtec_70_15_15.yaml`
- **Đầu ra bàn giao (Output):**
  - Sản phẩm: File trọng số `best.pt`, `last.pt` và file chỉ số `results.csv`
  - Bàn giao cho: Toàn bộ nhóm viết báo cáo và Người 1 đánh giá Ablation
  - Vị trí lưu: `experiments/testing/02/cbam_neck/` và Google Drive

## 2. Môi trường & Lệnh thực thi 1 chạm (Runscript)
- **Môi trường yêu cầu:** Colab Pro (GPU T4 / A100) hoặc Laptop RTX 3040 (6GB VRAM)
- **Thư mục làm việc:** Root thư mục dự án
- **Lệnh thực thi:**
  ```bash
  python src/training/train_cbam.py --model configs/models/yolo11n_cbam_neck.yaml --data configs/data/mvtec_70_15_15.yaml --epochs 100 --batch 16 --imgsz 640 --device 0 --name exp2_cbam_neck
  ```

## 3. Tiêu chí nghiệm thu đầu ra (Verification Contract)
- [x] Tồn tại file `experiments/testing/02/cbam_neck/weights/best.pt` (kích thước $\approx 6.0 - 6.5\text{ MB}$).
- [x] File `results.csv` có đủ 100 dòng số liệu không bị gián đoạn.
- [x] Metric mAP@0.5 đạt $\ge 64.0\%$ trên tập validation.

## 4. Kế hoạch dự phòng & Xử lý sự cố (Emergency Fallback)
- **Sự cố 1: Colab ngắt kết nối giữa chừng (ví dụ epoch 68/100):**
  - Checkpoint tự động lưu mỗi epoch lên Google Drive: `/MyDrive/KLCN-2026/runs/exp2_cbam_neck/weights/last.pt`
  - Lệnh khôi phục ngay:
    ```bash
    python src/training/train_cbam.py --resume /content/drive/MyDrive/KLCN-2026/runs/exp2_cbam_neck/weights/last.pt
    ```
- **Sự cố 2: Tràn VRAM khi train trên laptop RTX 3040:**
  - Đổi tham số `--batch 16` thành `--batch 8` và thêm `--workers 2`.
- **Sự cố 3: Người 3 hết lượt Colab Pro:**
  - Bàn giao link Colab notebook cho Người 1 tiếp tục huấn luyện trên tài khoản phụ hoặc trên máy local.

## 5. Quyết định kỹ thuật & Changelog
- **Quyết định:** Chèn CBAM ngay sau các tầng C3k2 ở Neck trước khi thực hiện phép nối Concat để lọc bớt nhiễu nền công nghiệp.
- **Changelog:**
  - 15/09/2026: Đặt cố định `torch.manual_seed(42)` để đảm bảo kết quả tái lập tuyệt đối.
```

---

### Ví dụ 3: Vai trò Người 1 (Leader + System & Research)

```markdown
# RUNBOOK: DELIV-LEAD-03 - Tổng hợp bảng so sánh Ablation Study 5 Mô hình

## 1. Định danh Deliverable & Chuỗi phụ thuộc
- **Tên Deliverable:** Bảng tổng hợp số liệu Ablation Study và biểu đồ trực quan mAP / Recall
- **Người phụ trách chính (Owner):** Người 1 (Leader / System)
- **Người dự phòng (Backup):** Người 2
- **Trạng thái:** [ ] In Progress
- **Đầu vào cần nhận (Input):**
  - Cần: 5 file `results.csv` của 5 mô hình (Baseline, Backbone, Neck, Neck+Head, Head)
  - Từ ai: Người 3 bàn giao sau khi train xong 100 epochs
  - Vị trí: `experiments/testing/02/`
- **Đầu ra bàn giao (Output):**
  - Sản phẩm: Bảng Markdown cập nhật tại `docs/EXPERIMENTS.md` và biểu đồ so sánh `ablation_comparison.png`
  - Bàn giao cho: Đưa vào Chương 4 Báo cáo Khóa luận tốt nghiệp
  - Vị trí lưu: `docs/EXPERIMENTS.md` và `docs/thesis/figures/`

## 2. Môi trường & Lệnh thực thi 1 chạm (Runscript)
- **Môi trường yêu cầu:** Local CPU (Python 3.10, Matplotlib, Pandas, Seaborn)
- **Thư mục làm việc:** `d:/KLCN-2026/`
- **Lệnh thực thi:**
  ```bash
  python src/evaluation/plot_comparison.py --exp-dir experiments/testing/02 --output docs/thesis/figures/
  ```

## 3. Tiêu chí nghiệm thu đầu ra (Verification Contract)
- [ ] File ảnh `ablation_comparison.png` hiển thị rõ 5 đường học mAP@0.5 và Recall.
- [ ] Bảng Markdown trong `docs/EXPERIMENTS.md` có đầy đủ các cột: Model, Params (M), GFLOPs, Precision, Recall, mAP50, mAP50-95.
- [ ] Độ chênh lệch $\Delta \text{mAP}$ so với Baseline được tính toán chính xác.

## 4. Kế hoạch dự phòng & Xử lý sự cố (Emergency Fallback)
- **Sự cố 1: Thiếu 1 trong 5 file kết quả do có mô hình đang chạy lại:**
  - Sử dụng cờ `--skip-missing` để xuất báo cáo tạm thời cho các mô hình đã hoàn tất.
- **Sự cố 2: Người 1 bận phản biện đề cương:**
  - Người 2 lấy script trên chạy trực tiếp trên máy cá nhân và copy nội dung bảng markdown được sinh tự động vào `docs/EXPERIMENTS.md`.

## 5. Quyết định kỹ thuật & Changelog
- **Quyết định:** Sử dụng mAP@0.5 làm metric chính để xếp hạng các biến thể CBAM, mAP@0.5:0.95 và Recall làm metric phụ trợ.
- **Changelog:**
  - 18/09/2026: Thêm metric FPS và số lượng tham số Params (M) vào bảng so sánh để đánh giá chi phí tính toán.
```
