# KIẾN TRÚC HỆ THỐNG & QUY CHUẨN DỰ ÁN (ARCHITECTURE)

# Đề tài: YOLO11n + CBAM — MVTec AD Defect Detection

# Mã đề tài: KLTN_38

# Năm: 2026

---

## 1. Tổng quan dự án

- **Đề tài:** Ứng dụng YOLO kết hợp cơ chế Attention nâng cao (CBAM) trong Phát hiện khuyết tật sản phẩm công nghiệp.
- **Mô hình cốt lõi:** YOLO11n (Baseline) cải tiến tích hợp CBAM (Convolutional Block Attention Module).
- **Bộ dữ liệu:** MVTec Anomaly Detection (15 categories, 5,354 ảnh, 48 loại lỗi).
- **Mục tiêu nghiên cứu:** Tối ưu hóa khả năng phát hiện khuyết tật kích thước nhỏ/khó nhận diện, thực hiện Ablation Study vị trí chèn CBAM và triển khai hệ thống thử nghiệm.
- **GitHub Repository:** [mizzhau/yolo11n-cbam-mvtec-defect-detection](https://github.com/mizzhau/yolo11n-cbam-mvtec-defect-detection.git)

---

## 2. Cấu trúc thư mục & Quyền truy cập

> [!IMPORTANT]
> **Quy định cập nhật:** Cấu trúc thư mục dưới đây được đối chiếu trực tiếp theo cấu trúc thực tế trên GitHub repository. Cấu trúc này **phải được cập nhật liên tục tùy thuộc vào mọi thay đổi trên repo ở mỗi phiên làm việc**.

### 2.1. Cấu trúc repository

```text
yolo11n-cbam-mvtec-defect-detection/
├── .gitignore                      # Định nghĩa các file loại trừ khỏi Git
├── pyproject.toml                  # Metadata và cấu hình gói dự án
├── requirements.txt                # Danh sách thư viện phụ thuộc (PyTorch, Ultralytics, ...)
├── README.md                       # Giới thiệu đề tài và hướng dẫn cài đặt nhanh
│
├── configs/                        # Cấu hình YAML có khả năng tái lập
│   ├── data/                       #   Cấu hình dataset và chia split
│   │   ├── mvtec_60_20_20.yaml
│   │   ├── mvtec_70_15_15.yaml
│   │   ├── mvtec_70_15_15_augmented.yaml
│   │   └── mvtec_80_10_10.yaml
│   └── models/                     #   Kiến trúc mô hình Baseline và biến thể CBAM
│       ├── yolo11n_baseline.yaml
│       ├── yolo11n_cbam_backbone.yaml
│       ├── yolo11n_cbam_neck.yaml
│       ├── yolo11n_cbam_neck_head.yaml
│       └── yolo11n_cbam_head.yaml
│
├── src/                            # Mã nguồn chính của dự án
│   ├── data/                       #   Xử lý mask sang YOLO bbox, split dữ liệu, augmentation
│   ├── models/                     #   Triển khai module CBAM và tích hợp với YOLO11n
│   ├── training/                   #   Script điều phối huấn luyện (Baseline & CBAM)
│   └── evaluation/                 #   Script tính toán metrics, vẽ biểu đồ so sánh, mAP
│
├── notebooks/                      # Thử nghiệm, EDA và môi trường huấn luyện Colab
│   ├── 01_eda_dataset.ipynb
│   ├── 02_check_dataSplit.ipynb
│   ├── 03_verify_augmented_data.ipynb
│   ├── 04_deteletNcheck.ipynb
│   ├── 05_colab_testingNtraining_B.ipynb
│   └── 06_colab_testingNtraining_B-N-H.ipynb
│
├── experiments/                    # Kết quả và nhật ký thực nghiệm
│   └── testing/                    #   Các đợt chạy thử nghiệm mô hình
│       ├── 01/                     #     Đợt thử nghiệm 01 (train 20e, 100e)
│       └── 02/                     #     Đợt thử nghiệm 02 (5 biến thể mô hình)
│
├── docs/                           # Tài liệu kỹ thuật dự án
│   ├── ARCHITECTURE.md             #   [Tài liệu này] Kiến trúc và chuẩn code toàn dự án
│   └── RUNBOOK_TEMPLATE.md         #   Mẫu kế hoạch dự phòng & vận hành cho deliverable
│
├── tests/                          # Kiểm thử tự động (Unit tests cho CBAM, Data, Pipeline)
│
└── data/                           # Dữ liệu cục bộ (Được quản lý theo quy định phân quyền)
    ├── raw/                        #   Bộ dữ liệu MVTec gốc
    ├── interim/                    #   Data trong giai đoạn xử lý
    └── processed/                  #   Dữ liệu chuẩn hóa format YOLO
```

### 2.2. Bảng phân quyền thao tác thư mục

| Thư mục                        |     Quyền      | Mục đích                          | Lưu ý an toàn                           |
| :----------------------------- | :------------: | :-------------------------------- | :-------------------------------------- |
| `data/raw/`, `data/processed/` | **Read-Only**  | Dữ liệu gốc và dữ liệu chuẩn YOLO | Tuyệt đối không xóa/sửa/ghi đè          |
| `data/interim/`                | **Read/Write** | Data trong giai đoạn xử lý        | Phục vụ script chuyển đổi trung gian    |
| `artifacts/`, `*.pt`           | **Read-Only**  | Model weights và checkpoints lớn  | Không index, không commit lên Git       |
| `configs/`                     | **Read/Write** | File cấu hình hệ thống (`.yaml`)  | Sửa đổi cần review kỹ lưỡng             |
| `src/`                         | **Read/Write** | Core modules và logic ứng dụng    | Tuân thủ nghiêm PEP 8 & type hinting    |
| `notebooks/`                   | **Read/Write** | Nghiên cứu, EDA và Colab runner   | Clear output nháp trước khi commit      |
| `experiments/`                 | **Read/Write** | Output số liệu đánh giá, logs     | Không ghi đè các thí nghiệm đã hoàn tất |
| `docs/`                        | **Read/Write** | Tài liệu kiến trúc và hướng dẫn   | Cập nhật đồng bộ khi có thay đổi        |
| `tests/`                       | **Read/Write** | Unit tests cho hệ thống           | Luôn bổ sung test khi tạo module mới    |

---

## 3. Quy chuẩn đặt tên (Naming Conventions)

| Hạng mục                               | Quy tắc đặt tên                          | Ví dụ chuẩn                                          | Ví dụ không hợp lệ              |
| :------------------------------------- | :--------------------------------------- | :--------------------------------------------------- | :------------------------------ |
| **Thư mục (Directories)**              | `snake_case` hoặc `kebab-case` đồng nhất | `models/`, `evaluation/`, `configs/`                 | `MyModels/`, `eval data/`       |
| **Tệp mã nguồn Python**                | `snake_case`, phản ánh rõ chức năng      | `cbam.py`, `train_baseline.py`                       | `CBAMModel.py`, `test1.py`      |
| **Class (Lớp đối tượng)**              | `PascalCase`                             | `CBAM`, `ChannelAttention`, `SpatialAttention`       | `cbam_module`, `cbamClass`      |
| **Hàm & Biến (Functions & Variables)** | `snake_case`, tự tường minh              | `compute_iou()`, `channel_ratio`                     | `calc()`, `doSomething()`, `x1` |
| **Hằng số (Constants)**                | `UPPER_SNAKE_CASE`                       | `IMAGE_SIZE = 640`, `DEFAULT_SEED = 42`              | `image_size`, `seedVal`         |
| **Tệp cấu hình (Config YAML)**         | `snake_case`, chứa tên mô hình/tỷ lệ     | `yolo11n_cbam_neck.yaml`, `mvtec_70_15_15.yaml`      | `config.yaml`, `model1.yaml`    |
| **Jupyter Notebooks**                  | Số thứ tự 2 chữ số + mô tả mục đích      | `01_eda_dataset.ipynb`, `05_colab_training.ipynb`    | `test.ipynb`, `untitled.ipynb`  |
| **Nhánh Git (Branches)**               | `<prefix>/<mo-ta-ngan-gon>`              | `feat/cbam-head`, `fix/bbox-parser`, `exp/neck-100e` | `update`, `hau-branch`, `test`  |

---

## 4. Quy chuẩn viết code & Kỹ thuật (Coding Standards)

### 4.1. Tiêu chuẩn Python chung

- **Chuẩn format:** Tuân thủ chặt chẽ **PEP 8**, độ dài dòng không vượt quá **100 ký tự**.
- **Type Hinting:** Bắt buộc áp dụng Type Annotations cho toàn bộ tham số đầu vào và kiểu dữ liệu trả về của hàm/phương thức.
- **Docstrings:** Viết theo phong cách Google Docstrings ngắn gọn cho toàn bộ public classes và functions.
- **Thứ tự Import:**
  1. Thư viện chuẩn (Standard libraries: `os`, `sys`, `pathlib`, ...)
  2. Thư viện bên thứ ba (Third-party libraries: `torch`, `ultralytics`, `numpy`, ...)
  3. Module nội bộ dự án (Local modules: `from src.models.cbam import CBAM`)
  - Mỗi nhóm phân cách bằng đúng 1 dòng trống.
- **Loại bỏ comment dư thừa:** Không viết comment mô tả các thao tác hiển nhiên (ví dụ `# gán biến`, `# import torch`). Chỉ comment khi cần giải thích công thức toán học, logic đặc thù hoặc quyết định kiến trúc quan trọng.

### 4.2. Nguyên tắc PyTorch & YOLO

- **Xử lý thiết bị rõ ràng (Explicit Device):**
  ```python
  device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ```
- **Tính tái lập (Reproducibility):** Luôn cấu hình seed đồng nhất:

  ```python
  import torch
  import numpy as np
  import random

  def set_seed(seed: int = 42) -> None:
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      if torch.cuda.is_available():
          torch.cuda.manual_seed_all(seed)
  ```

### 4.3. Quy tắc bất biến đối với Module CBAM

- **Shape Invariant:** Module CBAM **bắt buộc** phải giữ nguyên kích thước tensor đầu ra so với đầu vào:
  $$\text{Input: } (B, C, H, W) \longrightarrow \text{Output: } (B, C, H, W)$$
- **Kiểm tra bắt buộc:** Mọi thay đổi trong `src/models/cbam.py` phải đảm bảo pass kiểm tra:
  ```python
  assert output.shape == input.shape, f"Shape mismatch: {output.shape} vs {input.shape}"
  ```

---

## 5. Quy trình Git & Nguyên tắc An toàn

### 5.1. Định dạng thông điệp Commit (Conventional Commits)

Thông điệp commit phải tuân thủ cấu trúc chuẩn: `<loại>: <nội dung tóm tắt>`

| Loại       | Ý nghĩa                                     | Ví dụ                                                  |
| :--------- | :------------------------------------------ | :----------------------------------------------------- |
| `feat`     | Tính năng mới hoặc module mới               | `feat: integrate CBAM into detection head`             |
| `fix`      | Sửa lỗi trong code hoặc cấu hình            | `fix: resolve coordinate clamping in mask conversion`  |
| `exp`      | Chạy thực nghiệm hoặc thêm file log/kết quả | `exp: add evaluation metrics for Exp 3 100e`           |
| `docs`     | Thêm hoặc cập nhật tài liệu                 | `docs: update ARCHITECTURE and experiments table`      |
| `refactor` | Tối ưu hóa code mà không thay đổi tính năng | `refactor: optimize offline augmentation memory usage` |
| `test`     | Thêm hoặc sửa đổi unit test                 | `test: add gradient flow test for CBAM module`         |

### 5.2. Nguyên tắc an toàn Git

1. **Không commit trực tiếp vào `main`:** Toàn bộ công việc thực hiện trên nhánh riêng (`feat/*`, `fix/*`, `exp/*`) và gộp qua Pull Request.
2. **Loại trừ dữ liệu lớn:** Không bao giờ commit file dữ liệu thô (`.tar.xz`, `.png` dataset lớn) hoặc trọng số mô hình nặng (`.pt`, `.pth`). File nặng cần lưu trên Cloud Storage / Google Drive và liên kết đường dẫn.
3. **Quy định đối với AI Coding Agent:** AI **tuyệt đối không được tự ý thực hiện lệnh `git push`**. Phải liệt kê đầy đủ danh sách file dự kiến thay đổi và chờ thành viên xác nhận trực tiếp trước khi thực hiện.

---

## 6. Hướng dẫn dành cho AI Coding Agent

Tài liệu này là **tài liệu quy chuẩn duy nhất** được cung cấp cho các thành viên và công cụ AI Coding Agent (Cursor, Copilot, Antigravity, v.v.) khi hỗ trợ phát triển dự án. Mọi AI Agent khi tham gia viết mã nguồn hoặc hỗ trợ kỹ thuật phải tuân thủ nghiêm ngặt:

1. **Tuân thủ cấu trúc dự án:** Viết mã đúng vị trí thư mục được quy định tại **Mục 2**. Không tạo thêm thư mục tùy tiện ở root.
2. **Tuân thủ tiêu chuẩn kỹ thuật:** Mọi đoạn mã sinh ra phải đáp ứng đầy đủ quy chuẩn tại **Mục 4** (PEP 8 $\le 100$ ký tự, Type Hinting bắt buộc, seed cố định, CBAM shape invariant).
3. **Giới hạn quyền can thiệp:**
   - Tuyệt đối không tự ý chạy các lệnh huấn luyện ngốn tài nguyên GPU (`yolo train`, `python train.py`).
   - Không tự ý xóa, sửa hoặc ghi đè dữ liệu trong `data/raw/` và `data/processed/`.
   - Không đọc hoặc sửa đổi các tệp weights nhị phân nặng trong `artifacts/`.
4. **Quy tắc an toàn Git:** Tuyệt đối không tự ý chạy lệnh `git push`. Mọi hành động liên quan đến commit/push phải được liệt kê rõ ràng danh sách tệp thay đổi và có sự đồng ý trực tiếp từ người dùng.
5. **Kiểm thử bắt buộc:** Khi tạo hoặc chỉnh sửa module mô hình trong `src/models/`, AI phải hỗ trợ viết hoặc chạy unit test kiểm tra shape tensor và gradient flow trước khi hoàn tất.

---

## 7. Bảng kiểm tra trước khi Commit / Gửi code (Pre-push Checklist)

Trước khi tạo Pull Request hoặc gửi mã nguồn lên repository, thành viên cần đối chiếu:

- [ ] Mã nguồn tuân thủ PEP 8 và đã được format gọn gàng (dòng $\le 100$ ký tự).
- [ ] Mọi hàm và phương thức đều có đầy đủ Type Annotations.
- [ ] Không chứa comment giải thích code hiển nhiên hoặc comment dư thừa.
- [ ] File weights lớn (`*.pt`) và dữ liệu thô không nằm trong Git stage (`git status` kiểm tra kỹ).
- [ ] Các notebook đã được clear output nháp không cần thiết.
- [ ] Đã chạy unit tests liên quan và đạt trạng thái **PASS**.
- [ ] Commit message đúng định dạng quy định (`feat:`, `fix:`, `docs:`, ...).
- [ ] Cập nhật lại sơ đồ cấu trúc tại **Mục 2** của file này nếu có thêm/sửa/xóa thư mục hoặc tệp quan trọng trên repo.
