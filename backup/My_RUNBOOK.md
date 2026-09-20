# RUNBOOK & KẾ HOẠCH DỰ PHÒNG — THÀNH VIÊN 3
# Thành viên: Lê Thị Trà My
# Vai trò: Model Implementation · Training Pipeline · Hyperparameter Tuning · GPU Management

---

## 1. Thông tin thành viên & Phạm vi công việc

- **Họ và tên:** Lê Thị Trà My
- **Mã sinh viên:** 2045230063
- **Vai trò chính:** Kỹ sư mô hình & Huấn luyện (Model Engineer), Hiện thực hóa module CBAM PyTorch, Tích hợp kiến trúc YOLO11 YAML, Quản trị tài nguyên GPU (Google Colab Pro), Huấn luyện các mô hình Ablation Study, Tinh chỉnh siêu tham số.
- **Người tiếp quản dự phòng (Backup):** Phạm Nguyễn Minh Hậu (Leader).

---

## 2. Danh mục Deliverables phụ trách (Theo dõi tiến độ)

| Mã Task | Tên Deliverable | Sản phẩm bàn giao | Nơi lưu trữ | Trạng thái | Ghi chú |
|:---|:---|:---|:---|:---:|:---|
| `DELIV-MY-01` | <Tên Deliverable 1> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |
| `DELIV-MY-02` | <Tên Deliverable 2> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |
| `DELIV-MY-03` | <Tên Deliverable 3> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |

---

## 3. Môi trường & Lệnh thực thi (Runscripts)

> *Ghi chú: Điền lệnh dòng lệnh (CLI) thực thi trực tiếp cho từng deliverable để thành viên khác hoặc AI có thể copy-paste chạy ngay khi cần.*

### Môi trường thực thi:
- **Thiết bị:** Google Colab Pro (GPU Tesla T4 / A100) / PyTorch CUDA.
- **Môi trường Python:** Python 3.10+, PyTorch 2.x, Torchvision, Ultralytics, TensorBoard.
- **Thư mục làm việc:** Root workspace hoặc thư mục làm việc Colab.

### 3.1. [DELIV-MY-01] <Tên tác vụ>:
```bash
# <Lệnh CLI thực thi trực tiếp>
```

### 3.2. [DELIV-MY-02] <Tên tác vụ>:
```bash
# <Lệnh CLI thực thi trực tiếp>
```

---

## 4. Tiêu chí nghiệm thu đầu ra (Verification Contract)

> *Kiểm tra tính hoàn thành của sản phẩm bàn giao (File bắt buộc tồn tại, định lượng, tính đúng đắn).*

- [ ] **DELIV-MY-01:**
  - File bắt buộc: `<Đường dẫn file trọng số / log>`
  - Tiêu chí kiểm tra: `<Mô tả tiêu chí nghiệm thu / target metric>`
  - Lệnh xác thực nhanh:
    ```bash
    # <Lệnh kiểm tra hoặc unit test nếu có>
    ```
- [ ] **DELIV-MY-02:**
  - File bắt buộc: `<Đường dẫn file>`
  - Tiêu chí kiểm tra: `<Mô tả tiêu chí nghiệm thu>`

---

## 5. Kế hoạch dự phòng & Xử lý sự cố (Emergency Fallback)

### Kịch bản 1: <Mô tả sự cố huấn luyện / ngắt kết nối GPU>
- **Nguyên nhân:** `<Nguyên nhân dự kiến (ví dụ: Colab timeout, OOM)>`
- **Giải pháp xử lý:** `<Các bước resume từ checkpoint hoặc giảm batch size>`

### Kịch bản 2: Người phụ trách vắng mặt hoặc hết hạn ngạch GPU
- **Người tiếp quản:** Phạm Nguyễn Minh Hậu.
- **Quy trình tiếp quản nhanh:**
  1. Gửi / Lấy checkpoint `last.pt` từ Google Drive.
  2. Người tiếp quản kích hoạt GPU dự phòng chạy tiếp theo runscript tại **Mục 3**.

---

## 6. Quyết định kỹ thuật & Nhật ký thay đổi (Changelog)

- **Quyết định thiết kế cốt lõi:**
  - `<Ngày/Tháng/Năm>`: `<Nội dung quyết định kiến trúc / siêu tham số và lý do>`
- **Nhật ký cập nhật:**
  - `<Ngày/Tháng/Năm>`: `<Nội dung cập nhật tiến độ / deliverable>`
