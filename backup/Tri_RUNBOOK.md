# RUNBOOK & KẾ HOẠCH DỰ PHÒNG — THÀNH VIÊN 2
# Thành viên: Phan Minh Trí
# Vai trò: Data Engineer · Preprocessing · Augmentation · Experimental Validation

---

## 1. Thông tin thành viên & Phạm vi công việc

- **Họ và tên:** Phan Minh Trí
- **Mã sinh viên:** 2045230110
- **Vai trò chính:** Kỹ sư dữ liệu (Data Engineer), Tiền xử lý & Chuyển đổi nhãn Mask sang Bbox YOLO, Rà soát dữ liệu (CVAT), Tăng cường dữ liệu (Data Augmentation), Đánh giá thực nghiệm & Phân tích lỗi (Error Analysis).
- **Người tiếp quản dự phòng (Backup):** Phạm Nguyễn Minh Hậu (Leader).

---

## 2. Danh mục Deliverables phụ trách (Theo dõi tiến độ)

| Mã Task | Tên Deliverable | Sản phẩm bàn giao | Nơi lưu trữ | Trạng thái | Ghi chú |
|:---|:---|:---|:---|:---:|:---|
| `DELIV-TRI-01` | <Tên Deliverable 1> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |
| `DELIV-TRI-02` | <Tên Deliverable 2> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |
| `DELIV-TRI-03` | <Tên Deliverable 3> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |

---

## 3. Môi trường & Lệnh thực thi (Runscripts)

> *Ghi chú: Điền lệnh dòng lệnh (CLI) thực thi trực tiếp cho từng deliverable để thành viên khác hoặc AI có thể copy-paste chạy ngay khi cần.*

### Môi trường thực thi:
- **Thiết bị:** Máy tính cá nhân / Google Colab (CPU/GPU).
- **Môi trường Python:** Python 3.10+, OpenCV, NumPy, Albumentations, Shapely, PyYAML.
- **Thư mục làm việc:** `d:/KLCN-2026/`

### 3.1. [DELIV-TRI-01] <Tên tác vụ>:
```bash
# <Lệnh CLI thực thi trực tiếp>
```

### 3.2. [DELIV-TRI-02] <Tên tác vụ>:
```bash
# <Lệnh CLI thực thi trực tiếp>
```

---

## 4. Tiêu chí nghiệm thu đầu ra (Verification Contract)

> *Kiểm tra tính hoàn thành của sản phẩm bàn giao (File bắt buộc tồn tại, định lượng, tính đúng đắn).*

- [ ] **DELIV-TRI-01:**
  - File bắt buộc: `<Đường dẫn file>`
  - Tiêu chí kiểm tra: `<Mô tả tiêu chí nghiệm thu>`
  - Lệnh xác thực nhanh:
    ```bash
    # <Lệnh kiểm tra hoặc sanity check nếu có>
    ```
- [ ] **DELIV-TRI-02:**
  - File bắt buộc: `<Đường dẫn file>`
  - Tiêu chí kiểm tra: `<Mô tả tiêu chí nghiệm thu>`

---

## 5. Kế hoạch dự phòng & Xử lý sự cố (Emergency Fallback)

### Kịch bản 1: <Mô tả sự cố dữ liệu / format nhãn>
- **Nguyên nhân:** `<Nguyên nhân dự kiến>`
- **Giải pháp xử lý:** `<Các bước khắc phục hoặc script xử lý>`

### Kịch bản 2: Người phụ trách vắng mặt đột xuất
- **Người tiếp quản:** Phạm Nguyễn Minh Hậu.
- **Quy trình tiếp quản nhanh:**
  1. Kéo mã nguồn mới nhất: `git checkout main && git pull origin main`
  2. Lấy dữ liệu đã backup tại: `<Đường dẫn Drive / local>`
  3. Thực thi runscript tương ứng tại **Mục 3**.

---

## 6. Quyết định kỹ thuật & Nhật ký thay đổi (Changelog)

- **Quyết định thiết kế cốt lõi:**
  - `<Ngày/Tháng/Năm>`: `<Nội dung quyết định kỹ thuật và lý do>`
- **Nhật ký cập nhật:**
  - `<Ngày/Tháng/Năm>`: `<Nội dung cập nhật tiến độ / deliverable>`
