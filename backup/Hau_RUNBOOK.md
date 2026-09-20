# RUNBOOK & KẾ HOẠCH DỰ PHÒNG — THÀNH VIÊN 1
# Thành viên: Phạm Nguyễn Minh Hậu
# Vai trò: Leader · Research · System Integration

---

## 1. Thông tin thành viên & Phạm vi công việc

- **Họ và tên:** Phạm Nguyễn Minh Hậu
- **Mã sinh viên:** 2045230028
- **Vai trò chính:** Trưởng nhóm (Project Lead), Nghiên cứu lý thuyết & SOTA (Research), Thiết kế kiến trúc tổng thể (System Design), Tổng hợp kết quả Ablation Study, Dự phòng phần cứng (Laptop GPU RTX 3040).
- **Người tiếp quản dự phòng (Backup):** Phan Minh Trí / Lê Thị Trà My.

---

## 2. Danh mục Deliverables phụ trách (Theo dõi tiến độ)

| Mã Task | Tên Deliverable | Sản phẩm bàn giao | Nơi lưu trữ | Trạng thái | Ghi chú |
|:---|:---|:---|:---|:---:|:---|
| `DELIV-HAU-01` | <Tên Deliverable 1> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |
| `DELIV-HAU-02` | <Tên Deliverable 2> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |
| `DELIV-HAU-03` | <Tên Deliverable 3> | <Mô tả sản phẩm bàn giao> | `<Đường dẫn>` | [ ] Todo | <Ghi chú nếu có> |

---

## 3. Môi trường & Lệnh thực thi (Runscripts)

> *Ghi chú: Điền lệnh dòng lệnh (CLI) thực thi trực tiếp cho từng deliverable để thành viên khác hoặc AI có thể copy-paste chạy ngay khi cần.*

### Môi trường thực thi:
- **Thiết bị:** Laptop GPU (NVIDIA RTX 3040 Laptop 6GB) / CPU đa nhân.
- **Môi trường Python:** Python 3.10+, PyTorch 2.x, Ultralytics, Streamlit, Pandas, Matplotlib.
- **Thư mục làm việc:** `d:/KLCN-2026/`

### 3.1. [DELIV-HAU-01] <Tên tác vụ>:
```bash
# <Lệnh CLI thực thi trực tiếp>
```

### 3.2. [DELIV-HAU-02] <Tên tác vụ>:
```bash
# <Lệnh CLI thực thi trực tiếp>
```

---

## 4. Tiêu chí nghiệm thu đầu ra (Verification Contract)

> *Kiểm tra tính hoàn thành của sản phẩm bàn giao (File bắt buộc tồn tại, định lượng, tính đúng đắn).*

- [ ] **DELIV-HAU-01:**
  - File bắt buộc: `<Đường dẫn file>`
  - Tiêu chí kiểm tra: `<Mô tả tiêu chí nghiệm thu>`
  - Lệnh xác thực nhanh:
    ```bash
    # <Lệnh kiểm tra hoặc sanity check nếu có>
    ```
- [ ] **DELIV-HAU-02:**
  - File bắt buộc: `<Đường dẫn file>`
  - Tiêu chí kiểm tra: `<Mô tả tiêu chí nghiệm thu>`

---

## 5. Kế hoạch dự phòng & Xử lý sự cố (Emergency Fallback)

### Kịch bản 1: <Mô tả sự cố phần cứng / môi trường local>
- **Nguyên nhân:** `<Nguyên nhân dự kiến>`
- **Giải pháp xử lý:** `<Các bước khắc phục hoặc cờ điều chỉnh tham số>`

### Kịch bản 2: Người phụ trách vắng mặt đột xuất
- **Người tiếp quản:** Phan Minh Trí / Lê Thị Trà My.
- **Quy trình tiếp quản nhanh:**
  1. Kéo mã nguồn mới nhất: `git checkout main && git pull origin main`
  2. Lấy dữ liệu / checkpoint tại: `<Đường dẫn Drive / local>`
  3. Thực thi runscript tương ứng tại **Mục 3**.

---

## 6. Quyết định kỹ thuật & Nhật ký thay đổi (Changelog)

- **Quyết định thiết kế cốt lõi:**
  - `<Ngày/Tháng/Năm>`: `<Nội dung quyết định kỹ thuật và lý do>`
- **Nhật ký cập nhật:**
  - `<Ngày/Tháng/Năm>`: `<Nội dung cập nhật tiến độ / deliverable>`
