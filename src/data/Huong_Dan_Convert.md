# Hướng dẫn chạy Convert Mask → YOLO BBox

Script: `src/data/mvtec_mask_to_yolo.py` + `src/data/verify_yolo_boxes.py`

---

## 1. Yêu cầu trước khi chạy

- Python ≥ 3.9
- Cài dependencies:
  ```bash
  pip install opencv-python numpy
  ```
- Dataset MVTec AD đã giải nén đầy đủ 15 category tại:
  ```
  data/raw/mvtec_anomaly_detection/
  ```

---

## 2. Chạy convert mask → bounding box(verify_yolo_boxes.py)

```bash
python src/data/mvtec_mask_to_yolo.py \
  --mvtec_root data/raw/mvtec_anomaly_detection \
  --output_root data/interim/mvtec_yolo
```

**Kết quả tạo ra:**

```
data/interim/mvtec_yolo/
├── images/          # Ảnh gốc copy ra (đặt tên phẳng)
├── labels/          # File .txt YOLO format tương ứng
├── classes.txt      # Danh sách class (1 dòng = 1 tên defect)
└── dataset-yolo_summary.csv   # Thống kê số ảnh/box theo category
```

> Quy tắc đặt tên file trong scipt này: `{category}_{defect_type}_{id}.png` (ví dụ: `bottle_broken_large_000.png`).
> Ảnh "good" (không lỗi) có file label **rỗng**, đây là negative sample.

**Tham số tùy chỉnh trong script:**


| Tham số           | Mặc định | Ghi chú                                                    |
| ------------------ | ----------- | ----------------------------------------------------------- |
| `--min_area_ratio` | `0.0005`    | Bỏ qua vùng nhỏ hơn 0.05% diện tích ảnh (lọc noise) |

---

## 3. Kiểm tra kết quả bằng verify script(verify_yolo_boxes.py)

Sau khi convert xong, kiểm tra trực quan xem bbox có khớp vùng lỗi không:

```bash
# Kiểm tra 1 ảnh cụ thể
python src/data/verify_yolo_boxes.py \
  --output_root data/interim/mvtec_yolo \
  --stem bottle_broken_large_000

# Kiểm tra ngẫu nhiên 1 ảnh
python src/data/verify_yolo_boxes.py \
  --output_root data/interim/mvtec_yolo
```

Ảnh kết quả vẽ bbox xanh lá lưu vào thư mục `verify_output/` (mặc định). Đổi thư mục lưu bằng `--save_dir`.

---

## 4. Lỗi thường gặp & cách khắc phục

### ❌ `ModuleNotFoundError: No module named 'cv2'`

```bash
pip install opencv-python
```

### ❌ `FileNotFoundError` hoặc `Found 0 categories`

Kiểm tra `--mvtec_root` trỏ đúng đến thư mục chứa 15 folder category (`bottle/`, `cable/`, ...), không phải thư mục cha hay thư mục con.

### ❌ `[warn] missing mask for ...`

Ảnh test có nhưng thiếu file mask tương ứng trong `ground_truth/`. Kiểm tra xem dataset giải nén có bị thiếu file không. Quy ước mask phải đặt tên `{id}_mask.png`.

### ❌ `[skip] category: missing test/ or ground_truth/ folder`

Category chưa giải nén hoặc cấu trúc thư mục không đúng chuẩn MVTec. Cần có đủ 3 thư mục: `train/`, `test/`, `ground_truth/`.

### ❌ Ảnh verify hiện bbox sai vị trí hoặc quá lớn/nhỏ

- Nguyên nhân phổ biến: mask bị nhiễu, có pixel trắng rải rác.
- Thử tăng `--min_area_ratio` (ví dụ `0.002`) để lọc bớt vùng nhỏ.

### ❌ `KeyError` trên defect type khi convert

Nếu thêm category mới hoặc đổi tên folder, class mapping sẽ lệch. Script tự scan toàn bộ defect folder để build mapping, nên chỉ cần đảm bảo tên folder trong `test/` và `ground_truth/` khớp nhau.

---

## 5. Sau khi convert xong hãy

1. Mở `dataset-yolo_summary.csv` để kiểm tra số lượng ảnh/box mỗi loại.
2. Chạy verify trên vài ảnh mỗi category để spot-check trực quan.

> ⚠️ **Không chỉnh sửa dữ liệu gốc** trong `data/raw/`. Output luôn ghi vào thư mục riêng.
