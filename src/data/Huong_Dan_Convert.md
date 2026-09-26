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

## 2. Chạy convert mask → bounding box (mvtec_mask_to_yolo.py)

```bash
python src/data/mvtec_mask_to_yolo.py --mvtec_root "data/raw/mvtec_anomaly_detection" --output_root "data/interim/mvtec_yolo"
```

**Kết quả tạo ra:**

```
data/interim/mvtec_yolo/
├── images/          # Ảnh gốc copy ra (đặt tên phẳng)
├── labels/          # File .txt YOLO format tương ứng
├── classes.txt      # Danh sách class (1 dòng = 1 tên defect)
└── dataset-yolo_summary.csv   # Thống kê số ảnh/box theo category
```

> Quy tắc đặt tên file trong script này: `{category}_{defect_type}_{id}.png` (ví dụ: `bottle_broken_large_000.png`).
> Ảnh "good" (không lỗi) có file label **rỗng**, đây là negative sample.

**Tham số tùy chỉnh trong script:**

| Tham số            | Mặc định | Ghi chú                                             |
| ------------------ | -------- | ---------------------------------------------------- |
| `--min_area_ratio` | `0.0005` | Bỏ qua vùng nhỏ hơn 0.05% diện tích ảnh (lọc noise)  |

> ⚠️ **Lưu ý (đã kiểm chứng trên dataset thật):** với `min_area_ratio=0.0005`, khoảng **0.95% ảnh lỗi** (12/1258, chủ yếu ở `pill/color`) có mask thật nhưng bị lọc mất hết box do vùng lỗi quá nhỏ sau bước blur+erode. Script sẽ in cảnh báo `[WARNING] ... label will be EMPTY` cho từng ảnh này và ghi số lượng vào cột `zero_box_defects` trong CSV summary — hãy rà lại các dòng có giá trị > 0 sau khi chạy xong. Nếu muốn giữ lại các defect siêu nhỏ này, thử giảm `--min_area_ratio` (ví dụ `0.0001`) rồi so sánh lại summary.

---

## 3. (Tuỳ chọn) Chia lại theo category để upload Roboflow (split_by_category.py)

Output ở bước 2 là **1 folder phẳng** (mọi category chung `images/` + `labels/`). Nếu bạn muốn upload lên Roboflow theo từng batch category (để gắn tag category lúc upload, dễ lọc/review sau này), chạy thêm:

```bash
python src/data/split_by_category.py \
  --flat_root "data/interim/mvtec_yolo" \
  --mvtec_root "data/raw/mvtec_anomaly_detection" \
  --dest_root "data/interim/mvtec_yolo_by_category"
```

Kết quả: 15 folder con (1 folder / category), mỗi folder có `images/`, `labels/`, `classes.txt` (class ID **giữ nguyên toàn cục**, giống hệt nhau ở mọi category — không map lại riêng). File `dataset-yolo_summary.csv` cũng được copy vào thư mục gốc `--dest_root`.

> Bước này chỉ **copy** (không xoá bản phẳng ở `--flat_root`). Nếu không cần giữ bản phẳng nữa (đã copy xong, không dùng để làm gì khác), có thể xoá thủ công để đỡ tốn dung lượng — 2 bản chứa dữ liệu giống hệt nhau, chỉ khác cách tổ chức thư mục.

> Nếu không cần lọc theo category trong Roboflow (chỉ cần search theo tên file, vì tên file đã có tiền tố category sẵn), có thể bỏ qua bước này và upload thẳng bản phẳng ở bước 2.

---

## 4. Kiểm tra kết quả bằng verify script (verify_yolo_boxes.py)

Sau khi convert xong, kiểm tra trực quan xem bbox có khớp vùng lỗi không:

```bash
# Kiểm tra 1 ảnh cụ thể
python src/data/verify_yolo_boxes.py --output_root "data/interim/mvtec_yolo" --stem bottle_broken_large_000

# Kiểm tra ngẫu nhiên 1 ảnh
python src/data/verify_yolo_boxes.py --output_root "data/interim/mvtec_yolo"
```

Ảnh kết quả vẽ bbox xanh lá lưu vào thư mục `verify_output/` (mặc định). Đổi thư mục lưu bằng `--save_dir`.

---

## 5. Lỗi thường gặp & cách khắc phục

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

### ❌ `[WARNING] ... mask has defect pixels but all components were filtered out -> label will be EMPTY`

Ảnh có lỗi thật (mask không rỗng) nhưng vùng lỗi quá nhỏ, bị `min_area_ratio` lọc mất hết box. Ảnh vẫn được copy ra với label rỗng (giống ảnh "good"). Xem cột `zero_box_defects` trong CSV để biết category/defect_type nào bị ảnh hưởng, rồi quyết định: chấp nhận, xóa thủ công, hoặc giảm `--min_area_ratio`.

### ❌ Ảnh verify hiện bbox sai vị trí hoặc quá lớn/nhỏ

- Nguyên nhân phổ biến: mask bị nhiễu, có pixel trắng rải rác.
- Thử tăng `--min_area_ratio` (ví dụ `0.002`) để lọc bớt vùng nhỏ.

### ❌ `KeyError` trên defect type khi convert

Nếu thêm category mới hoặc đổi tên folder, class mapping sẽ lệch. Script tự scan toàn bộ defect folder để build mapping, nên chỉ cần đảm bảo tên folder trong `test/` và `ground_truth/` khớp nhau.

---

## 6. Sau khi convert xong hãy

1. Mở `dataset-yolo_summary.csv` để kiểm tra số lượng ảnh/box mỗi loại, đặc biệt cột `zero_box_defects`.
2. Chạy verify trên vài ảnh mỗi category để spot-check trực quan.
3. (Nếu đã chạy bước 3 - split_by_category.py) Upload từng category trong `mvtec_yolo_by_category\` lên Roboflow, gán tag theo category lúc upload.
4. Trên Roboflow: review, tự vẽ thêm bbox cho các ảnh nằm trong `zero_box_defects` (xem CSV), sửa box nào bị lệch/hụt.
5. Chỉ chia train/val/test **sau khi** đã sửa xong bbox trên Roboflow (lúc tạo Version để export) — không chia trước, vì tỉ lệ class có thể thay đổi sau khi sửa nhãn thủ công.

> ⚠️ **Không chỉnh sửa dữ liệu gốc** trong `data/raw/mvtec_anomaly_detection/`. Output convert luôn ghi vào thư mục riêng (`--output_root` bạn chỉ định, ví dụ `data/interim/mvtec_yolo/`). Bản phẳng và bản chia theo category chứa **dữ liệu giống hệt nhau** — chỉ giữ 1 trong 2 bản nếu muốn tiết kiệm dung lượng.
