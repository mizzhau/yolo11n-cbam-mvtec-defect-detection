# Augmentation: kỹ thuật và tham số

Gồm 2 phần: **(1)** augmentation OFFLINE theo từng category (đã sinh sẵn 319 ảnh, chỉ từ ảnh có khuyết tật trong tập train); **(2)** augmentation ONLINE mặc định của YOLO (bật/tắt, tham số).

Nguồn: `configs/aug_policy.yaml`, `configs/train_main.yaml`, `default.yaml` của Ultralytics 8.3.253.

---

## PHẦN 1. OFFLINE theo category

Mỗi ảnh augment dùng tối đa 1 phép hình học (ngoài lật) và tối đa 2 phép quang học, chọn trong các phép bên dưới. Không dùng grayscale và hue (sẽ làm sai ý nghĩa các lỗi về màu).

### bottle

- Lật ngang (flip ngang): bật
- Lật dọc (flip dọc): bật
- Xoay 90°: 90°, 180° hoặc 270° (ngẫu nhiên)
- Xoay góc bất kỳ: từ 0° đến 360°
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Cân bằng trắng (white balance): mỗi kênh màu nhân 1 ± 3%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,7

### cable

- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,7
- *Không dùng:* Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ, Cân bằng trắng, Độ bão hòa
- *Riêng một số loại lỗi bị cấm thêm:* `cable_swap` (Lật ngang, Lật dọc, Độ bão hòa, Tịnh tiến, Cân bằng trắng, phóng to); `combined` (Lật ngang, Lật dọc, Độ bão hòa, Tịnh tiến, Cân bằng trắng, phóng to); `missing_cable` (Lật ngang, Lật dọc, Độ bão hòa, Tịnh tiến, Cân bằng trắng, phóng to); `missing_wire` (Lật ngang, Lật dọc, Độ bão hòa, Tịnh tiến, Cân bằng trắng, phóng to)

### capsule

- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Cân bằng trắng (white balance): mỗi kênh màu nhân 1 ± 3%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,5
- *Không dùng:* Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ
- *Riêng một số loại lỗi bị cấm thêm:* `faulty_imprint` (Làm mờ Gaussian, Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ)

### carpet

- Lật ngang (flip ngang): bật
- Lật dọc (flip dọc): bật
- Xoay 90°: 90°, 180° hoặc 270° (ngẫu nhiên)
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,7
- *Không dùng:* Xoay góc bất kỳ, Cân bằng trắng
- *Riêng một số loại lỗi bị cấm thêm:* `color` (Chiếu sáng không đều, Độ bão hòa, Cân bằng trắng)

### grid

- Lật ngang (flip ngang): bật
- Lật dọc (flip dọc): bật
- Xoay 90°: 90°, 180° hoặc 270° (ngẫu nhiên)
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,5
- *Không dùng:* Xoay góc bất kỳ, Cân bằng trắng, Độ bão hòa
- *Riêng một số loại lỗi bị cấm thêm:* `bent` (Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ, Phóng to/thu nhỏ, Tịnh tiến)

### hazelnut

- Lật ngang (flip ngang): bật
- Lật dọc (flip dọc): bật
- Xoay 90°: 90°, 180° hoặc 270° (ngẫu nhiên)
- Xoay góc bất kỳ: từ 0° đến 360°
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Tương phản (contrast): từ −5,6% đến +5,6%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Cân bằng trắng (white balance): mỗi kênh màu nhân 1 ± 3%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,7
- *Không dùng:* Độ sáng

### leather

- Lật ngang (flip ngang): bật
- Lật dọc (flip dọc): bật
- Xoay 90°: 90°, 180° hoặc 270° (ngẫu nhiên)
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,7
- *Không dùng:* Xoay góc bất kỳ, Cân bằng trắng
- *Riêng một số loại lỗi bị cấm thêm:* `color` (Chiếu sáng không đều, Độ bão hòa, Cân bằng trắng)

### metal_nut

- Xoay 90°: 90°, 180° hoặc 270° (ngẫu nhiên)
- Xoay góc bất kỳ: từ 0° đến 360°
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,7
- *Không dùng:* Lật ngang, Lật dọc, Cân bằng trắng
- *Riêng một số loại lỗi bị cấm thêm:* `color` (Chiếu sáng không đều, Độ bão hòa, Cân bằng trắng); `flip` (Lật ngang, Lật dọc)

### pill

- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −5% đến +5%
- Tương phản (contrast): từ −5% đến +5%
- Gamma: γ từ 0,95 đến 1,05
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Nhiễu Gaussian (noise): σ từ 1 đến 2,5
- *Không dùng:* Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ, Cân bằng trắng, Độ bão hòa, Làm mờ Gaussian
- *Riêng một số loại lỗi bị cấm thêm:* `color` (Chiếu sáng không đều, Độ bão hòa, Cân bằng trắng); `combined` (Làm mờ Gaussian, Lật ngang, Lật dọc, Chiếu sáng không đều, Xoay 90°, Xoay góc bất kỳ, Độ bão hòa, Cân bằng trắng); `faulty_imprint` (Làm mờ Gaussian, Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ); `pill_type` (Chiếu sáng không đều, Độ bão hòa, Cân bằng trắng)

### screw

- Xoay 90°: 90°, 180° hoặc 270° (ngẫu nhiên)
- Xoay góc bất kỳ: từ 0° đến 360°
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,5
- *Không dùng:* Lật ngang, Lật dọc, Cân bằng trắng, Độ bão hòa

### tile

- Lật ngang (flip ngang): bật
- Lật dọc (flip dọc): bật
- Xoay 90°: 90°, 180° hoặc 270° (ngẫu nhiên)
- Xoay nhỏ: từ −5° đến +5°
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,7
- *Không dùng:* Xoay góc bất kỳ, Tịnh tiến, Phóng to/thu nhỏ, Cân bằng trắng
- *Riêng một số loại lỗi bị cấm thêm:* `glue_strip` (Độ bão hòa); `gray_stroke` (Chiếu sáng không đều, Độ bão hòa); `oil` (Chiếu sáng không đều, Độ bão hòa)

### toothbrush (không có ảnh augment vì không cần cân bằng thêm)

- Lật ngang (flip ngang): bật
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,9 đến 1,1
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Cân bằng trắng (white balance): mỗi kênh màu nhân 1 ± 3%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- *Không dùng:* Lật dọc, Xoay 90°, Xoay góc bất kỳ, Làm mờ Gaussian

### transistor

- Lật ngang (flip ngang): bật
- Xoay nhỏ: từ −2° đến +2°
- Tịnh tiến (translate): từ −2,8% đến +2,8% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,94 đến 1,06
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Cân bằng trắng (white balance): mỗi kênh màu nhân 1 ± 3%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,5
- *Không dùng:* Lật dọc, Xoay 90°, Xoay góc bất kỳ
- *Riêng một số loại lỗi bị cấm thêm:* `bent_lead` (Làm mờ Gaussian); `cut_lead` (Làm mờ Gaussian); `misplaced` (Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ, Phóng to/thu nhỏ, Tịnh tiến)

### wood

- Lật ngang (flip ngang): bật
- Lật dọc (flip dọc): bật
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Phóng to/thu nhỏ (scale): từ 0,96 đến 1,04
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 3% đến 10%
- Độ bão hòa (saturation): từ −10% đến +10%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,7
- *Không dùng:* Xoay 90°, Xoay góc bất kỳ, Cân bằng trắng
- *Riêng một số loại lỗi bị cấm thêm:* `color` (Chiếu sáng không đều, Độ bão hòa, Cân bằng trắng); `combined` (Chiếu sáng không đều, Độ bão hòa, Cân bằng trắng); `liquid` (Chiếu sáng không đều)

### zipper

- Lật ngang (flip ngang): bật
- Lật dọc (flip dọc): bật
- Xoay nhỏ: từ −5° đến +5°
- Tịnh tiến (translate): từ −5% đến +5% kích thước ảnh
- Độ sáng (brightness): từ −10% đến +10%
- Tương phản (contrast): từ −10% đến +10%
- Gamma: γ từ 0,9 đến 1,1
- Chiếu sáng không đều (gradient/vignette): biên độ từ 1,7% đến 5,6%
- Nhiễu Gaussian (noise): σ từ 1 đến 4
- Làm mờ Gaussian (blur): σ từ 0,3 đến 0,5
- *Không dùng:* Xoay 90°, Xoay góc bất kỳ, Phóng to/thu nhỏ, Cân bằng trắng, Độ bão hòa
- *Riêng một số loại lỗi bị cấm thêm:* `combined` (Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ, Phóng to/thu nhỏ, Tịnh tiến); `split_teeth` (Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ, Phóng to/thu nhỏ, Tịnh tiến); `squeezed_teeth` (Lật ngang, Lật dọc, Xoay 90°, Xoay góc bất kỳ, Phóng to/thu nhỏ, Tịnh tiến)

---

## PHẦN 2. ONLINE (mặc định của YOLO lúc train)

Cấu hình online dùng chung cho mọi run. Có **4 phép được bật** (`hsv_v`, `translate`, `scale`, `mosaic`) kèm tham số `close_mosaic`, còn lại tắt.

### Bật

- `hsv_v` (độ sáng): mặc định 0.4 → dùng **0.15**
- `translate` (tịnh tiến): mặc định 0.1 → dùng **0.05**
- `scale` (phóng to/thu nhỏ): mặc định 0.5 → dùng **0.1** (hệ số 0.9 đến 1.1)
- `mosaic` (ghép 4 ảnh thành 1): mặc định 1.0 → dùng **0.5** (xác suất 50% mỗi mẫu train là ảnh ghép, 50% còn lại là ảnh đơn nguyên bản)
- `close_mosaic`: mặc định 10 → dùng **10** (với `epochs: 100`: mosaic chạy từ epoch 1 đến 90, tắt hẳn ở 10 epoch cuối, từ 91 đến 100)

### Tắt

- `hsv_h` (hue): mặc định 0.015 → **0.0**
- `hsv_s` (saturation): mặc định 0.7 → **0.0**
- `bgr` (đảo kênh màu): mặc định 0.0 → **0.0**
- `degrees` (xoay): mặc định 0.0 → **0.0**
- `shear`: mặc định 0.0 → **0.0**
- `perspective`: mặc định 0.0 → **0.0**
- `flipud` (lật dọc): mặc định 0.0 → **0.0**
- `fliplr` (lật ngang): mặc định 0.5 → **0.0**
- `mixup`: mặc định 0.0 → **0.0**
- `cutmix`: mặc định 0.0 → **0.0**
- `copy_paste`: mặc định 0.0 → **0.0**
- `erasing`: mặc định 0.4 → **0.0** (chỉ có tác dụng với classification)
- `auto_augment` (randaugment): chỉ có tác dụng với classification, không ảnh hưởng detection
- Albumentations ngầm (Blur, MedianBlur, ToGray, CLAHE, mỗi phép p = 0.01): bật mặc định → **tắt** bằng `augmentations=[]`

Lý do tắt: các phép trên làm đổi ý nghĩa lỗi (màu, vị trí, hướng) hoặc làm lỗi nhỏ biến mất; phần an toàn theo từng category đã làm ở offline.

### Mosaic và close_mosaic: vì sao chọn 0.5 và 10

**Mosaic là gì và vì sao bật.** `mosaic` là xác suất áp dụng cho mỗi mẫu train: ghép ngẫu nhiên 4 ảnh, rồi cắt vùng `imgsz`×`imgsz` ở giữa. Dữ liệu MVTec có vật luôn nằm giữa ảnh; mosaic đưa vật và lỗi tới nhiều vị trí khác nhau, thêm nhiều nền, giúp model bớt phụ thuộc vào vị trí vật và bớt overfit trên tập nhỏ (khoảng 1000 đến 1300 ảnh lỗi, 48 class). Mosaic không lật, không đổi màu, nên không vi phạm các quy tắc cấm lật và cấm đổi màu ở trên.

**Vì sao chọn 0.5, không chọn 1.0 hay 0.3.**
- Mosaic chỉ giữ lại trung bình khoảng 1/4 diện tích mỗi ảnh, nên lỗi nhỏ có thể bị cắt dở (box bị cắt vẫn được giữ nếu còn đủ diện tích, nên nhãn có thể là một phần của lỗi) hoặc mất hẳn. Ảnh ghép còn trộn các category, làm mất ngữ cảnh với các lỗi phụ thuộc vị trí (ví dụ `misplaced` của transistor, `missing_cable` của cable). Vì vậy không dùng 1.0.
- Ở 0.5, một nửa số mẫu vẫn là ảnh nguyên bản giống tập test (vật nằm giữa, lỗi còn đủ), nên rủi ro bị giới hạn mà vẫn có đủ ảnh ghép để thấy tác động.
- 0.3 cũng an toàn, nhưng ảnh ghép chỉ chiếm 30% mẫu nên tác động có thể quá nhỏ so với nhiễu của test lõi (khoảng 127 ảnh lỗi), khó phân biệt có ích hay không. Dùng 0.3 làm phương án dự phòng nếu 0.5 làm giảm AP của các lỗi nhỏ hoặc lỗi phụ thuộc vị trí.

**Vì sao close_mosaic = 10.** Ở những epoch cuối, YOLO tắt hẳn mosaic (cùng mixup, cutmix, copy_paste) và train bằng ảnh đơn nguyên bản, để model và các lớp chuẩn hóa thích nghi với phân bố giống lúc kiểm tra. Với `epochs: 100`, 10 epoch cuối bằng 10% thời gian train, là mặc định của Ultralytics và là tỷ lệ thường dùng. Nếu đổi số epoch, giữ `close_mosaic` khoảng 10% số epoch. Cấu hình hiện tại có `patience: 100` bằng `epochs`, nên train không dừng sớm trước khi mosaic được tắt.

**Cách bật khi train.** Gói dữ liệu v2 đã chứa sẵn hai dòng sau trong `configs/train_main.yaml`, nên chỉ cần chạy `tools/run_train.py` như bình thường (mặc định `--cfg configs/train_main.yaml`):

```yaml
mosaic: 0.5
close_mosaic: 10
```

Bản đối chứng không mosaic (`mosaic: 0.0`, `close_mosaic: 0`, mọi tham số khác giống hệt) nằm ở `configs/train_no_mosaic.yaml`. Sau mỗi run, `run_train.py` đối chiếu `args.yaml` với đúng file cấu hình được truyền vào `--cfg`. Đã chạy thử một lần train ngắn trên CPU với cấu hình này: `args.yaml` khớp, Albumentations ngầm tắt, ảnh train batch có khoảng một nửa là ảnh ghép (xem `reports/05_smoke_train/`).

**Cách kiểm chứng và quy tắc dùng chung.**
- Mức mosaic này **chưa có thí nghiệm chứng minh là tốt hơn**. Trước khi chạy các ablation CBAM, chạy một lần so sánh trên cùng cấu hình split: (A) `--cfg configs/train_no_mosaic.yaml` và (B) `--cfg configs/train_main.yaml`. Chọn theo **val**, không chọn theo test.
- Xem thêm AP riêng của các lỗi dễ bị mosaic ảnh hưởng (lỗi nhỏ của screw, capsule; `misplaced` của transistor; `missing_cable`, `missing_wire` của cable). Nếu các lỗi này giảm rõ rệt, giảm xuống `mosaic: 0.3` hoặc quay về (A), rồi sửa `train_main.yaml` cho thống nhất.
- Mọi run ablation CBAM phải dùng **cùng một thiết lập mosaic** để so sánh công bằng.
