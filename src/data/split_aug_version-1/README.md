# [DEPRECATED] Split & Offline Augmentation — Version 1 (Luu Tru Lich Su)

> CANH BAO / DEPRECATION NOTICE:
> Toan bo cac script trong thu muc nay thuoc Giai doan 1 (Testing Phase cu) va KHONG con duoc su dung trong pipeline huan luyen chinh thuc hien tai.

---

## 1. Ly Do Ngung Su Dung (Deprecation Context)
1. **Du lieu chuan moi (Dataset V2):**
   - Da duoc xay dung va dong goi hoan chinh tai docs/menbers-task/KLTN_split_aug_v2/ va chia se duoi dang goi zip KLTN_prep_v2_{split}.zip tren Google Drive.
   - Tap split moi dam bao tinh long nhau chat che (nested split), khong ro ri du lieu (0% leakage), va co tap test loi bat bien (test_core.txt).
2. **Quy chuan Augmentation Online (Profile v2):**
   - Thay vi offline augment theo logic cu, giai doan moi ap dung truc tiep ho so online v2 trong src/training/train_utils.py theo docs/protocols/aug_data.md.

## 2. Muc Dich Luu Tru
- Thu muc nay duoc giu lai chi nham muc dich doi chieu lich su (historical audit trail) phuc vu viec viet bao cao luan van khi so sanh phuong phap tiep can giua Phase 1 va Phase 2.

## 3. Duong Dan Tham Chieu Chinh Thuc Hien Tai
- Quy chuan data v2: docs/menbers-task/KLTN_split_aug_v2/
- Quy chuan augmentation: docs/protocols/aug_data.md
- Cau hinh data YAML: configs/data/exp_2/data_70_15_15.yaml
- Pipeline training & audit: notebooks/07_verify_and_train_dataset_v2.ipynb va notebooks/08_Training_v2.ipynb
