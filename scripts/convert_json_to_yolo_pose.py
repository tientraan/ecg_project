import json
import cv2
from pathlib import Path
import shutil
import random

ROOT = Path("datasets/ecg_paper")

JSON_DIR = ROOT / "jsons"
IMAGE_DIR = ROOT / "images"
LABEL_DIR = ROOT / "labels"

CLASS_ID = 0
VAL_RATIO = 0.25

for p in [
    IMAGE_DIR / "train",
    IMAGE_DIR / "val",
    LABEL_DIR / "train",
    LABEL_DIR / "val",
]:
    p.mkdir(parents=True, exist_ok=True)

# xóa label cũ
for old_txt in LABEL_DIR.rglob("*.txt"):
    old_txt.unlink()

# xóa cache cũ
for cache in LABEL_DIR.rglob("*.cache"):
    cache.unlink()

image_exts = [".jpg", ".jpeg", ".png", ".bmp"]

items = []

for json_file in JSON_DIR.rglob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_name = data.get("imagePath", json_file.stem + ".jpg")
    img_path = None

    for ext in image_exts:
        candidates = [
            IMAGE_DIR / img_name,
            IMAGE_DIR / "train" / img_name,
            IMAGE_DIR / "val" / img_name,
            IMAGE_DIR / f"{json_file.stem}{ext}",
            IMAGE_DIR / "train" / f"{json_file.stem}{ext}",
            IMAGE_DIR / "val" / f"{json_file.stem}{ext}",
        ]

        for candidate in candidates:
            if candidate.exists():
                img_path = candidate
                break

        if img_path is not None:
            break

    if img_path is None:
        print(f"[SKIP] Không tìm thấy ảnh cho {json_file.name}")
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[SKIP] Không đọc được ảnh: {img_path}")
        continue

    h, w = img.shape[:2]

    lines = []

    for shape in data.get("shapes", []):
        points = shape.get("points", [])

        if len(points) < 3:
            continue

        coords = []

        for p in points:
            x, y = p[0], p[1]

            # normalize về 0..1
            x = max(0, min(1, x / w))
            y = max(0, min(1, y / h))

            coords.append(f"{x:.6f}")
            coords.append(f"{y:.6f}")

        line = str(CLASS_ID) + " " + " ".join(coords)
        lines.append(line)

    if not lines:
        print(f"[SKIP] Không có polygon hợp lệ: {json_file.name}")
        continue

    items.append((img_path, json_file.stem, lines))

random.shuffle(items)

val_count = max(1, int(len(items) * VAL_RATIO))

for i, (img_path, stem, lines) in enumerate(items):
    split = "val" if i < val_count else "train"

    dst_img = IMAGE_DIR / split / img_path.name
    dst_txt = LABEL_DIR / split / f"{img_path.stem}.txt"

    if not dst_img.exists():
        shutil.copy2(img_path, dst_img)

    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] {split}: {img_path.name} -> {dst_txt.name}")

print(f"\nDone. Total: {len(items)}")