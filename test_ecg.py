import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/segment/ecg_seg-2/weights/best.pt"
IMG_PATH = "img4.jpg"

OUT_W = 1600
OUT_H = 600


def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


# Orientation check is now embedded inside the main logic


model = YOLO(MODEL_PATH)

img = cv2.imread(IMG_PATH)
if img is None:
    raise Exception(f"Không đọc được ảnh: {IMG_PATH}")

results = model(img)

if results[0].masks is None:
    raise Exception("Không detect được ECG paper")


masks = results[0].masks.data.cpu().numpy()

best_mask = None
best_area = 0

for m in masks:
    m = cv2.resize(m, (img.shape[1], img.shape[0]))
    m = (m > 0.5).astype(np.uint8) * 255

    area = cv2.countNonZero(m)

    if area > best_area:
        best_area = area
        best_mask = m

if best_mask is None:
    raise Exception("Không lấy được mask")

mask = best_mask


kernel = np.ones((9, 9), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    raise Exception("Không tìm thấy contour")

cnt = max(contours, key=cv2.contourArea)


rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.array(box, dtype=np.float32)

src = order_points(box)
tl, tr, br, bl = src

# Calculate physical width and height from the detected box to check portrait vs landscape
width_top = np.linalg.norm(tr - tl)
width_bottom = np.linalg.norm(br - bl)
height_left = np.linalg.norm(bl - tl)
height_right = np.linalg.norm(br - tr)

box_width = max(width_top, width_bottom)
box_height = max(height_left, height_right)

is_vertical = box_height > box_width

if is_vertical:
    # Map to a vertical canvas of OUT_H x OUT_W to avoid squishing
    dst = np.array([
        [0, 0],
        [OUT_H - 1, 0],
        [OUT_H - 1, OUT_W - 1],
        [0, OUT_W - 1]
    ], dtype="float32")
    # Perspective Warp Transformation
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, (OUT_H, OUT_W))
    # Rotate 90 degrees counter-clockwise to make it horizontal
    warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
else:
    # Map directly to a horizontal canvas of OUT_W x OUT_H
    dst = np.array([
        [0, 0],
        [OUT_W - 1, 0],
        [OUT_W - 1, OUT_H - 1],
        [0, OUT_H - 1]
    ], dtype="float32")
    # Perspective Warp Transformation
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img, M, (OUT_W, OUT_H))


debug = img.copy()

cv2.drawContours(
    debug,
    [box.astype(np.int32)],
    -1,
    (0, 255, 0),
    5
)

for i, (x, y) in enumerate(src.astype(int)):
    cv2.circle(debug, (x, y), 12, (0, 0, 255), -1)

    cv2.putText(
        debug,
        str(i),
        (x + 10, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )


cv2.imshow("Detected ECG", debug)
cv2.imshow("Warped ECG", warp)

cv2.imwrite("ecg_warped.jpg", warp)

cv2.waitKey(0)
cv2.destroyAllWindows()