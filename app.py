import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
from skimage.transform import resize, ProjectiveTransform, warp
from skimage.measure import find_contours
from skimage.morphology import binary_closing, binary_opening, square

MODEL_PATH = "runs/segment/ecg_seg-2/weights/best.pt"
OUT_W = 1600
OUT_H = 600

st.set_page_config(page_title="ECG Paper Segment", layout="wide")
st.title("ECG Paper Detection + Warp")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_box_from_mask(mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise Exception("Mask rỗng")

    pts = np.column_stack([xs, ys]).astype(np.float32)

    # PCA lấy hình chữ nhật xoay gần giống minAreaRect
    center = pts.mean(axis=0)
    pts_centered = pts - center

    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    rotated = pts_centered @ eigvecs

    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)

    corners_rotated = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]],
    ])

    corners = corners_rotated @ eigvecs.T + center

    return order_points(corners)


def fix_ecg_orientation(img):
    h, w = img.shape[:2]

    if h > w:
        img = np.rot90(img, k=3)

    return img


def process_ecg(img_rgb):
    model = load_model()

    results = model(img_rgb)

    if results[0].masks is None:
        raise Exception("Không detect được ECG paper")

    masks = results[0].masks.data.cpu().numpy()

    best_mask = None
    best_area = 0

    for m in masks:
        m = resize(
            m,
            (img_rgb.shape[0], img_rgb.shape[1]),
            preserve_range=True,
            anti_aliasing=False
        )

        mask = m > 0.5
        area = np.sum(mask)

        if area > best_area:
            best_area = area
            best_mask = mask

    if best_mask is None:
        raise Exception("Không lấy được mask")

    mask = binary_closing(best_mask, square(9))
    mask = binary_opening(mask, square(9))

    src = get_box_from_mask(mask)

    dst = np.array([
        [0, 0],
        [OUT_W - 1, 0],
        [OUT_W - 1, OUT_H - 1],
        [0, OUT_H - 1]
    ], dtype=np.float32)

    tform = ProjectiveTransform()
    tform.estimate(dst, src)

    warped = warp(
        img_rgb,
        tform,
        output_shape=(OUT_H, OUT_W),
        preserve_range=True
    ).astype(np.uint8)

    warped = fix_ecg_orientation(warped)

    debug = Image.fromarray(img_rgb.copy())
    draw = ImageDraw.Draw(debug)

    points = [(float(x), float(y)) for x, y in src]
    draw.line(points + [points[0]], fill=(0, 255, 0), width=5)

    for i, (x, y) in enumerate(points):
        r = 10
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))
        draw.text((x + 10, y - 10), str(i), fill=(0, 0, 255))

    return np.array(debug), warped


uploaded_file = st.file_uploader(
    "Upload ảnh ECG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    try:
        debug_rgb, warp_rgb = process_ecg(img_rgb)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Detected ECG Paper")
            st.image(debug_rgb, use_container_width=True)

        with col2:
            st.subheader("Warped ECG")
            st.image(warp_rgb, use_container_width=True)

        out_img = Image.fromarray(warp_rgb)
        out_img.save("ecg_warped.jpg")

        with open("ecg_warped.jpg", "rb") as f:
            st.download_button(
                "Tải ảnh ECG đã căn chỉnh",
                f,
                file_name="ecg_warped.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(str(e))