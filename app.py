import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

MODEL_PATH = "runs/segment/ecg_seg-2/weights/best.pt"
OUT_W = 1600
OUT_H = 600

st.set_page_config(page_title="ECG Paper Segment", layout="wide")
st.title("ECG Paper Detection + Warp")


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


def order_points(pts):
    pts = np.array(pts, dtype="float32")

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def fix_ecg_orientation(warp):
    h, w = warp.shape[:2]

    if h > w:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)

    return warp


def process_ecg(img_bgr):
    model = load_model()

    results = model(img_bgr)

    if results[0].masks is None:
        raise Exception("Không detect được ECG paper")

    masks = results[0].masks.data.cpu().numpy()

    best_mask = None
    best_area = 0

    for m in masks:
        m = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]))
        m = (m > 0.5).astype(np.uint8) * 255

        area = cv2.countNonZero(m)

        if area > best_area:
            best_area = area
            best_mask = m

    if best_mask is None:
        raise Exception("Không lấy được mask")

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
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

    dst = np.array([
        [0, 0],
        [OUT_W - 1, 0],
        [OUT_W - 1, OUT_H - 1],
        [0, OUT_H - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img_bgr, M, (OUT_W, OUT_H))
    warp = fix_ecg_orientation(warp)

    debug = img_bgr.copy()

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

    return debug, warp


uploaded_file = st.file_uploader(
    "Upload ảnh ECG",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(image)

    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    try:
        debug_bgr, warp_bgr = process_ecg(img_bgr)

        debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)
        warp_rgb = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Detected ECG Paper")
            st.image(debug_rgb, use_container_width=True)

        with col2:
            st.subheader("Warped ECG")
            st.image(warp_rgb, use_container_width=True)

        cv2.imwrite("ecg_warped.jpg", warp_bgr)

        with open("ecg_warped.jpg", "rb") as f:
            st.download_button(
                "Tải ảnh ECG đã căn chỉnh",
                f,
                file_name="ecg_warped.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(str(e))