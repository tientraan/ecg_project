import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import hashlib
import sys
import tempfile
import subprocess
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "segment", "ecg_seg-2", "weights", "best.pt")
OUT_W = 1600
OUT_H = 600

# ==================================================================
# DEWARPNET CONFIG
# ==================================================================
# Đặt repo DewarpNet vào thư mục project, ví dụ:
# ecg_project/
#   app.py
#   DewarpNet/
#     infer.py
#     eval/models/unetnc_doc3d.pkl
#     eval/models/dnetccnl_doc3d.pkl
#
# Nếu infer.py gốc của bạn không hỗ trợ --input_path/--output_path,
# tạo file dewarpnet_adapter.py cạnh app.py và định nghĩa hàm:
#   def dewarp_bgr(img_bgr, wc_model_path, bm_model_path, device="cpu"):
#       return output_bgr
DEWARPNET_DIR = os.path.join(BASE_DIR, "DewarpNet")
DEWARPNET_INFER_PY = os.path.join(DEWARPNET_DIR, "infer.py")
DEWARPNET_ADAPTER_PY = os.path.join(BASE_DIR, "dewarpnet_adapter.py")
DEWARPNET_WC_MODEL = os.path.join(DEWARPNET_DIR, "eval", "models", "unetnc_doc3d_final.pkl")
DEWARPNET_BM_MODEL = os.path.join(DEWARPNET_DIR, "eval", "models", "dnetccnl_doc3d_final.pkl")
DEWARPNET_DEVICE = "cpu"  # hiện bạn đã test thành công với torch CPU

st.set_page_config(
    page_title="ECG Digitizer AI - Paper Detection & Warp",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .brand-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-subtitle {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
    }

    .pulse-container {
        display: inline-block;
        position: relative;
        width: 14px;
        height: 14px;
    }

    .pulse-dot {
        width: 10px;
        height: 10px;
        background: #00F2FE;
        border-radius: 50%;
        position: absolute;
        top: 2px;
        left: 2px;
        box-shadow: 0 0 10px #00F2FE;
    }

    .pulse-ring {
        border: 3px solid #00F2FE;
        border-radius: 30px;
        height: 24px;
        width: 24px;
        position: absolute;
        left: -5px;
        top: -5px;
        animation: pulsate 1.5s ease-out infinite;
        opacity: 0;
    }

    @keyframes pulsate {
        0% {transform: scale(0.1, 0.1); opacity: 0.0;}
        50% {opacity: 1.0;}
        100% {transform: scale(1.2, 1.2); opacity: 0.0;}
    }

    .css-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .css-card:hover {
        border-color: rgba(0, 242, 254, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 242, 254, 0.1);
    }

    .stDownloadButton button {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
        color: white !important;
        border: none !important;
        padding: 12px 28px !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        margin-top: 15px;
    }

    .stDownloadButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5) !important;
        background: linear-gradient(135deg, #00E1EC 0%, #3E9AE6 100%) !important;
    }

    .ai-badge {
        background-color: rgba(0, 242, 254, 0.1);
        color: #00F2FE;
        border: 1px solid rgba(0, 242, 254, 0.2);
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="ai-badge">
        <span class="pulse-container">
            <span class="pulse-dot"></span>
            <span class="pulse-ring"></span>
        </span>
        YOLOv8-Segmentation Powered
    </div>
    <h1 class="brand-title">ECG Digitizer AI</h1>
    <div class="brand-subtitle">Trí tuệ nhân tạo nhận diện, căn chỉnh góc chụp, làm phẳng và cải thiện độ rõ nét giấy ECG</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số model tại: {MODEL_PATH}. "
            "Hãy đảm bảo bạn đã đặt file best.pt trong thư mục runs/segment/ecg_seg-2/weights/"
        )
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


def process_ecg(img_bgr, rotation_mode="Tự động xoay ngang (90° xuôi chiều)"):
    model = load_model()
    results = model(img_bgr)

    if len(results) == 0 or results[0].masks is None:
        raise Exception("Không thể phát hiện vùng giấy ECG trong ảnh.")

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
        raise Exception("Lỗi trích xuất mặt nạ phân đoạn.")

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        raise Exception("Không tìm thấy đường viền giấy ECG hợp lệ.")

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)

    src = order_points(box)
    tl, tr, br, bl = src

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    box_width = max(width_top, width_bottom)
    box_height = max(height_left, height_right)
    is_vertical = box_height > box_width

    if is_vertical:
        dst = np.array([
            [0, 0], [OUT_H - 1, 0], [OUT_H - 1, OUT_W - 1], [0, OUT_W - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(img_bgr, M, (OUT_H, OUT_W))
    else:
        dst = np.array([
            [0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]
        ], dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        warp = cv2.warpPerspective(img_bgr, M, (OUT_W, OUT_H))

    if rotation_mode in ("Tự động căn chỉnh", "Tự động xoay ngang (90° xuôi chiều)"):
        if is_vertical:
            warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_mode == "Tự động xoay ngang (90° ngược chiều)":
        if is_vertical:
            warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_mode == "Xoay 90° xuôi chiều kim đồng hồ":
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_mode == "Xoay 90° ngược chiều kim đồng hồ":
        warp = cv2.rotate(warp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_mode == "Xoay 180°":
        warp = cv2.rotate(warp, cv2.ROTATE_180)

    debug = img_bgr.copy()
    cv2.drawContours(debug, [box.astype(np.int32)], -1, (0, 255, 0), 5)
    for i, (x, y) in enumerate(src.astype(int)):
        cv2.circle(debug, (x, y), 12, (0, 0, 255), -1)
        cv2.putText(debug, str(i), (x + 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    return debug, warp


# ==================================================================
# DEWARPNET FUNCTIONS
# ==================================================================

def _load_dewarpnet_adapter():
    """Load adapter nếu bạn tự viết wrapper cho DewarpNet."""
    if not os.path.exists(DEWARPNET_ADAPTER_PY):
        return None
    spec = importlib.util.spec_from_file_location("dewarpnet_adapter", DEWARPNET_ADAPTER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "dewarp_bgr"):
        raise RuntimeError("dewarpnet_adapter.py phải có hàm dewarp_bgr(img_bgr, wc_model_path, bm_model_path, device).")
    return module.dewarp_bgr


def dewarpnet_flatten(img_bgr):
    """
    Chạy DewarpNet sau bước YOLO + perspective warp.

    Repo DewarpNet bạn đang dùng nhận input theo THƯ MỤC:
        --img_path <folder>
        --out_path <folder>

    Vì vậy app sẽ:
    1. Ghi ảnh warp vào thư mục tạm.
    2. Gọi DewarpNet/infer.py bằng subprocess.
    3. Đọc file output cùng tên từ thư mục output tạm.
    4. Nếu lỗi thì trả lại ảnh warp gốc để app không crash.
    """
    try:
        if not os.path.exists(DEWARPNET_INFER_PY):
            return img_bgr, f"Không thấy infer.py: {DEWARPNET_INFER_PY}"
        if not os.path.exists(DEWARPNET_WC_MODEL):
            return img_bgr, f"Không thấy Shape model: {DEWARPNET_WC_MODEL}"
        if not os.path.exists(DEWARPNET_BM_MODEL):
            return img_bgr, f"Không thấy BM model: {DEWARPNET_BM_MODEL}"

        # Nếu sau này bạn tự viết adapter thì app vẫn ưu tiên adapter.
        adapter = _load_dewarpnet_adapter()
        if adapter is not None:
            out = adapter(
                img_bgr.copy(),
                wc_model_path=DEWARPNET_WC_MODEL,
                bm_model_path=DEWARPNET_BM_MODEL,
                device=DEWARPNET_DEVICE
            )
            if out is None:
                return img_bgr, "Adapter DewarpNet trả về None. Đã dùng ảnh warp thay thế."
            out = cv2.resize(out, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
            return out, "DewarpNet adapter OK."

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = os.path.join(tmpdir, "inp")
            output_dir = os.path.join(tmpdir, "uw")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            input_name = "ecg_warped.jpg"
            input_path = os.path.join(input_dir, input_name)
            output_path = os.path.join(output_dir, input_name)
            cv2.imwrite(input_path, img_bgr)

            cmd = [
                sys.executable, DEWARPNET_INFER_PY,
                "--wc_model_path", DEWARPNET_WC_MODEL,
                "--bm_model_path", DEWARPNET_BM_MODEL,
                "--img_path", input_dir,
                "--out_path", output_dir,
            ]

            proc = subprocess.run(
                cmd,
                cwd=DEWARPNET_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=240
            )

            if proc.returncode != 0:
                msg = proc.stderr.strip() or proc.stdout.strip() or "DewarpNet infer.py lỗi không rõ."
                return img_bgr, "DewarpNet lỗi, đã dùng ảnh warp thay thế: " + msg[-700:]

            if not os.path.exists(output_path):
                # fallback: tìm bất kỳ ảnh nào trong output_dir
                candidates = [
                    os.path.join(output_dir, f)
                    for f in os.listdir(output_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if not candidates:
                    return img_bgr, "DewarpNet chạy xong nhưng không tạo output. Đã dùng ảnh warp thay thế."
                output_path = candidates[0]

            out = cv2.imread(output_path)
            if out is None:
                return img_bgr, "Không đọc được output DewarpNet. Đã dùng ảnh warp thay thế."

            out = cv2.resize(out, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
            return out, "DewarpNet OK. Nếu ảnh bị méo, hãy tắt checkbox DewarpNet."

    except subprocess.TimeoutExpired:
        return img_bgr, "DewarpNet quá thời gian xử lý. Đã dùng ảnh warp thay thế."
    except Exception as e:
        return img_bgr, f"DewarpNet exception, đã dùng ảnh warp thay thế: {e}"


# ==================================================================
# ENHANCE FUNCTIONS — VIẾT LẠI HOÀN TOÀN
# ==================================================================

def sharpness_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def contrast_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray.std()


def apply_clahe(img_bgr, clip_limit=2.0, tile_grid_size=8):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size))
    )
    l2 = clahe.apply(l)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def unsharp_mask(img_bgr, sigma=1.2, amount=1.5):
    blur = cv2.GaussianBlur(img_bgr, (0, 0), sigma)
    return cv2.addWeighted(img_bgr, 1.0 + amount, blur, -amount, 0)


def denoise_image(img_bgr, strength=5):
    if strength <= 0:
        return img_bgr
    return cv2.fastNlMeansDenoisingColored(
        img_bgr, None, int(strength), int(strength), 7, 21
    )


def frequency_sharpen(img_bgr, radius=2.0, amount=2.0):
    """High-pass sharpen: hiệu quả hơn unsharp_mask cho đường mảnh."""
    blur = cv2.GaussianBlur(img_bgr, (0, 0), float(radius))
    high_pass = cv2.subtract(img_bgr, blur)
    sharpened = cv2.addWeighted(img_bgr, 1.0, high_pass, float(amount), 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def ecg_line_boost(img_bgr, sensitivity=15, boost=1.4):
    """
    Phát hiện pixel tối (đường ECG) dựa trên độ chênh với nền,
    sau đó làm đậm riêng những pixel đó. Không ảnh hưởng nền sáng.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg_estimate = cv2.medianBlur(gray, 51)
    diff = bg_estimate.astype(np.int16) - gray.astype(np.int16)
    line_mask = np.clip(diff - sensitivity, 0, 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    line_mask = cv2.dilate(line_mask, kernel, iterations=1)
    line_mask_3ch = cv2.cvtColor(line_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    darken_factor = 1.0 / max(boost, 1.01)
    result = img_bgr.astype(np.float32)
    result = result * (1.0 - line_mask_3ch * (1.0 - darken_factor))
    return np.clip(result, 0, 255).astype(np.uint8)


def normalize_background(img_bgr, blur_radius=101):
    """
    Division normalization: loại bỏ gradient sáng không đều trên nền
    (thường gặp khi chụp ECG bằng điện thoại, đèn một bên).
    """
    # blur_radius phải lẻ
    r = int(blur_radius)
    if r % 2 == 0:
        r += 1
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(gray, (0, 0), r)
    mean_bg = float(np.mean(bg))

    result = np.zeros_like(img_bgr, dtype=np.float32)
    for i in range(3):
        ch = img_bgr[:, :, i].astype(np.float32)
        bg_f = bg.astype(np.float32)
        result[:, :, i] = np.divide(ch, bg_f + 1e-6) * mean_bg

    return np.clip(result, 0, 255).astype(np.uint8)


def bilateral_clahe(img_bgr, d=9, sigma_color=75, sigma_space=75,
                    clip=2.5, tile=8):
    """Bilateral filter (edge-preserving) + CLAHE."""
    smoothed = cv2.bilateralFilter(img_bgr, d, sigma_color, sigma_space)
    return apply_clahe(smoothed, clip_limit=clip, tile_grid_size=tile)


def adaptive_local_contrast(img_bgr, tile=32, clip=3.0):
    """CLAHE tile nhỏ + gamma correction nhẹ."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip),
        tileGridSize=(int(tile), int(tile))
    )
    l2 = clahe.apply(l)
    lut = np.array([min(255, int(((i / 255.0) ** 0.9) * 255)) for i in range(256)],
                   dtype=np.uint8)
    l2 = cv2.LUT(l2, lut)
    merged = cv2.merge([l2, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_ecg_image(
    img_bgr,
    method="ECG Pro (Khuyến nghị)",
    denoise_strength=5,
    clahe_clip=2.0,
    clahe_grid=8,
    sharp_amount=1.5,
    sharp_sigma=1.2
):
    img = img_bgr.copy()

    if method == "ECG Pro (Khuyến nghị)":
        # 1. Chuẩn hoá nền không đều
        # 2. Bilateral + CLAHE (giữ edge, tăng contrast)
        # 3. Tăng đường tín hiệu
        # 4. High-pass sharpen
        img = normalize_background(img, blur_radius=101)
        img = bilateral_clahe(img, d=9, sigma_color=75, sigma_space=75,
                              clip=clahe_clip, tile=clahe_grid)
        img = ecg_line_boost(img, sensitivity=12, boost=sharp_amount * 0.9)
        img = frequency_sharpen(img, radius=sharp_sigma, amount=sharp_amount * 0.8)

    elif method == "ECG Line Boost":
        # Tối đa hoá độ đậm của đường tín hiệu (2 lần boost)
        img = normalize_background(img)
        img = bilateral_clahe(img, clip=clahe_clip, tile=clahe_grid)
        img = ecg_line_boost(img, sensitivity=10, boost=sharp_amount * 1.2)
        img = ecg_line_boost(img, sensitivity=8,  boost=sharp_amount * 0.6)
        img = unsharp_mask(img, sigma=sharp_sigma, amount=sharp_amount * 0.5)

    elif method == "Chuẩn hoá nền + Tần số cao":
        # Tốt cho ảnh chụp ánh sáng không đều
        img = normalize_background(img)
        img = apply_clahe(img, clip_limit=clahe_clip, tile_grid_size=clahe_grid)
        img = frequency_sharpen(img, radius=sharp_sigma, amount=sharp_amount)

    elif method == "Bilateral + CLAHE":
        # Nhẹ, giữ màu tự nhiên
        img = bilateral_clahe(img, d=9, sigma_color=75, sigma_space=75,
                              clip=clahe_clip, tile=clahe_grid)
        img = frequency_sharpen(img, radius=sharp_sigma, amount=sharp_amount * 0.7)

    elif method == "Denoise + CLAHE + Sharpen":
        # Thuật toán cũ (giữ lại để so sánh)
        img = denoise_image(img, denoise_strength)
        img = apply_clahe(img, clahe_clip, clahe_grid)
        img = unsharp_mask(img, sharp_sigma, sharp_amount)

    elif method == "Super Sharpen ECG":
        img = denoise_image(img, denoise_strength)
        img = apply_clahe(img, clahe_clip, clahe_grid)
        img = ecg_line_boost(img, sensitivity=15, boost=sharp_amount)
        img = frequency_sharpen(img, radius=sharp_sigma, amount=sharp_amount)

    elif method == "Chỉ giảm nhiễu":
        img = bilateral_clahe(img, d=11, sigma_color=80, sigma_space=80,
                              clip=1.5, tile=clahe_grid)

    elif method == "Chỉ tăng tương phản":
        img = adaptive_local_contrast(img, tile=clahe_grid, clip=clahe_clip)

    elif method == "Chỉ làm nét":
        img = frequency_sharpen(img, radius=sharp_sigma, amount=sharp_amount)

    return img


# ==================================================================
# SIDEBAR
# ==================================================================

with st.sidebar:
    st.markdown("### ⚙️ Cấu hình căn chỉnh")

    rotation_mode = st.selectbox(
        "🔄 Chiều xoay ảnh kết quả",
        options=[
            "Tự động xoay ngang (90° xuôi chiều)",
            "Tự động xoay ngang (90° ngược chiều)",
            "Không xoay",
            "Xoay 90° xuôi chiều kim đồng hồ",
            "Xoay 90° ngược chiều kim đồng hồ",
            "Xoay 180°"
        ],
        index=0
    )

    use_dewarpnet = st.checkbox(
        "📄 Thêm ảnh làm phẳng bằng DewarpNet",
        value=True,
        help="Chạy DewarpNet sau bước YOLO crop + Perspective Warp để xử lý cong giấy. Cần có repo/model DewarpNet trong project."
    )

    st.markdown("---")
    st.markdown("### ✨ Cấu hình làm rõ nét")

    enhance_method = st.selectbox(
        "Thuật toán làm rõ nét",
        options=[
            "ECG Pro (Khuyến nghị)",
            "ECG Line Boost",
            "Chuẩn hoá nền + Tần số cao",
            "Bilateral + CLAHE",
            "Denoise + CLAHE + Sharpen",
            "Super Sharpen ECG",
            "Chỉ giảm nhiễu",
            "Chỉ tăng tương phản",
            "Chỉ làm nét"
        ],
        index=0,
        help=(
            "ECG Pro: pipeline 4 bước tối ưu nhất cho ECG.\n"
            "ECG Line Boost: khi đường tín hiệu quá mờ, cần làm đậm tối đa.\n"
            "Chuẩn hoá nền: khi ảnh bị bóng đổ hoặc ánh sáng không đều.\n"
            "Bilateral + CLAHE: khi ảnh đã khá rõ, chỉ cần tăng contrast nhẹ."
        )
    )

    st.markdown("**Thông số chi tiết**")

    denoise_strength = st.slider("Giảm nhiễu", 0, 20, 5, 1)
    clahe_clip = st.slider("Tăng tương phản CLAHE", 1.0, 5.0, 2.0, 0.1)
    clahe_grid = st.slider("Kích thước lưới CLAHE", 4, 16, 8, 1)
    sharp_amount = st.slider("Độ làm nét / tăng đường ECG", 1.0, 3.0, 1.5, 0.1)
    sharp_sigma = st.slider("Bán kính làm nét", 0.5, 3.0, 1.2, 0.1)

    st.markdown("---")
    st.markdown("### 📋 Hướng dẫn sử dụng")
    st.info(
        "1. Tab Căn chỉnh ECG: tải ảnh giấy ECG để AI crop, xoay và duỗi phẳng.\n"
        "2. Tab Làm rõ nét ảnh: tải ảnh để tăng độ rõ.\n"
        "3. Khuyến nghị dùng ECG Pro cho hầu hết trường hợp.\n"
        "4. Nếu ảnh bị bóng đổ, thử 'Chuẩn hoá nền + Tần số cao'.\n"
        "5. Nếu đường tín hiệu quá mờ, thử 'ECG Line Boost'."
    )

    st.markdown("---")
    st.markdown("### 🤖 Công nghệ")
    st.markdown(
        "⚡ **YOLOv8 Segmentation**: nhận diện giấy ECG.\n\n"
        "📏 **Perspective Transform**: duỗi phẳng ảnh nghiêng.\n\n"
        "🧹 **Background Normalisation**: chuẩn hoá nền không đều.\n\n"
        "🔬 **Bilateral Filter**: giảm nhiễu giữ cạnh sắc nét.\n\n"
        "⚡ **ECG Line Boost**: tăng đậm đường tín hiệu.\n\n"
        "✨ **High-pass Sharpen**: làm rõ tần số cao."
    )

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>ECG Digitizer AI © 2026</p>",
                unsafe_allow_html=True)


# ==================================================================
# MAIN TABS
# ==================================================================

tab1, tab2 = st.tabs(["📐 Căn chỉnh ECG", "✨ Làm rõ nét ảnh"])


with tab1:
    col_upload, _ = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Tải lên hình ảnh giấy ECG của bạn (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="warp_uploader"
        )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()

        if "last_processed_hash" not in st.session_state:
            st.session_state.last_processed_hash = ""
            st.session_state.last_rotation_mode = ""
            st.session_state.debug_rgb = None
            st.session_state.base_warp_bgr = None
            st.session_state.base_dewarp_bgr = None
            st.session_state.dewarp_status = ""
            st.session_state.last_use_dewarpnet = None
            st.session_state.user_rotation = 0

        if st.session_state.last_processed_hash != file_hash:
            st.session_state.user_rotation = 0

        if (st.session_state.last_processed_hash != file_hash
                or st.session_state.last_rotation_mode != rotation_mode
                or st.session_state.last_use_dewarpnet != use_dewarpnet):
            with st.spinner("🧠 Hệ thống đang xử lý và phân tách ECG bằng AI..."):
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    img_rgb = np.array(image)
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    debug_bgr, warp_bgr = process_ecg(img_bgr, rotation_mode=rotation_mode)
                    debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)

                    dewarp_bgr = None
                    dewarp_status = "DewarpNet tắt."
                    if use_dewarpnet:
                        dewarp_bgr, dewarp_status = dewarpnet_flatten(warp_bgr)

                    st.session_state.last_processed_hash = file_hash
                    st.session_state.last_rotation_mode = rotation_mode
                    st.session_state.last_use_dewarpnet = use_dewarpnet
                    st.session_state.debug_rgb = debug_rgb
                    st.session_state.base_warp_bgr = warp_bgr
                    st.session_state.base_dewarp_bgr = dewarp_bgr
                    st.session_state.dewarp_status = dewarp_status
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình xử lý: {str(e)}")

        if (st.session_state.last_processed_hash == file_hash
                and st.session_state.debug_rgb is not None
                and st.session_state.base_warp_bgr is not None):

            st.markdown("<br>", unsafe_allow_html=True)
            warp_bgr = st.session_state.base_warp_bgr.copy()

            if st.session_state.user_rotation == 90:
                warp_bgr = cv2.rotate(warp_bgr, cv2.ROTATE_90_CLOCKWISE)
            elif st.session_state.user_rotation == 180:
                warp_bgr = cv2.rotate(warp_bgr, cv2.ROTATE_180)
            elif st.session_state.user_rotation == 270:
                warp_bgr = cv2.rotate(warp_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            dewarp_bgr = None
            if use_dewarpnet and st.session_state.base_dewarp_bgr is not None:
                dewarp_bgr = st.session_state.base_dewarp_bgr.copy()
                if st.session_state.user_rotation == 90:
                    dewarp_bgr = cv2.rotate(dewarp_bgr, cv2.ROTATE_90_CLOCKWISE)
                elif st.session_state.user_rotation == 180:
                    dewarp_bgr = cv2.rotate(dewarp_bgr, cv2.ROTATE_180)
                elif st.session_state.user_rotation == 270:
                    dewarp_bgr = cv2.rotate(dewarp_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

            warp_rgb = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode(".jpg", warp_bgr)
            download_bytes = buffer.tobytes()

            if use_dewarpnet:
                col1, col2, col3 = st.columns(3)
            else:
                col1, col2 = st.columns(2)
            with col1:
                st.markdown("<div class='css-card'>", unsafe_allow_html=True)
                st.subheader("🎯 Nhận diện giấy ECG")
                st.image(st.session_state.debug_rgb, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown("<div class='css-card'>", unsafe_allow_html=True)
                st.subheader("📐 Kết quả duỗi phẳng")
                st.image(warp_rgb, use_container_width=True)
                st.markdown(
                    "<p style='margin-top:15px;margin-bottom:5px;font-weight:500;"
                    "font-size:0.9rem;color:#A0AEC0;'>🔄 Xoay nhanh kết quả:</p>",
                    unsafe_allow_html=True
                )
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    if st.button("↪️ Xoay Trái 90°", key="rot_ccw", use_container_width=True):
                        st.session_state.user_rotation = (st.session_state.user_rotation - 90) % 360
                        st.rerun()
                with col_btn2:
                    if st.button("🔁 Xoay 180°", key="rot_180", use_container_width=True):
                        st.session_state.user_rotation = (st.session_state.user_rotation + 180) % 360
                        st.rerun()
                with col_btn3:
                    if st.button("↩️ Xoay Phải 90°", key="rot_cw", use_container_width=True):
                        st.session_state.user_rotation = (st.session_state.user_rotation + 90) % 360
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            if use_dewarpnet:
                with col3:
                    st.markdown("<div class='css-card'>", unsafe_allow_html=True)
                    st.subheader("📄 DewarpNet Flatten")
                    if dewarp_bgr is not None:
                        dewarp_rgb = cv2.cvtColor(dewarp_bgr, cv2.COLOR_BGR2RGB)
                        st.image(dewarp_rgb, use_container_width=True)
                        st.caption(st.session_state.dewarp_status)

                        _, dewarp_buffer = cv2.imencode(".jpg", dewarp_bgr)
                        st.download_button(
                            label="💾 Tải ảnh DewarpNet flatten (.jpg)",
                            data=dewarp_buffer.tobytes(),
                            file_name=f"ecg_dewarpnet_{file_hash[:8]}.jpg",
                            mime="image/jpeg",
                            key="download_dewarpnet"
                        )
                    else:
                        st.warning(st.session_state.dewarp_status or "Không có kết quả DewarpNet.")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div style='max-width: 400px; margin: 0 auto;'>", unsafe_allow_html=True)
            st.download_button(
                label="💾 Tải ảnh ECG đã căn chỉnh (.jpg)",
                data=download_bytes,
                file_name=f"ecg_warped_{file_hash[:8]}.jpg",
                mime="image/jpeg"
            )
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        col_info, _ = st.columns([2, 1])
        with col_info:
            st.info("💡 Vui lòng kéo thả hoặc click chọn hình ảnh ECG ở phía trên để bắt đầu.")


with tab2:
    st.markdown("### ✨ Làm rõ nét ảnh ECG")
    st.caption(
        "Tải ảnh lên, chọn thuật toán ở sidebar. "
        "Khuyến nghị dùng **ECG Pro** cho hầu hết trường hợp."
    )

    enhance_file = st.file_uploader(
        "Tải ảnh cần làm rõ nét",
        type=["jpg", "jpeg", "png"],
        key="enhance_uploader"
    )

    if enhance_file is not None:
        image = Image.open(enhance_file).convert("RGB")
        img_rgb = np.array(image)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        with st.spinner("⏳ Đang xử lý ảnh..."):
            enhanced_bgr = enhance_ecg_image(
                img_bgr,
                method=enhance_method,
                denoise_strength=denoise_strength,
                clahe_clip=clahe_clip,
                clahe_grid=clahe_grid,
                sharp_amount=sharp_amount,
                sharp_sigma=sharp_sigma
            )

        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)

        before_sharpness = sharpness_score(img_bgr)
        after_sharpness = sharpness_score(enhanced_bgr)
        before_contrast = contrast_score(img_bgr)
        after_contrast = contrast_score(enhanced_bgr)

        sharpness_improve = ((after_sharpness - before_sharpness) / (before_sharpness + 1e-6)) * 100
        contrast_improve = ((after_contrast - before_contrast) / (before_contrast + 1e-6)) * 100

        st.markdown("<br>", unsafe_allow_html=True)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Độ nét trước", f"{before_sharpness:.2f}")
        with metric_col2:
            st.metric("Độ nét sau", f"{after_sharpness:.2f}",
                      delta=f"{sharpness_improve:.1f}%")
        with metric_col3:
            st.metric("Tương phản", f"{after_contrast:.2f}",
                      delta=f"{contrast_improve:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='css-card'>", unsafe_allow_html=True)
            st.subheader("🖼️ Ảnh gốc")
            st.image(img_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='css-card'>", unsafe_allow_html=True)
            st.subheader("✨ Ảnh đã làm rõ")
            st.image(enhanced_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        _, buffer = cv2.imencode(".jpg", enhanced_bgr)
        st.markdown("<div style='max-width: 400px; margin: 0 auto;'>", unsafe_allow_html=True)
        st.download_button(
            label="💾 Tải ảnh đã làm rõ nét (.jpg)",
            data=buffer.tobytes(),
            file_name="ecg_enhanced.jpg",
            mime="image/jpeg"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.info(
            "📌 Độ nét đo bằng Variance of Laplacian — càng cao = nhiều chi tiết/biên hơn. "
            "So sánh trước/sau trên cùng một ảnh mới có ý nghĩa."
        )

    else:
        st.info("💡 Vui lòng tải ảnh lên để bắt đầu làm rõ nét.")