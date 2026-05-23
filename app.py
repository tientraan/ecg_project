import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import hashlib

# Dynamic path resolution to ensure compatibility with Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "segment", "ecg_seg-2", "weights", "best.pt")
OUT_W = 1600
OUT_H = 600

# Set premium page config
st.set_page_config(
    page_title="ECG Digitizer AI - Paper Detection & Warp",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS for Stunning Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Typography & Font Family */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sleek Title & Branding */
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
    
    /* Medical ECG Pulse Animation */
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

    /* Styled Cards for UI containers */
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
    
    /* Download Button Style Override */
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

    .stDownloadButton button:active {
        transform: translateY(0px) !important;
    }
    
    /* Custom Badge */
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

# Application Header
st.markdown("""
    <div class="ai-badge">
        <span class="pulse-container">
            <span class="pulse-dot"></span>
            <span class="pulse-ring"></span>
        </span>
        YOLOv8-Segmentation Powered
    </div>
    <h1 class="brand-title">ECG Digitizer AI</h1>
    <div class="brand-subtitle">Trí tuệ nhân tạo nhận diện, căn chỉnh góc chụp và số hóa giấy điện tâm đồ (ECG)</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load and cache the YOLOv8 segmentation model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số model tại: {MODEL_PATH}. "
            "Hãy đảm bảo bạn đã đặt file best.pt trong thư mục runs/segment/ecg_seg-2/weights/"
        )
    return YOLO(MODEL_PATH)


def order_points(pts):
    """Sort coordinates to: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    pts = np.array(pts, dtype="float32")
    
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype="float32")


def fix_ecg_orientation(warp):
    """Automatically rotate image 90 degrees if height exceeds width."""
    h, w = warp.shape[:2]
    if h > w:
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    return warp


def process_ecg(img_bgr):
    """Detect ECG paper, extract contour, apply perspective wrap, and return debug and warped images."""
    model = load_model()
    results = model(img_bgr)
    
    if len(results) == 0 or results[0].masks is None:
        raise Exception("Không thể phát hiện vùng giấy ECG trong ảnh. Hãy thử chụp ảnh rõ ràng hơn với góc chụp vuông góc hơn.")
        
    masks = results[0].masks.data.cpu().numpy()
    
    best_mask = None
    best_area = 0
    
    for m in masks:
        # Resize mask back to original image shape
        m = cv2.resize(m, (img_bgr.shape[1], img_bgr.shape[0]))
        m = (m > 0.5).astype(np.uint8) * 255
        
        area = cv2.countNonZero(m)
        if area > best_area:
            best_area = area
            best_mask = m
            
    if best_mask is None:
        raise Exception("Lỗi trích xuất mặt nạ (mask) phân đoạn.")
        
    # Morphological processing to smooth edges
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find outer contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise Exception("Không tìm thấy đường viền giấy ECG hợp lệ.")
        
    cnt = max(contours, key=cv2.contourArea)
    
    # Fit minimum bounding box
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.float32)
    
    src = order_points(box)
    
    # Standard output canvas coordinates
    dst = np.array([
        [0, 0],
        [OUT_W - 1, 0],
        [OUT_W - 1, OUT_H - 1],
        [0, OUT_H - 1]
    ], dtype="float32")
    
    # Perspective Warp Transformation
    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img_bgr, M, (OUT_W, OUT_H))
    warp = fix_ecg_orientation(warp)
    
    # Draw original image debug overlays
    debug = img_bgr.copy()
    cv2.drawContours(debug, [box.astype(np.int32)], -1, (0, 255, 0), 5)
    
    for i, (x, y) in enumerate(src.astype(int)):
        cv2.circle(debug, (x, y), 12, (0, 0, 255), -1)
        cv2.putText(
            debug,
            str(i),
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3
        )
        
    return debug, warp


# Sidebar Information & Documentation
with st.sidebar:
    st.markdown("### 📋 Hướng dẫn sử dụng")
    st.info(
        "1. **Tải ảnh lên**: Chọn file ảnh chụp giấy điện tâm đồ (ECG).\n"
        "2. **Xử lý tự động**: AI sẽ nhận diện đường biên, khoanh vùng và tự động căn chỉnh góc xoay, khử méo góc nhìn.\n"
        "3. **Tải kết quả**: Tải ngay ảnh kết quả chất lượng cao đã được duỗi phẳng và định hướng chuẩn xác."
    )
    
    st.markdown("---")
    st.markdown("### 🤖 Công nghệ cốt lõi")
    st.markdown(
        "⚡ **YOLOv8 Segmentation**: Model được huấn luyện đặc thù cho việc phát hiện và tách nền giấy ECG từ ảnh chụp thực tế.\n\n"
        "📏 **Perspective Transformation**: Thuật toán hình học xạ ảnh chiếu ảnh nghiêng về dạng xem trực diện phẳng chuẩn xác.\n\n"
        "🔄 **Orientation Correction**: Tự động nhận diện và xoay ngang ảnh nếu hướng giấy bị dọc."
    )
    
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #888;'>ECG Digitizer AI © 2026</p>", unsafe_allow_html=True)


# Main Content Area
col_upload, _ = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Tải lên hình ảnh giấy ECG của bạn (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    # Compute hash of file content to implement efficient caching
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    
    # Initialize session state cache if not existing
    if "last_processed_hash" not in st.session_state:
        st.session_state.last_processed_hash = ""
        st.session_state.debug_rgb = None
        st.session_state.warp_rgb = None
        st.session_state.download_bytes = None
    
    # If it is a new image, run processing and cache it
    if st.session_state.last_processed_hash != file_hash:
        with st.spinner("🧠 Hệ thống đang xử lý và phân tách ECG bằng AI..."):
            try:
                # Open image and convert to OpenCV format
                image = Image.open(uploaded_file).convert("RGB")
                img_rgb = np.array(image)
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                
                # Core processing
                debug_bgr, warp_bgr = process_ecg(img_bgr)
                
                # Convert back to RGB for Streamlit rendering
                debug_rgb = cv2.cvtColor(debug_bgr, cv2.COLOR_BGR2RGB)
                warp_rgb = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2RGB)
                
                # In-memory image encoding for secure concurrent download
                _, buffer = cv2.imencode(".jpg", warp_bgr)
                warp_bytes = buffer.tobytes()
                
                # Save to session state
                st.session_state.last_processed_hash = file_hash
                st.session_state.debug_rgb = debug_rgb
                st.session_state.warp_rgb = warp_rgb
                st.session_state.download_bytes = warp_bytes
                
            except Exception as e:
                st.error(f"❌ Có lỗi xảy ra trong quá trình xử lý: {str(e)}")
                
    # If processing succeeded, render results
    if st.session_state.last_processed_hash == file_hash and st.session_state.debug_rgb is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Dual-column presentation
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='css-card'>", unsafe_allow_html=True)
            st.subheader("🎯 Nhận diện giấy ECG (Detected)")
            st.image(st.session_state.debug_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='css-card'>", unsafe_allow_html=True)
            st.subheader("📐 Kết quả duỗi phẳng (Warped)")
            st.image(st.session_state.warp_rgb, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        # Download Section
        st.markdown("<div style='max-width: 400px; margin: 0 auto;'>", unsafe_allow_html=True)
        st.download_button(
            label="💾 Tải ảnh ECG đã căn chỉnh (.jpg)",
            data=st.session_state.download_bytes,
            file_name=f"ecg_warped_{file_hash[:8]}.jpg",
            mime="image/jpeg"
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    # Beautiful landing placeholder
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_info, _ = st.columns([2, 1])
    with col_info:
        st.info("💡 Vui lòng kéo thả hoặc click chọn hình ảnh ECG ở phía trên để bắt đầu nhận diện và căn chỉnh tự động.")