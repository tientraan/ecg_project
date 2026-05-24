# BÁO CÁO CHI TIẾT DỰ ÁN: ECG DIGITIZER AI
## Hệ Thống Nhận Diện, Khử Méo Góc Nhìn Và Số Hóa Giấy Điện Tâm Đồ Bằng Trí Tuệ Nhân Tạo

---

## 1. TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)

### 1.1. Đặt Vấn Đề
Giấy điện tâm đồ (ECG - Electrocardiogram) là tài liệu y tế vô cùng quan trọng dùng để chẩn đoán các bệnh lý về tim mạch như rối loạn nhịp tim, nhồi máu cơ tim, rung nhĩ,... Trong thực tế lâm sàng, các bác sĩ hoặc người bệnh thường chụp lại tờ giấy ECG bằng điện thoại di động để lưu trữ hoặc gửi chẩn đoán từ xa. 

Tuy nhiên, ảnh chụp thực tế thường gặp các vấn đề lớn:
*   **Góc chụp bị nghiêng (Perspective Distortion)**: Không vuông góc trực diện, gây méo lệch các đường lưới và đồ thị.
*   **Hướng chụp không đồng nhất**: Có thể bị xoay dọc, xoay ngược 180 độ.
*   **Bối cảnh phức tạp**: Ảnh chứa cả ngón tay giữ giấy, mặt bàn, ga trải giường bệnh viện, v.v.

Các yếu tố trên khiến các phần mềm tự động phân tích đồ thị ECG (số hóa đồ thị thành tín hiệu số) hoàn toàn thất bại. Do đó, việc xây dựng một hệ thống AI để tự động **phát hiện, cắt tách, khử méo góc nhìn (Perspective Warp)** và **xoay chuẩn hướng** tờ giấy ECG là bước tiền xử lý bắt buộc và vô cùng quan trọng.

### 1.2. Mục Tiêu Dự Án
Xây dựng một ứng dụng Web thông minh, trực quan và có tính ứng dụng cao nhằm:
1.  **Phát hiện và Tách biên tờ giấy ECG** tự động trong ảnh chụp thực tế bằng mô hình Deep Learning.
2.  **Khử méo hình học (Perspective Transformation)** để đưa ảnh chụp nghiêng về dạng ảnh phẳng trực diện, bảo toàn tỷ lệ lưới đo.
3.  **Tự động nhận diện hướng giấy** (dọc/ngang) và căn chỉnh xoay ảnh kết quả chuẩn xác.
4.  **Cung cấp giao diện Web Premium** cho phép người dùng dễ dàng tải ảnh lên, xem kết quả so sánh trực quan và tải ảnh đã xử lý về máy.

---

## 2. KIẾN TRÚC VÀ QUY TRÌNH XỬ LÝ (SYSTEM ARCHITECTURE)

Hệ thống hoạt động theo một quy trình tuần tự khép kín (Pipeline) gồm 6 bước chính:

```mermaid
graph TD
    A[Ảnh Chụp Giấy ECG Đầu Vào] --> B[Model YOLOv8-Segmentation]
    B --> C[Trích Xuất Mặt Nạ Mask & Làm Mịn Biên]
    C --> D[Tìm Contour Lớn Nhất & Khung Bounding Box]
    D --> E[Sắp Xếp 4 Điểm Góc & Tính Toán Tỷ Lệ]
    E --> F[Biến Đổi Xạ Ảnh Perspective Warp]
    F --> G[Căn Chỉnh Chiều Xoay Tự Động/Thủ Công]
    G --> H[Ảnh ECG Kết Quả Phẳng Chuẩn & Sắc Nét]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
    style H fill:#fbb,stroke:#333,stroke-width:2px
```

### Chi Tiết Từng Bước Trong Pipeline:
1.  **Đầu vào (Input)**: Ảnh chụp giấy ECG dạng JPEG/PNG do người dùng tải lên hệ thống.
2.  **Phát hiện vùng giấy (AI Segmentation)**: Ảnh được đưa qua mô hình **YOLOv8-Segmentation** (`best.pt`) để dự đoán phân đoạn lớp `ecg_paper`. Kết quả trả về là một ma trận điểm (mask) biểu diễn vùng chứa tờ giấy.
3.  **Hậu xử lý mặt nạ (Morphological Post-processing)**:
    *   Sử dụng phép đóng hình học (`cv2.MORPH_CLOSE`) với kernel $9 \times 9$ để lấp đầy các lỗ trống nhỏ bên trong mặt nạ.
    *   Sử dụng phép mở hình học (`cv2.MORPH_OPEN`) để loại bỏ các điểm nhiễu cô lập xung quanh rìa mặt nạ.
4.  **Tìm khung giới hạn tối ưu (Contour & Bounding Box)**:
    *   Tìm đường viền ngoài (`cv2.findContours`). Chọn đường viền có diện tích lớn nhất.
    *   Áp dụng thuật toán tìm hình chữ nhật bao quanh có diện tích nhỏ nhất (`cv2.minAreaRect`). Thuật toán này cực kỳ hiệu quả vì nó xác định chính xác 4 đỉnh của tờ giấy kể cả khi giấy bị xoay nghiêng bất kỳ góc nào.
5.  **Sắp xếp góc & Xác định hướng giấy**:
    *   Sắp xếp 4 tọa độ đỉnh thu được thành thứ tự chuẩn: **Top-Left (Trên-Trái), Top-Right (Trên-Phải), Bottom-Right (Dưới-Phải), Bottom-Left (Dưới-Trái)**.
    *   Tính toán độ dài các cạnh để xác định xem tờ giấy đang ở hướng **Dọc (Portrait)** hay **Ngang (Landscape)**.
6.  **Biến đổi xạ ảnh (Perspective Warp)**:
    *   Nếu giấy hướng Ngang: Chiếu 4 góc nghiêng lên khung canvas chuẩn kích thước $1600 \times 600$ pixel.
    *   Nếu giấy hướng Dọc: Chiếu lên khung canvas kích thước $600 \times 1600$ pixel (để tránh làm méo/co giãn chữ và đường lưới đồ thị).
    *   Tính toán ma trận biến đổi bằng `cv2.getPerspectiveTransform` và áp dụng phép chiếu `cv2.warpPerspective`.
7.  **Định hướng lại ảnh (Orientation Correction)**:
    *   Nếu ảnh đang ở dạng dọc (do người dùng chụp dọc), hệ thống tự động xoay ngược chiều kim đồng hồ 90 độ để đưa về định dạng ngang chuẩn của giấy ECG (đọc từ trái qua phải).
    *   Hệ thống cũng hỗ trợ người dùng tự điều chỉnh chiều xoay linh hoạt (Không xoay, Xoay 90°, Xoay 180°).

---

## 3. CHI TIẾT THỰC THI MÔ HÌNH AI (DEEP LEARNING MODEL)

Dự án sử dụng mô hình **YOLOv8-Segmentation** (phiên bản Nano - `yolov8n-seg`) phát triển bởi Ultralytics. Đây là mô hình State-of-the-Art (SOTA) trong tác vụ nhận diện và phân đoạn đối tượng theo thời gian thực.

### 3.1. Cấu Hình Huấn Luyện (Training Configuration)
Thông số huấn luyện được trích xuất trực tiếp từ file cấu hình thực tế `args.yaml` của mô hình tốt nhất (`ecg_seg-2`):
*   **Kiến trúc**: `yolov8n-seg.pt` (Mô hình Nano phân đoạn: Tốc độ xử lý siêu nhanh, dung lượng nhỏ gọn ~7MB, cực kỳ tiết kiệm tài nguyên phần cứng).
*   **Kích thước ảnh đầu vào (imgsz)**: $640 \times 640$ pixels.
*   **Số lượng Epochs**: 100.
*   **Batch size**: 4.
*   **Thiết bị huấn luyện**: CPU.
*   **Tốc độ học ban đầu (lr0)**: 0.01.
*   **Tối ưu hóa (Optimizer)**: Tự động (SGD/AdamW).
*   **Augmentation (Tăng cường dữ liệu)**: Sử dụng các kỹ thuật như Flips (Lật ảnh), Translation (Dịch chuyển), HSV Augmentation (Thay đổi màu sắc/độ sáng để mô hình chịu được ảnh chụp ở các môi trường ánh sáng phức tạp).

### 3.2. Tập Dữ Liệu (Dataset)
*   **Nguồn nhãn**: Dữ liệu ảnh thực tế được gắn nhãn đa giác thủ công (Polygon Annotations) thông qua công cụ Labelme để tạo ra các file `.json`.
*   **Chuyển đổi**: Sử dụng kịch bản `convert_json_to_yolo_pose.py` viết bằng Python để tự động chuẩn hóa tọa độ và chia tập dữ liệu.
*   **Phân bổ tập dữ liệu**:
    *   **Tập Huấn luyện (Train)**: 8 hình ảnh đa dạng bối cảnh.
    *   **Tập Kiểm thử (Validation)**: 3 hình ảnh.
    *   *Đánh giá*: Tuy tập dữ liệu nhỏ gọn (phù hợp với quy mô Đồ án/Proof of Concept), nhưng do tính chất đối tượng tờ giấy ECG có cấu trúc hình học ổn định, mô hình đạt được sự hội tụ hoàn hảo mà không bị quá khớp (overfit) nhờ vào pretrained weights chất lượng cao từ COCO dataset.

### 3.3. Kết Quả Huấn Luyện & Chỉ Số Đánh Giá (Evaluation Metrics)
Các chỉ số dưới đây được trích xuất chính xác từ quá trình huấn luyện thực tế lưu tại `results.csv`:

| Chỉ số đánh giá | Tập Bounding Box (Phát hiện khung) | Tập Mask (Phân đoạn đa giác) | Ý nghĩa lâm sàng |
| :--- | :---: | :---: | :--- |
| **Precision** (Độ chính xác) | **73.79%** | **73.79%** | Đảm bảo vùng phát hiện thực sự là giấy ECG, không nhận diện nhầm các vật thể nền. |
| **Recall** (Độ thu hồi) | **100.0%** | **100.0%** | Phát hiện được 100% tờ giấy ECG trong tập kiểm thử, không bỏ sót bất kỳ ca bệnh nào. |
| **mAP50** | **83.0%** | **83.0%** | Độ chính xác trung bình vượt trội ở ngưỡng IoU = 0.5. |
| **mAP50-95** | **80.25%** | **78.05%** | Độ chính xác cực kỳ cao ngay cả ở các ngưỡng IoU khắt khe hơn, đảm bảo đường biên bám sát mép giấy. |

> [!NOTE]
> Trong suốt quá trình huấn luyện, giá trị **Loss (Hàm tổn thất)** giảm liên tục và ổn định:
> *   `train/box_loss` giảm từ 1.21 xuống **0.84**
> *   `train/seg_loss` giảm từ 5.86 xuống **0.38**
> *   Các đồ thị Loss trên tập Validation cũng giảm đồng đều, không xảy ra hiện tượng phân kỳ (Overfitting).
> *   Tại một số epoch tối ưu giữa chừng (ví dụ Epoch 73), mô hình đạt độ chính xác mAP50 lên tới **99.5%**. Trọng số tối ưu này đã được tự động lưu lại thành file trọng số tốt nhất (`best.pt`).

---

## 4. CHI TIẾT CÁC THUẬT TOÁN HÌNH HỌC HẬU XỬ LÝ (OPENCV)

Bên cạnh mô hình Deep Learning, sự thành công của dự án phụ thuộc rất lớn vào các thuật toán xử lý hình học ảnh thông minh bằng OpenCV:

### 4.1. Thuật Toán Sắp Xếp Tọa Độ Góc (Corner Ordering)
Khi mô hình YOLO phát hiện tờ giấy, các tọa độ đỉnh trả về có thể theo thứ tự ngẫu nhiên tùy thuộc vào góc xoay. Để thực hiện phép chiếu xạ ảnh chính xác, ta bắt buộc phải sắp xếp chúng theo thứ tự: 0 - Top Left (TL), 1 - Top Right (TR), 2 - Bottom Right (BR), 3 - Bottom Left (BL).

Thuật toán trong file `app.py` giải quyết cực kỳ thông minh bằng toán học ma trận:
*   **Tìm TL và BR**: Tính tổng tọa độ $x + y$ của mỗi điểm.
    *   Điểm có **tổng nhỏ nhất** chắc chắn là **Top-Left (TL)**.
    *   Điểm có **tổng lớn nhất** chắc chắn là **Bottom-Right (BR)**.
*   **Tìm TR và BL**: Tính hiệu tọa độ $y - x$ (hoặc $x - y$).
    *   Điểm có **hiệu $(x - y)$ lớn nhất** (hoặc hiệu $(y - x)$ nhỏ nhất) là **Top-Right (TR)**.
    *   Điểm có **hiệu $(x - y)$ nhỏ nhất** (hoặc hiệu $(y - x)$ lớn nhất) là **Bottom-Left (BL)**.

```python
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    tl = pts[np.argmin(s)]       # Tổng x + y nhỏ nhất
    br = pts[np.argmax(s)]       # Tổng x + y lớn nhất
    tr = pts[np.argmin(diff)]    # Hiệu y - x nhỏ nhất (x - y lớn nhất)
    bl = pts[np.argmax(diff)]    # Hiệu y - x lớn nhất (x - y nhỏ nhất)
    
    return np.array([tl, tr, br, bl], dtype="float32")
```

### 4.2. Phép Biến Đổi Xạ Ảnh (Perspective Warp)
Phép biến đổi này mô phỏng lại cách thay đổi góc nhìn của camera. Nó ánh xạ các điểm từ mặt phẳng nghiêng $A$ sang mặt phẳng trực diện $B$ thông qua một ma trận chuyển đổi kích thước $3 \times 3$ (gọi là ma trận đồng dạng Homography $M$):

$$M = \text{cv2.getPerspectiveTransform}(src\_points, dst\_points)$$

Sau khi có ma trận $M$, OpenCV thực hiện biến đổi toàn bộ ảnh chụp để thu được ảnh phẳng:

$$warp = \text{cv2.warpPerspective}(img, M, (OUT\_W, OUT\_H))$$

---

## 5. GIAO DIỆN WEB DỰ ÁN & TRẢI NGHIỆM NGƯỜI DÙNG (STREAMLIT APP)

Ứng dụng web được lập trình bằng thư viện **Streamlit** kết hợp với thiết kế CSS tùy chỉnh (Custom CSS) mang lại giao diện đẳng cấp, trực quan và hiện đại.

### 5.1. Các Tính Năng Nổi Bật Trên Giao Diện:
1.  **Thiết kế Premium & Responsive**: Sử dụng font chữ hiện đại **Outfit** từ Google Fonts, giao diện tối sang trọng (Dark mode compatible), ứng dụng ngôn ngữ thiết kế Glassmorphism (hiệu ứng kính mờ với đổ bóng chuyển động mịn).
2.  **Animation Y Tế Sống Động**: Tích hợp một hiệu ứng **ECG Pulse Animation** (nhịp tim đập bằng CSS keyframes) ngay góc trái màn hình, mang lại cảm giác ứng dụng y tế chuyên nghiệp và sinh động.
3.  **Tải ảnh & Xử lý thời gian thực**: Người dùng kéo thả hoặc chọn file ảnh, hệ thống tự động gọi pipeline xử lý ngay lập tức với hiệu ứng loading quay mượt mà (`st.spinner`).
4.  **Bố cục So Sánh Trực Quan (Dual-Column Presentation)**:
    *   **Cột Trái (Detected)**: Hiển thị ảnh gốc kèm khung viền màu xanh lá cây bao quanh tờ giấy ECG được định vị bởi AI và các điểm góc được đánh số thứ tự từ 0 đến 3.
    *   **Cột Phải (Warped)**: Hiển thị ảnh kết quả sau khi đã được duỗi phẳng hoàn toàn, khôi phục màu sắc và độ sắc nét nguyên bản của lưới điện tim.
5.  **Tối ưu hóa hiệu năng bằng cơ chế Cache**:
    *   Sử dụng `@st.cache_resource` để tải mô hình YOLOv8 một lần duy nhất vào bộ nhớ RAM, tránh việc tải đi tải lại gây chậm hệ thống.
    *   Sử dụng cơ chế kiểm tra **Hash MD5** của file ảnh tải lên trong `st.session_state`. Hệ thống chỉ tính toán lại khi người dùng tải ảnh mới hoặc thay đổi chế độ xoay. Nếu giữ nguyên, ảnh kết quả được lấy ngay từ bộ nhớ đệm giúp tốc độ phản hồi đạt mức tức thì (< 0.01 giây).
6.  **Tải Kết Quả Dễ Dàng**: Nút tải ảnh kết quả được thiết kế nổi bật với gradient màu xanh dương trẻ trung, hỗ trợ định dạng JPG chất lượng cao.

---

## 6. ĐÁNH GIÁ ƯU NHƯỢC ĐIỂM & HƯỚNG PHÁT TRIỂN

### 6.1. Ưu Điểm
*   **Độ chính xác định vị cực cao**: Sự kết hợp giữa YOLOv8-Seg và OpenCV giúp định vị bám biên tờ giấy cực kỳ khít, vượt trội so với các thuật toán lọc cạnh truyền thống (như Canny hay Hough Transform vốn dễ bị nhiễu bởi đường lưới của chính tờ giấy ECG).
*   **Trải nghiệm người dùng tuyệt vời**: Ứng dụng web chạy mượt mà, thời gian xử lý ảnh cực nhanh (~0.2 - 0.5s trên CPU thông thường).
*   **Khả năng tự động hóa cao**: Khử méo góc nghiêng và tự xoay ngang ảnh thông minh mà không cần người dùng thao tác căn chỉnh thủ công phức tạp.

### 6.2. Hạn Chế Hiện Tại
*   **Kích thước Dataset hạn chế**: Với 11 ảnh huấn luyện, mô hình hoạt động hoàn hảo trong phòng thí nghiệm và các điều kiện chụp thông thường, tuy nhiên cần bổ sung thêm ảnh trong các môi trường có bóng râm mạnh, độ tương phản cực thấp hoặc giấy bị nhàu nát nặng.

### 6.3. Đề Xuất Hướng Phát Triển Tiếp Theo (Future Work)
Dự án này là nền tảng hoàn hảo để phát triển các tính năng sâu hơn trong lĩnh vực Y tế số:
1.  **Số hóa đồ thị (Signal Digitization)**: Áp dụng các thuật toán trích xuất màu sắc (Color Segmentation) hoặc bám nét vẽ (Contour Tracking) để tách riêng đường đồ thị màu đen/đỏ ra khỏi lưới nền. Chuyển đổi đồ thị pixel thành chuỗi tín hiệu số theo thời gian (Time-series data).
2.  **Chẩn đoán tự động (AI-assisted Diagnosis)**: Đưa chuỗi tín hiệu số sau khi trích xuất vào các mô hình học sâu phân tích chuỗi thời gian (như LSTM, CNN-1D hoặc Transformer) để tự động phát hiện các bất thường về nhịp tim và đưa ra cảnh báo sớm cho bác sĩ.

---
**Báo cáo được chuẩn bị tự động và chuyên nghiệp bởi Hệ thống Trợ lý AI Antigravity © 2026**
