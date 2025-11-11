# YOLOv8 OpenVINO - Real-time Object Detection with Webcam

Ứng dụng nhận dạng đối tượng (object detection) thời gian thực sử dụng YOLOv8, OpenVINO và webcam.

## Tính năng

✅ **2 chế độ hoạt động:**
1. **Classification Demo**: Nhận dạng phân loại trên ảnh tĩnh (chế độ cũ)
2. **Object Detection with Webcam**: Nhận dạng đối tượng real-time từ webcam (chế độ mới)

✅ **Tính năng Object Detection:**
- Nhận dạng 80 loại đối tượng từ COCO dataset
- Hiển thị bounding boxes với màu sắc riêng cho mỗi class
- Hiển thị confidence score (độ tin cậy)
- Tính toán và hiển thị FPS (frames per second)
- Hiển thị thời gian inference
- Sử dụng Non-Maximum Suppression (NMS) để loại bỏ các detection trùng lặp

## Yêu cầu hệ thống

### NuGet Packages
- `Sdcb.OpenVINO` >= 0.6.6
- `Sdcb.OpenVINO.runtime.win-x64` >= 2024.2.0
- `Sdcb.OpenVINO.Extensions.OpenCvSharp4` >= 0.6.1
- `OpenCvSharp4` >= 4.10.0
- `OpenCvSharp4.runtime.win` >= 4.10.0

### Phần cứng
- Webcam (cho chế độ object detection)
- CPU hỗ trợ Intel OpenVINO hoặc GPU tương thích

## Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd sdcb-openvino-yolov8-cls
```

### 2. Chuẩn bị YOLOv8 Object Detection Model

Model classification đã có sẵn, nhưng để sử dụng **Object Detection**, bạn cần convert model:

#### Bước 1: Cài đặt Ultralytics

```bash
pip install ultralytics
```

#### Bước 2: Download YOLOv8n Detection Model

```bash
# Tải model PyTorch
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Hoặc sử dụng Python:

```python
from ultralytics import YOLO

# Download model
model = YOLO('yolov8n.pt')  # Tự động download nếu chưa có
```

#### Bước 3: Convert sang OpenVINO

```bash
yolo export model=yolov8n.pt format=openvino
```

Sau khi convert, bạn sẽ có thư mục `yolov8n_openvino_model` chứa:
- `yolov8n.xml` (~200KB)
- `yolov8n.bin` (~6MB)

#### Bước 4: Copy vào project

```bash
# Copy vào thư mục model
cp yolov8n_openvino_model/yolov8n.xml sdcb-openvino-yolov8-cls/model/
cp yolov8n_openvino_model/yolov8n.bin sdcb-openvino-yolov8-cls/model/
```

### 3. Build project

```bash
cd sdcb-openvino-yolov8-cls
dotnet build
```

## Cách sử dụng

### Chạy ứng dụng

```bash
dotnet run
```

Khi chạy, bạn sẽ thấy menu:

```
=== YOLOv8 OpenVINO Demo ===
1. Classification Demo (static image)
2. Object Detection with Webcam (real-time)

Select option (1 or 2):
```

### Chế độ 1: Classification Demo

- Nhập `1` để chạy demo phân loại trên ảnh `hen.jpg`
- Kết quả sẽ hiển thị class name và confidence score
- Hiển thị thời gian preprocess, inference, postprocess

### Chế độ 2: Object Detection with Webcam

- Nhập `2` để khởi động webcam
- Ứng dụng sẽ mở cửa sổ hiển thị video real-time
- Các đối tượng được nhận dạng sẽ có:
  - Bounding box màu sắc riêng
  - Label với tên class và confidence score
  - FPS counter ở góc trên bên trái
  - Thời gian inference và số objects detected

**Điều khiển:**
- Nhấn `q` hoặc `ESC` để thoát

## Cấu trúc Model

### YOLOv8n Classification
- **Input**: `1x3x224x224` (NCHW format, RGB)
- **Output**: `1x1000` (1000 ImageNet classes)

### YOLOv8n Object Detection
- **Input**: `1x3x640x640` (NCHW format, RGB)
- **Output**: `1x84x8400`
  - 84 = 4 (bbox coordinates) + 80 (class scores)
  - 8400 = số lượng predictions
  - Bbox format: (center_x, center_y, width, height)

## Tùy chỉnh

### Thay đổi Confidence Threshold

Trong `Program.cs`, dòng 57-61:

```csharp
using var detector = new WebcamObjectDetection(
    modelPath: modelFile,
    labelsPath: labelsFile,
    confidenceThreshold: 0.25f,  // Tăng để giảm false positives
    nmsThreshold: 0.45f           // Điều chỉnh NMS
);
```

### Thay đổi Camera

```csharp
detector.Run(cameraIndex: 0);  // 0 = default camera, 1 = camera thứ 2, etc.
```

### Sử dụng model YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x

Các model lớn hơn cho độ chính xác cao hơn:

```bash
# Download và convert YOLOv8s (model size lớn hơn, chính xác hơn)
yolo export model=yolov8s.pt format=openvino

# Hoặc YOLOv8m
yolo export model=yolov8m.pt format=openvino
```

Sau đó cập nhật `modelFile` trong `Program.cs`:

```csharp
string modelFile = @"./model/yolov8s.xml";  // Thay vì yolov8n.xml
```

## COCO Dataset Classes

Model detect 80 loại objects từ COCO dataset, bao gồm:

- **Người**: person
- **Xe cộ**: bicycle, car, motorcycle, airplane, bus, train, truck, boat
- **Động vật**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Đồ vật**: chair, couch, bed, dining table, toilet, tv, laptop, keyboard, cell phone
- **Thực phẩm**: banana, apple, sandwich, orange, pizza, donut, cake
- Và nhiều hơn nữa...

Xem danh sách đầy đủ trong file `coco-labels.txt`.

## Hiệu năng

### YOLOv8n (nano) - Fastest
- Size: ~6MB
- Speed: ~2-5ms inference trên CPU Intel i7
- FPS: 30-60 FPS trên CPU
- Accuracy: Good

### YOLOv8s (small)
- Size: ~22MB
- Speed: ~5-10ms inference
- FPS: 20-40 FPS
- Accuracy: Better

### YOLOv8m (medium)
- Size: ~52MB
- Speed: ~10-20ms inference
- FPS: 10-20 FPS
- Accuracy: Very good

## Troubleshooting

### Lỗi: "Model file not found"

```
ERROR: Model file not found: ./model/yolov8n.xml
```

**Giải pháp**: Làm theo hướng dẫn ở phần "Chuẩn bị YOLOv8 Object Detection Model"

### Lỗi: "Failed to open camera"

```
Failed to open camera 0
```

**Giải pháp**:
- Kiểm tra webcam đã được kết nối
- Thử camera index khác (1, 2, ...)
- Kiểm tra quyền truy cập camera
- Đóng các ứng dụng khác đang sử dụng webcam

### FPS thấp

**Giải pháp**:
- Sử dụng model nhỏ hơn (YOLOv8n thay vì YOLOv8m)
- Giảm resolution input
- Tăng confidence threshold để giảm số detections
- Sử dụng GPU thay vì CPU (cần cài runtime GPU của OpenVINO)

### Quá nhiều false positives

**Giải pháp**:
- Tăng `confidenceThreshold` lên 0.4 hoặc 0.5
- Tăng `nmsThreshold`

## Tài liệu tham khảo

- [YOLOv8 Documentation](https://docs.ultralytics.com/models/yolov8/)
- [OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET)
- [OpenVINO Toolkit](https://docs.openvino.ai/)
- [COCO Dataset](https://cocodataset.org/)

## License

MIT License - xem file LICENSE.txt

## Tác giả

Phát triển dựa trên [OpenVINO.NET](https://github.com/sdcb/OpenVINO.NET) by sdcb

---

**Chúc bạn code vui vẻ! 🚀**
