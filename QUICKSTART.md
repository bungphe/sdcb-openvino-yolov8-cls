# 🚀 Quick Start - YOLOv8 Object Detection với Webcam

Hướng dẫn nhanh để chạy ứng dụng nhận dạng objects real-time từ webcam.

## ⚡ Cách nhanh nhất (3 bước)

### 1️⃣ Download và convert model (chỉ cần làm 1 lần)

```bash
python download_and_convert_model.py
```

Script này sẽ tự động:
- Download YOLOv8n detection model
- Convert sang OpenVINO format
- Copy vào thư mục model

**Lưu ý**: Cần cài Python và pip trước.

### 2️⃣ Build project

```bash
cd sdcb-openvino-yolov8-cls
dotnet build
```

### 3️⃣ Chạy ứng dụng

```bash
dotnet run
```

Chọn option `2` để chạy object detection với webcam.

---

## 🎯 Các tùy chọn model

### YOLOv8n - Nhanh nhất (khuyến nghị cho bắt đầu)

```bash
python download_and_convert_model.py --model yolov8n
```

- Speed: ⚡⚡⚡⚡⚡ (30-60 FPS)
- Accuracy: ⭐⭐⭐ (Good)
- Size: 6MB

### YOLOv8s - Cân bằng tốt

```bash
python download_and_convert_model.py --model yolov8s
```

- Speed: ⚡⚡⚡⚡ (20-40 FPS)
- Accuracy: ⭐⭐⭐⭐ (Better)
- Size: 22MB

### YOLOv8m - Chính xác cao

```bash
python download_and_convert_model.py --model yolov8m
```

- Speed: ⚡⚡⚡ (10-20 FPS)
- Accuracy: ⭐⭐⭐⭐⭐ (Very Good)
- Size: 52MB

---

## 📋 Yêu cầu

### Phần mềm
- ✅ .NET 6.0 SDK hoặc mới hơn
- ✅ Python 3.8+ (để download model)
- ✅ Webcam

### NuGet Packages (tự động cài khi build)
- Sdcb.OpenVINO
- Sdcb.OpenVINO.runtime.win-x64
- OpenCvSharp4
- OpenCvSharp4.runtime.win

---

## 🎮 Cách sử dụng

### Khi ứng dụng chạy:

1. Chọn `2` - Object Detection with Webcam
2. Cửa sổ webcam sẽ mở
3. Đưa các vật thể vào trước camera
4. Ứng dụng sẽ hiển thị:
   - Bounding boxes xung quanh objects
   - Tên object và confidence score
   - FPS (frames per second)
   - Thời gian inference

### Thoát:
- Nhấn `q` hoặc `ESC`

---

## 🔧 Troubleshooting

### ❌ Lỗi: "Model file not found"

**Nguyên nhân**: Chưa download và convert model

**Giải pháp**: Chạy `python download_and_convert_model.py`

---

### ❌ Lỗi: "Failed to open camera"

**Nguyên nhân**: Webcam không khả dụng hoặc đang được sử dụng

**Giải pháp**:
- Kiểm tra webcam đã kết nối
- Đóng các app khác đang dùng webcam (Zoom, Teams, Skype...)
- Thử camera khác: Sửa `cameraIndex: 0` thành `cameraIndex: 1` trong `Program.cs`

---

### 😢 FPS thấp (< 10)

**Giải pháp**:
1. Sử dụng model nhỏ hơn: `yolov8n` thay vì `yolov8m`
2. Đóng các ứng dụng khác
3. Tăng confidence threshold (ít detections hơn = nhanh hơn)

---

## 🎨 80 loại objects có thể detect

Model nhận dạng được 80 loại vật thể từ COCO dataset:

**Người & Động vật:**
- person, cat, dog, bird, horse, sheep, cow, elephant, bear...

**Xe cộ:**
- car, bicycle, motorcycle, bus, train, truck, airplane, boat...

**Đồ vật:**
- laptop, cell phone, keyboard, mouse, tv, chair, couch, bed...

**Thực phẩm:**
- banana, apple, pizza, donut, cake, sandwich, orange...

Xem đầy đủ trong `coco-labels.txt`

---

## 📚 Tài liệu đầy đủ

Xem `README_OBJECT_DETECTION.md` cho:
- Hướng dẫn chi tiết
- Cấu hình nâng cao
- Giải thích technical
- Troubleshooting đầy đủ

---

## 💡 Tips

1. **Ánh sáng tốt** = Detection tốt hơn
2. **Objects rõ ràng** = Confidence cao hơn
3. **Không che khuất** = Detect chính xác hơn
4. **Khoảng cách vừa phải** = Best performance

---

**Chúc bạn thành công! 🎉**

Nếu gặp vấn đề, xem `README_OBJECT_DETECTION.md` hoặc tạo issue trên GitHub.
