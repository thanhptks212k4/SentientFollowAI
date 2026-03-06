# Camera Quality Checker Guide

## 🎥 Giới Thiệu

Tool kiểm tra chất lượng camera toàn diện trước khi đưa vào model AI, bao gồm:

- ✅ FPS (Frames Per Second)
- ✅ Độ sáng (Brightness)
- ✅ Độ nét (Sharpness/Focus)
- ✅ Nhiễu (Noise Level)
- ✅ Độ tương phản (Contrast)
- ✅ Histogram RGB
- ✅ Exposure (quá sáng/quá tối)
- ✅ Đánh giá tổng thể (Overall Quality)

## 🚀 Cách Chạy

### Cơ Bản

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\Activate.ps1  # Windows

# Chạy với camera mặc định (ID 0)
python tools/camera_quality_checker.py

# Chạy với camera khác
python tools/camera_quality_checker.py 1  # Camera ID 1
python tools/camera_quality_checker.py 2  # Camera ID 2
```

### Controls

Khi tool đang chạy:

- **'q'** - Thoát
- **'s'** - Lưu screenshot với thông số
- **'r'** - Hiển thị báo cáo chi tiết

## 📊 Các Thông Số Kiểm Tra

### 1. FPS (Frames Per Second)

**Ý nghĩa:** Số khung hình mỗi giây

**Ngưỡng:**
- ✅ GOOD: FPS ≥ 10
- ❌ LOW: FPS < 10

**Ảnh hưởng đến AI:**
- FPS thấp → Tracking không mượt
- FPS cao → Tracking tốt hơn

**Khắc phục:**
```python
# Giảm resolution
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# Hoặc kiểm tra USB connection
ls /dev/video*
```

### 2. Brightness (Độ Sáng)

**Ý nghĩa:** Độ sáng trung bình của frame (0-255)

**Ngưỡng:**
- ❌ UNDEREXPOSED: < 40 (quá tối)
- ✅ GOOD: 40-220
- ❌ OVEREXPOSED: > 220 (quá sáng)

**Ảnh hưởng đến AI:**
- Quá tối → Model không detect được người
- Quá sáng → Mất chi tiết, false positives

**Khắc phục:**
```python
# Tăng độ sáng
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)

# Hoặc cải thiện ánh sáng môi trường
# - Thêm đèn
# - Mở cửa sổ
# - Tránh ngược sáng
```

### 3. Sharpness (Độ Nét)

**Ý nghĩa:** Độ sắc nét của hình ảnh (Laplacian variance)

**Ngưỡng:**
- ❌ BLURRY: < 100 (mờ)
- ✅ SHARP: ≥ 100 (nét)

**Ảnh hưởng đến AI:**
- Ảnh mờ → Model khó detect
- Ảnh nét → Accuracy cao

**Khắc phục:**
```python
# Điều chỉnh focus camera
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Hoặc:
# - Lau lens camera
# - Điều chỉnh khoảng cách
# - Kiểm tra autofocus
```

### 4. Noise (Nhiễu)

**Ý nghĩa:** Mức độ nhiễu trong ảnh

**Ngưỡng:**
- ✅ LOW NOISE: ≤ 15
- ⚠️ HIGH NOISE: > 15

**Ảnh hưởng đến AI:**
- Nhiễu cao → False detections
- Nhiễu thấp → Stable tracking

**Khắc phục:**
```python
# Giảm gain/ISO
cap.set(cv2.CAP_PROP_GAIN, 0)

# Hoặc:
# - Cải thiện ánh sáng
# - Giảm exposure time
# - Dùng camera chất lượng tốt hơn
```

### 5. Contrast (Độ Tương Phản)

**Ý nghĩa:** Độ chênh lệch giữa vùng sáng và tối

**Ảnh hưởng đến AI:**
- Contrast thấp → Khó phân biệt đối tượng
- Contrast cao → Dễ detect

**Khắc phục:**
```python
# Tăng contrast
cap.set(cv2.CAP_PROP_CONTRAST, 0.7)
```

### 6. Histogram

**Ý nghĩa:** Phân bố giá trị pixel cho từng kênh màu (R, G, B)

**Đánh giá:**
- Histogram cân bằng → Ảnh tốt
- Histogram lệch trái → Quá tối
- Histogram lệch phải → Quá sáng
- Histogram tập trung → Contrast thấp

## 📈 Overall Quality Levels

### EXCELLENT (75-100%)
✅ Tất cả thông số đạt yêu cầu  
✅ Sẵn sàng cho AI model  
✅ Tracking sẽ rất ổn định

### GOOD (50-74%)
✅ Hầu hết thông số đạt yêu cầu  
✅ Có thể dùng cho AI model  
⚠️ Có thể cần điều chỉnh nhỏ

### FAIR (25-49%)
⚠️ Một số thông số chưa đạt  
⚠️ AI model có thể hoạt động kém  
🔧 Nên cải thiện trước khi dùng

### POOR (0-24%)
❌ Nhiều thông số không đạt  
❌ AI model sẽ hoạt động rất kém  
🔧 Bắt buộc phải cải thiện

## 🔧 Troubleshooting

### Vấn đề: FPS thấp (< 10)

**Nguyên nhân:**
- USB bandwidth không đủ
- CPU quá tải
- Resolution quá cao

**Giải pháp:**
```bash
# 1. Kiểm tra USB
lsusb
# Dùng USB 3.0 thay vì USB 2.0

# 2. Giảm resolution
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

# 3. Đóng các app khác
```

### Vấn đề: Ảnh quá tối (UNDEREXPOSED)

**Nguyên nhân:**
- Ánh sáng môi trường kém
- Camera brightness thấp
- Exposure time ngắn

**Giải pháp:**
```python
# Tăng brightness
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)

# Tăng exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

# Hoặc thêm ánh sáng
```

### Vấn đề: Ảnh mờ (BLURRY)

**Nguyên nhân:**
- Camera out of focus
- Lens bẩn
- Camera chuyển động

**Giải pháp:**
```python
# Bật autofocus
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

# Hoặc:
# - Lau lens
# - Cố định camera
# - Điều chỉnh khoảng cách
```

### Vấn đề: Nhiễu cao (HIGH NOISE)

**Nguyên nhân:**
- Ánh sáng yếu
- Gain/ISO cao
- Camera chất lượng thấp

**Giải pháp:**
```python
# Giảm gain
cap.set(cv2.CAP_PROP_GAIN, 0)

# Cải thiện ánh sáng
# Dùng camera tốt hơn
```

## 📋 Checklist Trước Khi Chạy AI Model

```
Camera Quality Checklist:

Hardware:
[ ] Camera kết nối USB 3.0
[ ] Lens sạch, không bụi
[ ] Camera cố định, không rung
[ ] Ánh sáng đủ (không quá tối/sáng)

Software:
[ ] FPS ≥ 10
[ ] Brightness: 40-220
[ ] Sharpness ≥ 100
[ ] Noise ≤ 15
[ ] Overall Quality: GOOD hoặc EXCELLENT

Environment:
[ ] Ánh sáng ổn định (không nhấp nháy)
[ ] Không có ngược sáng
[ ] Background không quá phức tạp
[ ] Khoảng cách phù hợp (1-5m)
```

## 🎯 Recommended Settings

### Indoor (Trong Nhà)

```python
# Camera settings
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 30

# Quality thresholds
MIN_BRIGHTNESS = 50
MAX_BRIGHTNESS = 200
MIN_SHARPNESS = 120
MAX_NOISE = 12
```

### Outdoor (Ngoài Trời)

```python
# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Quality thresholds
MIN_BRIGHTNESS = 60
MAX_BRIGHTNESS = 220
MIN_SHARPNESS = 150
MAX_NOISE = 10
```

### Low Light (Ánh Sáng Yếu)

```python
# Camera settings
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)
cap.set(cv2.CAP_PROP_EXPOSURE, -4)
cap.set(cv2.CAP_PROP_GAIN, 50)

# Quality thresholds
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 180
MIN_SHARPNESS = 80
MAX_NOISE = 18
```

## 📊 Example Output

```
======================================================================
CAMERA INFORMATION
======================================================================
Camera ID: 0
Resolution: 640x480
FPS (reported): 30
Backend: MSMF

Camera Settings:
  Brightness: 128.0
  Contrast: 32.0
  Saturation: 64.0
  Exposure: -5.0
  Gain: 0.0
======================================================================

FPS:  15.2 | Bright:  125.3 | Sharp:  245.7 | Noise:   8.2 | Quality: EXCELLENT

======================================================================
FINAL QUALITY REPORT
======================================================================

Average Metrics:
  FPS: 15.23 (min: 14.50, max: 15.80)
  Brightness: 125.34 (std: 3.21)
  Sharpness: 245.67 (std: 12.45)
  Noise: 8.23 (std: 1.12)

Final Verdict: EXCELLENT
✅ Camera quality is suitable for AI model
======================================================================
```

## 🔗 Integration với AI Model

### Kiểm Tra Trước Khi Chạy Detection

```python
from tools.camera_quality_checker import CameraQualityChecker

# Check camera quality
checker = CameraQualityChecker(camera_id=0)
checker.open_camera()

# Get one frame
ret, frame = checker.cap.read()

# Calculate metrics
brightness = checker.calculate_brightness(frame)
sharpness = checker.calculate_sharpness(frame)
noise = checker.calculate_noise(frame)

metrics = {
    'brightness': brightness,
    'sharpness': sharpness,
    'noise': noise,
    'contrast': 0
}

# Check quality
overall = checker.calculate_overall_quality(metrics, 15)

if overall in ["EXCELLENT", "GOOD"]:
    print("✅ Camera ready for AI model")
    # Start detection
else:
    print("⚠️ Camera quality not optimal")
    print("Please improve camera settings")
    # Show recommendations
```

## 📝 Tips & Best Practices

1. **Luôn kiểm tra camera trước khi deploy**
2. **Chạy quality check ít nhất 30 giây để có metrics ổn định**
3. **Lưu screenshot khi quality EXCELLENT để làm reference**
4. **Monitor quality trong quá trình chạy AI model**
5. **Điều chỉnh thresholds theo môi trường cụ thể**

## 🎓 Advanced Usage

### Custom Thresholds

```python
checker = CameraQualityChecker(camera_id=0)

# Customize thresholds
checker.min_brightness = 50
checker.max_brightness = 200
checker.min_sharpness = 150
checker.max_noise = 10
checker.min_fps = 15

checker.run()
```

### Automated Quality Check

```python
def auto_check_camera(camera_id=0, duration=10):
    """
    Tự động kiểm tra camera trong N giây
    Returns: True nếu quality đạt yêu cầu
    """
    checker = CameraQualityChecker(camera_id)
    checker.open_camera()
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        ret, frame = checker.cap.read()
        if not ret:
            continue
        
        brightness = checker.calculate_brightness(frame)
        sharpness = checker.calculate_sharpness(frame)
        noise = checker.calculate_noise(frame)
        
        checker.brightness_history.append(brightness)
        checker.sharpness_history.append(sharpness)
        checker.noise_history.append(noise)
    
    # Calculate average
    avg_brightness = np.mean(checker.brightness_history)
    avg_sharpness = np.mean(checker.sharpness_history)
    avg_noise = np.mean(checker.noise_history)
    
    metrics = {
        'brightness': avg_brightness,
        'sharpness': avg_sharpness,
        'noise': avg_noise,
        'contrast': 0
    }
    
    overall = checker.calculate_overall_quality(metrics, 15)
    
    checker.cap.release()
    
    return overall in ["EXCELLENT", "GOOD"]
```

---

**Ready to check your camera? Run the tool now!** 🎥
