# Quick Start Guide

## 🚀 Cài Đặt Nhanh (5 phút)

```bash
# 1. Clone project
git clone https://github.com/your-username/SentientFollowAI.git
cd SentientFollowAI

# 2. Tạo virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Cài dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Chạy detection
python src/person_detector.py

# 5. Chạy với tracking
python src/person_detector_with_tracking.py
```

**Nhấn `q` để thoát, `s` để xem statistics, `l` để lock target**

---

## 📊 Hiệu Suất

| Platform | FPS | Latency | CPU | Memory |
|----------|-----|---------|-----|--------|
| Desktop (Intel i5) | 15 | 40-45ms | 40-45% | 200 MB |
| Raspberry Pi 5 | 12-15 | 50-65ms | 45-50% | 200 MB |
| Raspberry Pi 4 | 8-10 | 80-100ms | 50-60% | 200 MB |

---

## 🐛 Xử Lý Lỗi Nhanh

### Camera không hoạt động
```bash
ls /dev/video*
# Thử CAMERA_ID = 1, 2, 3...
```

### Module not found
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### CPU quá cao
```python
# Giảm TARGET_FPS trong src/person_detector.py
TARGET_FPS = 12  # thay vì 15
```

---

## ✅ Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Camera working
- [ ] Detector running

---

**Cần trợ giúp? Xem README.md để biết chi tiết!**
