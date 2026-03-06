"""
Camera Quality Checker
- Kiểm tra tất cả thông số camera trước khi đưa vào model AI
- Đánh giá chất lượng hình ảnh (độ sáng, độ nét, nhiễu)
- Hiển thị histogram, FPS, resolution, exposure
- Cảnh báo nếu chất lượng không đạt yêu cầu
"""

import cv2
import numpy as np
import time
from collections import deque
import sys


class CameraQualityChecker:
    """
    Kiểm tra chất lượng camera toàn diện
    """
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        
        # Thresholds cho quality check
        self.min_brightness = 40      # Tối thiểu độ sáng trung bình
        self.max_brightness = 220     # Tối đa độ sáng trung bình
        self.min_sharpness = 100      # Tối thiểu độ nét (Laplacian variance)
        self.max_noise = 15           # Tối đa noise level
        self.min_fps = 10             # Tối thiểu FPS
        
        # Stats tracking
        self.fps_history = deque(maxlen=30)
        self.brightness_history = deque(maxlen=30)
        self.sharpness_history = deque(maxlen=30)
        self.noise_history = deque(maxlen=30)
        
        print("[CameraQualityChecker] Initialized")
    
    def open_camera(self):
        """Mở camera và lấy thông số"""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
        
        # Lấy thông số camera
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"\n{'='*70}")
        print("CAMERA INFORMATION")
        print(f"{'='*70}")
        print(f"Camera ID: {self.camera_id}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS (reported): {fps}")
        print(f"Backend: {self.cap.getBackendName()}")
        
        # Thông số nâng cao (nếu có)
        try:
            brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)
            saturation = self.cap.get(cv2.CAP_PROP_SATURATION)
            exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            gain = self.cap.get(cv2.CAP_PROP_GAIN)
            
            print(f"\nCamera Settings:")
            print(f"  Brightness: {brightness}")
            print(f"  Contrast: {contrast}")
            print(f"  Saturation: {saturation}")
            print(f"  Exposure: {exposure}")
            print(f"  Gain: {gain}")
        except:
            print("\nAdvanced settings not available")
        
        print(f"{'='*70}\n")
        
        return True
    
    def calculate_brightness(self, frame):
        """
        Tính độ sáng trung bình của frame
        Returns: 0-255
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def calculate_sharpness(self, frame):
        """
        Tính độ nét của frame bằng Laplacian variance
        Giá trị cao = ảnh nét, giá trị thấp = ảnh mờ
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance
    
    def calculate_noise(self, frame):
        """
        Ước lượng noise level bằng high-frequency components
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Tính noise bằng standard deviation của Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise = np.std(laplacian)
        
        return noise
    
    def calculate_contrast(self, frame):
        """
        Tính độ tương phản (contrast)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def calculate_histogram(self, frame):
        """
        Tính histogram cho RGB channels
        """
        histograms = []
        colors = ('b', 'g', 'r')
        
        for i, color in enumerate(colors):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            histograms.append(hist)
        
        return histograms
    
    def check_exposure(self, brightness):
        """
        Kiểm tra exposure (quá sáng hoặc quá tối)
        """
        if brightness < self.min_brightness:
            return "UNDEREXPOSED", (0, 0, 255)  # Red
        elif brightness > self.max_brightness:
            return "OVEREXPOSED", (0, 165, 255)  # Orange
        else:
            return "GOOD", (0, 255, 0)  # Green
    
    def check_focus(self, sharpness):
        """
        Kiểm tra focus (nét hay mờ)
        """
        if sharpness < self.min_sharpness:
            return "BLURRY", (0, 0, 255)  # Red
        else:
            return "SHARP", (0, 255, 0)  # Green
    
    def check_noise_level(self, noise):
        """
        Kiểm tra noise level
        """
        if noise > self.max_noise:
            return "HIGH NOISE", (0, 165, 255)  # Orange
        else:
            return "LOW NOISE", (0, 255, 0)  # Green
    
    def draw_histogram(self, frame, histograms):
        """
        Vẽ histogram lên frame
        """
        hist_height = 100
        hist_width = 256
        hist_img = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
        
        for i, (hist, color) in enumerate(zip(histograms, colors)):
            hist = cv2.normalize(hist, hist, 0, hist_height, cv2.NORM_MINMAX)
            
            for x in range(hist_width):
                y = int(hist[x])
                cv2.line(hist_img, (x, hist_height), (x, hist_height - y), color, 1)
        
        return hist_img
    
    def draw_quality_info(self, frame, metrics, fps):
        """
        Vẽ thông tin chất lượng lên frame
        """
        h, w = frame.shape[:2]
        
        # Background cho text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 280), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        y_offset = 40
        line_height = 30
        
        # Title
        cv2.putText(frame, "CAMERA QUALITY CHECK", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += line_height + 10
        
        # FPS
        fps_status = "GOOD" if fps >= self.min_fps else "LOW"
        fps_color = (0, 255, 0) if fps >= self.min_fps else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f} [{fps_status}]", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        y_offset += line_height
        
        # Brightness
        brightness = metrics['brightness']
        exposure_status, exposure_color = self.check_exposure(brightness)
        cv2.putText(frame, f"Brightness: {brightness:.1f} [{exposure_status}]", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, exposure_color, 2)
        y_offset += line_height
        
        # Sharpness
        sharpness = metrics['sharpness']
        focus_status, focus_color = self.check_focus(sharpness)
        cv2.putText(frame, f"Sharpness: {sharpness:.1f} [{focus_status}]", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, focus_color, 2)
        y_offset += line_height
        
        # Noise
        noise = metrics['noise']
        noise_status, noise_color = self.check_noise_level(noise)
        cv2.putText(frame, f"Noise: {noise:.1f} [{noise_status}]", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, noise_color, 2)
        y_offset += line_height
        
        # Contrast
        contrast = metrics['contrast']
        cv2.putText(frame, f"Contrast: {contrast:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += line_height
        
        # Overall quality
        overall_quality = self.calculate_overall_quality(metrics, fps)
        quality_color = self.get_quality_color(overall_quality)
        cv2.putText(frame, f"Overall: {overall_quality}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
        
        return frame
    
    def calculate_overall_quality(self, metrics, fps):
        """
        Tính chất lượng tổng thể
        """
        score = 0
        max_score = 4
        
        # Check FPS
        if fps >= self.min_fps:
            score += 1
        
        # Check brightness
        brightness = metrics['brightness']
        if self.min_brightness <= brightness <= self.max_brightness:
            score += 1
        
        # Check sharpness
        if metrics['sharpness'] >= self.min_sharpness:
            score += 1
        
        # Check noise
        if metrics['noise'] <= self.max_noise:
            score += 1
        
        # Convert to quality level
        percentage = (score / max_score) * 100
        
        if percentage >= 75:
            return "EXCELLENT"
        elif percentage >= 50:
            return "GOOD"
        elif percentage >= 25:
            return "FAIR"
        else:
            return "POOR"
    
    def get_quality_color(self, quality):
        """
        Lấy màu theo quality level
        """
        colors = {
            "EXCELLENT": (0, 255, 0),    # Green
            "GOOD": (0, 255, 255),       # Yellow
            "FAIR": (0, 165, 255),       # Orange
            "POOR": (0, 0, 255)          # Red
        }
        return colors.get(quality, (255, 255, 255))
    
    def run(self):
        """
        Chạy quality checker
        """
        if not self.open_camera():
            return
        
        print("Starting quality check...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'r' - Show detailed report")
        print(f"{'='*70}\n")
        
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                frame_count += 1
                
                # Calculate FPS
                elapsed = time.time() - start_time
                if elapsed > 1.0:
                    fps = frame_count / elapsed
                    self.fps_history.append(fps)
                    frame_count = 0
                    start_time = time.time()
                
                # Calculate metrics
                brightness = self.calculate_brightness(frame)
                sharpness = self.calculate_sharpness(frame)
                noise = self.calculate_noise(frame)
                contrast = self.calculate_contrast(frame)
                
                self.brightness_history.append(brightness)
                self.sharpness_history.append(sharpness)
                self.noise_history.append(noise)
                
                metrics = {
                    'brightness': brightness,
                    'sharpness': sharpness,
                    'noise': noise,
                    'contrast': contrast
                }
                
                # Draw quality info
                display_frame = self.draw_quality_info(frame.copy(), metrics, fps)
                
                # Calculate and draw histogram
                histograms = self.calculate_histogram(frame)
                hist_img = self.draw_histogram(frame, histograms)
                
                # Resize histogram to fit
                h, w = display_frame.shape[:2]
                hist_img = cv2.resize(hist_img, (w - 20, 100))
                
                # Add histogram to bottom
                display_frame[h - 120:h - 20, 10:w - 10] = hist_img
                
                # Show frame
                cv2.imshow('Camera Quality Checker', display_frame)
                
                # Console output
                overall = self.calculate_overall_quality(metrics, fps)
                print(f"\rFPS: {fps:5.1f} | Bright: {brightness:6.1f} | "
                      f"Sharp: {sharpness:6.1f} | Noise: {noise:5.1f} | "
                      f"Quality: {overall:10s}", end='', flush=True)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"camera_quality_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"\n[Saved] Screenshot: {filename}")
                elif key == ord('r'):
                    self.print_detailed_report(metrics, fps)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.print_final_report()
    
    def print_detailed_report(self, metrics, fps):
        """
        In báo cáo chi tiết
        """
        print(f"\n\n{'='*70}")
        print("DETAILED QUALITY REPORT")
        print(f"{'='*70}")
        
        print(f"\nCurrent Metrics:")
        print(f"  FPS: {fps:.2f}")
        print(f"  Brightness: {metrics['brightness']:.2f}")
        print(f"  Sharpness: {metrics['sharpness']:.2f}")
        print(f"  Noise: {metrics['noise']:.2f}")
        print(f"  Contrast: {metrics['contrast']:.2f}")
        
        print(f"\nQuality Assessment:")
        exposure_status, _ = self.check_exposure(metrics['brightness'])
        focus_status, _ = self.check_focus(metrics['sharpness'])
        noise_status, _ = self.check_noise_level(metrics['noise'])
        
        print(f"  Exposure: {exposure_status}")
        print(f"  Focus: {focus_status}")
        print(f"  Noise Level: {noise_status}")
        print(f"  Overall: {self.calculate_overall_quality(metrics, fps)}")
        
        print(f"\nRecommendations:")
        if metrics['brightness'] < self.min_brightness:
            print("  - Increase lighting or camera brightness")
        elif metrics['brightness'] > self.max_brightness:
            print("  - Reduce lighting or camera exposure")
        
        if metrics['sharpness'] < self.min_sharpness:
            print("  - Adjust camera focus")
            print("  - Clean camera lens")
        
        if metrics['noise'] > self.max_noise:
            print("  - Improve lighting conditions")
            print("  - Reduce camera gain/ISO")
        
        if fps < self.min_fps:
            print("  - Check USB connection")
            print("  - Reduce resolution")
            print("  - Close other applications")
        
        print(f"{'='*70}\n")
    
    def print_final_report(self):
        """
        In báo cáo cuối cùng
        """
        if len(self.fps_history) == 0:
            return
        
        print(f"\n\n{'='*70}")
        print("FINAL QUALITY REPORT")
        print(f"{'='*70}")
        
        avg_fps = np.mean(self.fps_history)
        avg_brightness = np.mean(self.brightness_history)
        avg_sharpness = np.mean(self.sharpness_history)
        avg_noise = np.mean(self.noise_history)
        
        print(f"\nAverage Metrics:")
        print(f"  FPS: {avg_fps:.2f} (min: {np.min(self.fps_history):.2f}, "
              f"max: {np.max(self.fps_history):.2f})")
        print(f"  Brightness: {avg_brightness:.2f} (std: {np.std(self.brightness_history):.2f})")
        print(f"  Sharpness: {avg_sharpness:.2f} (std: {np.std(self.sharpness_history):.2f})")
        print(f"  Noise: {avg_noise:.2f} (std: {np.std(self.noise_history):.2f})")
        
        # Final verdict
        metrics = {
            'brightness': avg_brightness,
            'sharpness': avg_sharpness,
            'noise': avg_noise,
            'contrast': 0
        }
        overall = self.calculate_overall_quality(metrics, avg_fps)
        
        print(f"\nFinal Verdict: {overall}")
        
        if overall in ["EXCELLENT", "GOOD"]:
            print("✅ Camera quality is suitable for AI model")
        else:
            print("⚠️  Camera quality may affect AI model performance")
            print("   Please check recommendations above")
        
        print(f"{'='*70}\n")


def main():
    """Main function"""
    
    # Parse camera ID from command line
    camera_id = 0
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except:
            print(f"Invalid camera ID: {sys.argv[1]}")
            print("Usage: python camera_quality_checker.py [camera_id]")
            return
    
    print("\n" + "="*70)
    print("Camera Quality Checker for AI Models")
    print("="*70)
    print(f"Camera ID: {camera_id}")
    print("="*70 + "\n")
    
    checker = CameraQualityChecker(camera_id=camera_id)
    checker.run()


if __name__ == "__main__":
    main()
