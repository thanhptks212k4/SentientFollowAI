"""
Astra Camera wrapper - Using C++ backend for reliable data access
"""

import sys
import os
import cv2
import numpy as np
import time
from threading import Thread, Lock
import subprocess
import re

# Try to import pyorbbecsdk first
try:
    import pyorbbecsdk as ob
    PYORBBECSDK_AVAILABLE = True
    print("[AstraCamera] pyorbbecsdk imported successfully")
except ImportError as e:
    PYORBBECSDK_AVAILABLE = False
    ob = None
    print(f"[AstraCamera] pyorbbecsdk not available: {e}")

class AstraCamera:
    """Astra camera wrapper using C++ backend for reliable access"""
    
    def __init__(self, fps=30):
        self.fps = fps
        self.frame = None
        self.depth_frame = None
        self.lock = Lock()
        self.running = False
        
        # Camera FPS tracking
        self.camera_fps = 0
        self.fps_frame_count = 0
        self.fps_start_time = time.time()
        
        # C++ backend
        self.process = None
        self.cpp_executable = os.path.abspath("cam_depth/astra_cpp_driver/build/fast_exporter")
        
        if not os.path.exists(self.cpp_executable):
            raise FileNotFoundError(f"C++ executable not found: {self.cpp_executable}")
        
        # Frame dimensions
        self.color_width = 640
        self.color_height = 480
        self.depth_width = 160
        self.depth_height = 120
        
        print("[AstraCamera] Initialized using C++ Backend")
    
    def start(self):
        if self.running:
            return self
        
        try:
            print("[AstraCamera] Starting C++ backend...")
            
            # Start C++ process
            self.process = subprocess.Popen(
                [self.cpp_executable],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            self.running = True
            self.thread = Thread(target=self._update, daemon=True)
            self.thread.start()
            time.sleep(1.0)  # Give time for initialization
            print("[AstraCamera] Thread started")
            return self
            
        except Exception as e:
            print(f"[AstraCamera] Start failed: {e}")
            raise RuntimeError(f"Failed to start Astra camera: {e}")
    
    def _update(self):
        print("[AstraCamera] Update thread started")
        
        frame_count = 0
        start_time = time.time()
        
        while self.running and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                
                # Parse frame info: FRAME:1,DEPTH:640x480,COLOR:640x480,CENTER_DEPTH:1500,COLOR_FILE:/tmp/...,DEPTH_FILE:/tmp/...
                if line.startswith("FRAME:"):
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try:
                            frame_num = int(parts[0].split(':')[1])
                            
                            # Parse dimensions
                            depth_dims = parts[1].split(':')[1].split('x')
                            color_dims = parts[2].split(':')[1].split('x')
                            depth_w, depth_h = int(depth_dims[0]), int(depth_dims[1])
                            color_w, color_h = int(color_dims[0]), int(color_dims[1])
                            
                            # Parse center depth
                            center_depth = int(parts[3].split(':')[1])
                            
                            # Parse file paths
                            color_file = parts[4].split(':')[1]
                            depth_file = parts[5].split(':')[1]
                            
                            # Update dimensions
                            if depth_w != self.depth_width or depth_h != self.depth_height:
                                self.depth_width, self.depth_height = depth_w, depth_h
                            if color_w != self.color_width or color_h != self.color_height:
                                self.color_width, self.color_height = color_w, color_h
                            
                            # Load files immediately
                            try:
                                color_frame = None
                                depth_frame = None
                                
                                # Load color image
                                if os.path.exists(color_file):
                                    color_frame = cv2.imread(color_file)
                                
                                # Load depth data
                                if os.path.exists(depth_file):
                                    depth_data = np.fromfile(depth_file, dtype=np.uint16)
                                    depth_frame = depth_data.reshape((self.depth_height, self.depth_width))
                                
                                # Update frames
                                if color_frame is not None or depth_frame is not None:
                                    with self.lock:
                                        if color_frame is not None:
                                            self.frame = color_frame
                                        if depth_frame is not None:
                                            self.depth_frame = depth_frame
                                        
                                        frame_count += 1
                                        
                                        # Calculate FPS
                                        elapsed = time.time() - start_time
                                        if elapsed >= 1.0:
                                            self.camera_fps = frame_count / elapsed
                                            frame_count = 0
                                            start_time = time.time()
                                
                            except Exception as e:
                                print(f"[AstraCamera] Error loading files: {e}")
                                
                        except Exception as e:
                            print(f"[AstraCamera] Error parsing frame info: {e}")
                
                elif line == "CAMERA_READY":
                    print("[AstraCamera] Camera ready signal received")
                    continue
                    
            except Exception as e:
                print(f"[AstraCamera] Frame read error: {e}")
                break
        
        print("[AstraCamera] Update thread stopped")
    
    def read(self):
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
            depth_frame = self.depth_frame.copy() if self.depth_frame is not None else None
            cam_fps = self.camera_fps
        return frame, depth_frame, cam_fps
    
    def stop(self):
        print("[AstraCamera] Stopping...")
        self.running = False
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
        
        print("[AstraCamera] Stopped")

# Test function
def test_astra_camera():
    """Test Astra camera functionality"""
    
    if not PYORBBECSDK_AVAILABLE:
        print("❌ pyorbbecsdk not available")
        return False
    
    try:
        print("🔍 Testing Astra camera...")
        camera = AstraCamera(fps=30)
        camera.start()
        
        print("📸 Reading frames for 5 seconds...")
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5.0:
            frame, depth_frame, fps = camera.read()
            
            if frame is not None:
                frame_count += 1
                if frame_count % 30 == 0:  # Print every 30 frames
                    print(f"  Frame {frame_count}: {frame.shape}, FPS: {fps:.1f}")
                    if depth_frame is not None:
                        print(f"  Depth: {depth_frame.shape}, Range: {depth_frame.min()}-{depth_frame.max()}mm")
            
            time.sleep(0.033)  # ~30 FPS
        
        camera.stop()
        
        if frame_count > 0:
            print(f"✅ Astra camera test successful! Got {frame_count} frames")
            return True
        else:
            print("❌ No frames received")
            return False
            
    except Exception as e:
        print(f"❌ Astra camera test failed: {e}")
        return False

if __name__ == "__main__":
    test_astra_camera()