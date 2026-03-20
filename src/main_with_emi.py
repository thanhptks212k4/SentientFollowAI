#!/usr/bin/env python3

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import numpy as np
cv2.setNumThreads(1)

import time
import gc
from threading import Thread, Lock
from collections import deque
from bytetrack_tracker import ByteTracker
from astra_camera import AstraCamera, PYORBBECSDK_AVAILABLE
from config import *
from decision_maker import VisualServoingDecisionMaker
from emi_sound_interaction import EMISoundInteractionSystem, RobotState, InteractionConfig

from main import ThreadedCamera, PreProcessor, ONNXDetector, depth_dist


class EMIIntegratedSystem:
    def __init__(self):
        print("Initializing EMI Integrated System...")
        
        self.using_astra = False
        if PYORBBECSDK_AVAILABLE and not FORCE_USB_CAMERA:
            try:
                self.camera = AstraCamera(fps=CAMERA_FPS).start()
                self.using_astra = True
                print("Using Astra camera")
            except Exception as e:
                print(f"Astra camera failed: {e}")
        
        if not self.using_astra:
            self.camera = ThreadedCamera(CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS).start()
            print("Using USB camera")
        
        for _ in range(5):
            if self.using_astra:
                frame, *_ = self.camera.read()
            else:
                frame, _ = self.camera.read()
            if frame is not None:
                break
            time.sleep(0.3)
        else:
            raise RuntimeError("Camera initialization failed")
        
        self.detector = ONNXDetector(MODEL_PATH, INPUT_SIZE, CONF_THRESH, IOU_THRESH)
        self.preprocessor = PreProcessor(INPUT_SIZE)
        self.tracker = ByteTracker(TRACK_THRESH, TRACK_BUFFER, MATCH_THRESH)
        self.decision_maker = VisualServoingDecisionMaker()
        
        emi_config = InteractionConfig(
            wake_word="emi ơi",
            search_timeout=10.0,
            listen_timeout=5.0,
            min_person_confidence=CONF_THRESH
        )
        self.emi_system = EMISoundInteractionSystem(self.decision_maker, emi_config)
        
        self.running = False
        self.locked_track_id = None
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = time.time()
        self.gc_counter = 0
        
        print("EMI Integrated System initialized")
    
    def start(self):
        print("Starting EMI Integrated System")
        
        self.emi_system.start()
        
        self.running = True
        self.main_loop()
    
    def stop(self):
        print("Stopping EMI Integrated System")
        
        self.running = False
        self.emi_system.stop()
        self.decision_maker.stop()
        self.decision_maker.close()
        self.camera.stop()
        cv2.destroyAllWindows()
        gc.collect()
        
        print("EMI Integrated System stopped")
    
    def main_loop(self):
        try:
            while self.running:
                if self.using_astra:
                    frame, depth_frame, camera_fps = self.camera.read()
                else:
                    frame, camera_fps = self.camera.read()
                    depth_frame = None
                
                if frame is None:
                    continue
                
                self.frame_count += 1
                
                img, ratio, pad = self.preprocessor.run(frame)
                detections = self.detector.run(img, frame.shape, ratio, pad)
                all_tracks = self.tracker.update(detections)
                
                self.emi_system.process_frame_with_detections(frame, detections)
                
                emi_state = self.emi_system.get_current_state()
                
                if emi_state == RobotState.IDLE:
                    self._handle_normal_tracking(all_tracks, depth_frame, frame)
                
                elif emi_state == RobotState.WAKE:
                    self._display_emi_status(frame, "WAKE - Responding to call")
                
                elif emi_state == RobotState.SEARCH:
                    self._display_emi_status(frame, "SEARCH - Looking for caller")
                
                elif emi_state == RobotState.TARGET_LOCK:
                    self._display_emi_status(frame, "TARGET_LOCK - Caller found")
                
                elif emi_state == RobotState.LISTEN_COMMAND:
                    self._display_emi_status(frame, "LISTEN - Waiting for command")
                
                self._draw_tracking_results(frame, all_tracks, depth_frame)
                
                self._draw_system_status(frame, camera_fps, emi_state)
                
                cv2.imshow(WINDOW_NAME, frame)
                
                key = cv2.waitKey(get_wait_ms()) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._show_statistics()
                elif key == ord('e'):
                    if emi_state == RobotState.IDLE:
                        print("EMI System is ready - say 'Emi ơi' to activate")
                    else:
                        print(f"EMI State: {emi_state.value}")
                
                self.gc_counter += 1
                if self.gc_counter >= GC_INTERVAL:
                    gc.collect()
                    self.gc_counter = 0
        
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        
        finally:
            self.stop()
    
    def _handle_normal_tracking(self, all_tracks, depth_frame, frame):
        current_target = None
        
        if self.locked_track_id is not None:
            for track in all_tracks:
                if track['track_id'] == self.locked_track_id:
                    current_target = track
                    break
        
        if current_target is None and len(all_tracks) > 0:
            self.locked_track_id = all_tracks[0]['track_id']
            current_target = all_tracks[0]
        
        if current_target is not None:
            distance_str, distance_m = depth_dist(depth_frame, current_target['bbox'], frame.shape)
            if distance_m is not None:
                distance_mm = distance_m * 1000.0
                try:
                    action = self.decision_maker.process_target(
                        current_target['bbox'], distance_mm, 
                        frame.shape[1], frame.shape[0], depth_frame
                    )
                except Exception as e:
                    print(f"process_target error: {e}")
                    self.decision_maker.stop()
            else:
                self.decision_maker.stop()
        else:
            if self.locked_track_id is not None:
                self.decision_maker.stop()
                self.locked_track_id = None
    
    def _display_emi_status(self, frame, status_text):
        cv2.putText(frame, f"EMI: {status_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def _draw_tracking_results(self, frame, all_tracks, depth_frame):
        for track in all_tracks:
            tid = track['track_id']
            bbox = track['bbox']
            score = track['score']
            
            try:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(x1 + 1, min(x2, frame.shape[1]))
                y2 = max(y1 + 1, min(y2, frame.shape[0]))
                
                if tid == self.locked_track_id:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    rd_str, rd = depth_dist(depth_frame, bbox, frame.shape)
                    label = f"TARGET ID:{tid} ({score:.2f}){rd_str}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    label = f"ID:{tid} ({score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            except Exception:
                continue
    
    def _draw_system_status(self, frame, camera_fps, emi_state):
        y_offset = 60
        
        cv2.putText(frame, f"Camera: {camera_fps:.1f}fps", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 30
        
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        cv2.putText(frame, f"AI: {self.fps:.1f}fps", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        state_color = (0, 255, 0) if emi_state == RobotState.IDLE else (0, 255, 255)
        cv2.putText(frame, f"EMI: {emi_state.value.upper()}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
        y_offset += 20
        
        if hasattr(self.decision_maker, 'last_command'):
            cv2.putText(frame, f"Nav: {self.decision_maker.last_command}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 20
        
        if self.locked_track_id is not None:
            cv2.putText(frame, f"Locked: ID-{self.locked_track_id}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
    def _show_statistics(self):
        dm_stats = self.decision_maker.get_statistics()
        print(f"\nSystem Statistics:")
        print(f"Camera FPS: {self.fps:.2f}")
        print(f"Tracks: {len(self.tracker.tracked_stracks) if hasattr(self.tracker, 'tracked_stracks') else 0}")
        print(f"Commands: F={dm_stats['move_forward_count']}, B={dm_stats['move_backward_count']}, L={dm_stats['turn_left_count']}, R={dm_stats['turn_right_count']}")
        
        emi_stats = self.emi_system.get_statistics()
        print(f"EMI State: {emi_stats['current_state']}")
        print(f"Wake detections: {emi_stats['wake_word_detections']}")
        print(f"Target locks: {emi_stats['successful_target_locks']}")
        print(f"Search attempts: {emi_stats['search_attempts']}")


def main():
    print("SentientFollowAI with EMI Sound Interaction")
    print("=" * 50)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Show statistics")
    print("  'e' - EMI status")
    print("  Say 'Emi ơi' - Activate EMI")
    print("=" * 50)
    
    try:
        system = EMIIntegratedSystem()
        system.start()
    except Exception as e:
        print(f"System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()