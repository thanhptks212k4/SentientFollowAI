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

class ThreadedCamera:
    def __init__(self, src=0, width=320, height=240, fps=30):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.lock = Lock()
        self.running = False
        self.camera_fps = 0.0
        self._cnt = 0
        self._t0 = time.time()
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {src}")

    def start(self):
        self.running = True
        Thread(target=self._run, daemon=True).start()
        time.sleep(0.5)
        return self

    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self._cnt += 1
                t = time.time()
                if t - self._t0 >= 1.0:
                    self.camera_fps = self._cnt / (t - self._t0)
                    self._cnt = 0
                    self._t0 = t

    def read(self):
        with self.lock:
            f = self.frame.copy() if self.frame is not None else None
            return f, self.camera_fps

    def stop(self):
        self.running = False
        self.cap.release()

class PreProcessor:
    def __init__(self, sz=320):
        self.sz = sz
        self._canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        self._buf = np.empty((1, 3, sz, sz), dtype=np.float32)

    def run(self, frame):
        h, w = frame.shape[:2]
        r = min(self.sz/h, self.sz/w)
        nw, nh = int(round(w*r)), int(round(h*r))
        dw, dh = (self.sz-nw)//2, (self.sz-nh)//2
        cv2.resize(frame, (nw, nh), dst=self._canvas[dh:dh+nh, dw:dw+nw], interpolation=cv2.INTER_LINEAR)
        if dh > 0:
            self._canvas[:dh, :] = 114
            self._canvas[dh+nh:, :] = 114
        if dw > 0:
            self._canvas[:, :dw] = 114
            self._canvas[:, dw+nw:] = 114
        rgb = cv2.cvtColor(self._canvas, cv2.COLOR_BGR2RGB)
        np.divide(rgb.transpose(2,0,1), 255.0, out=self._buf[0])
        return self._buf, r, (dw, dh)

class ONNXDetector:
    def __init__(self, model_path, sz=320, conf=0.45, iou=0.45):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime")
        self.conf = conf
        self.iou = iou
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        try:
            opts.enable_mem_pattern = True
            opts.enable_cpu_mem_arena = False
            opts.enable_profiling = False
        except AttributeError:
            pass
        self.sess = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.inp = self.sess.get_inputs()[0].name
        self.outs = [o.name for o in self.sess.get_outputs()]
        dummy = np.zeros((1,3,sz,sz), dtype=np.float32)
        for _ in range(5):
            self.sess.run(self.outs, {self.inp: dummy})

    def run(self, img, orig_shape, ratio, pad):
        raw = self.sess.run(self.outs, {self.inp: img})[0]
        dets = self._post(raw, orig_shape, ratio, pad)
        del raw
        return dets

    def _post(self, raw, orig_shape, ratio, pad):
        p = raw[0].T
        scores = p[:, 4:]
        cids = np.argmax(scores, axis=1)
        cconf = scores[np.arange(len(scores)), cids]
        mask = (cconf > self.conf) & (cids == PERSON_CLASS)
        if not np.any(mask):
            return []
        b = p[mask, :4]
        cf = cconf[mask]
        pw, ph = pad
        x1 = np.clip((b[:,0]-b[:,2]/2-pw)/ratio, 0, orig_shape[1])
        y1 = np.clip((b[:,1]-b[:,3]/2-ph)/ratio, 0, orig_shape[0])
        x2 = np.clip((b[:,0]+b[:,2]/2-pw)/ratio, 0, orig_shape[1])
        y2 = np.clip((b[:,1]+b[:,3]/2-ph)/ratio, 0, orig_shape[0])
        xyxy = np.stack([x1,y1,x2,y2], axis=1)
        keep = self._nms(xyxy, cf)
        return [{'bbox': xyxy[i].astype(np.int32), 'confidence': float(cf[i]), 'class_id': PERSON_CLASS} for i in keep]

    @staticmethod
    def _nms(boxes, scores, thr=IOU_THRESH):
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            ix1=np.maximum(x1[i],x1[order[1:]])
            iy1=np.maximum(y1[i],y1[order[1:]])
            ix2=np.minimum(x2[i],x2[order[1:]])
            iy2=np.minimum(y2[i],y2[order[1:]])
            inter=np.maximum(0,ix2-ix1)*np.maximum(0,iy2-iy1)
            iou=inter/(areas[i]+areas[order[1:]]-inter+1e-6)
            order=order[np.where(iou<=thr)[0]+1]
        return keep

def depth_dist(depth_frame, bbox, fshape):
    if depth_frame is None: 
        return "", None
    try:
        x1,y1,x2,y2 = bbox
        dh,dw = depth_frame.shape[:2]
        sx,sy = dw/fshape[1], dh/fshape[0]
        roi = depth_frame[max(0,int(y1*sy)):max(1,int(y2*sy)), max(0,int(x1*sx)):max(1,int(x2*sx))]
        v = roi[roi>0]
        if not len(v): 
            return "", None
        m = np.median(v)
        return f" {m/1000:.2f}m", m/1000.0
    except Exception:
        return "", None

def main():
    print_config_summary()
    using_astra = False
    if PYORBBECSDK_AVAILABLE and not FORCE_USB_CAMERA:
        try:
            camera = AstraCamera(fps=CAMERA_FPS).start()
            using_astra = True
        except Exception:
            pass
    if not using_astra:
        camera = ThreadedCamera(CAMERA_ID, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS).start()
    
    for _ in range(5):
        if using_astra:
            ft, *_ = camera.read()
        else:
            ft, _ = camera.read()
        if ft is not None:
            break
        time.sleep(0.3)
    else:
        camera.stop()
        return
    
    detector = ONNXDetector(MODEL_PATH, INPUT_SIZE, CONF_THRESH, IOU_THRESH)
    preprocessor = PreProcessor(INPUT_SIZE)
    tracker = ByteTracker(TRACK_THRESH, TRACK_BUFFER, MATCH_THRESH)
    decision_maker = VisualServoingDecisionMaker()
    
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    gc_counter = 0
    locked_track_id = None
    
    try:
        while True:
            if using_astra:
                frame, depth_frame, camera_fps = camera.read()
            else:
                frame, camera_fps = camera.read()
                depth_frame = None
            
            if frame is None:
                continue
            
            frame_count += 1
            img, ratio, pad = preprocessor.run(frame)
            detections = detector.run(img, frame.shape, ratio, pad)
            all_tracks = tracker.update(detections)
            
            current_target = None
            if locked_track_id is not None:
                for track in all_tracks:
                    if track['track_id'] == locked_track_id:
                        current_target = track
                        break
            
            if current_target is None and len(all_tracks) > 0:
                locked_track_id = all_tracks[0]['track_id']
                current_target = all_tracks[0]
            
            if current_target is not None:
                distance_str, distance_m = depth_dist(depth_frame, current_target['bbox'], frame.shape)
                if distance_m is not None:
                    distance_mm = distance_m * 1000.0
                    try:
                        action = decision_maker.process_target(current_target['bbox'], distance_mm)
                    except Exception as e:
                        decision_maker.stop()
            else:
                if locked_track_id is not None:
                    decision_maker.stop()
                    locked_track_id = None
            
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
                    
                    if tid == locked_track_id:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        if current_target is not None:
                            rd_str, rd = depth_dist(depth_frame, current_target['bbox'], frame.shape)
                            label = f"TARGET ID:{tid} ({score:.2f}){rd_str}"
                        else:
                            label = f"TARGET ID:{tid} ({score:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                        label = f"ID:{tid} ({score:.2f})"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                except Exception:
                    continue
            
            y_offset = 30
            cv2.putText(frame, f"Camera: {camera_fps:.1f}fps", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = frame_count / elapsed
            cv2.putText(frame, f"AI: {fps:.1f}fps", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            if current_target is not None:
                y_offset += 25
                cv2.putText(frame, f"Nav: {decision_maker.last_command}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
                cv2.putText(frame, f"Locked: ID-{locked_track_id}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(get_wait_ms()) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                dm_stats = decision_maker.get_statistics()
                print(f"Camera FPS: {camera_fps:.2f}, AI FPS: {fps:.2f}, Tracks: {len(all_tracks)}")
                print(f"Commands: F={dm_stats['move_forward_count']}, B={dm_stats['move_backward_count']}, L={dm_stats['turn_left_count']}, R={dm_stats['turn_right_count']}")
            
            gc_counter += 1
            if gc_counter >= GC_INTERVAL:
                gc.collect()
                gc_counter = 0
    
    except KeyboardInterrupt:
        pass
    
    finally:
        decision_maker.stop()
        camera.stop()
        cv2.destroyAllWindows()
        gc.collect()

if __name__ == "__main__":
    main()