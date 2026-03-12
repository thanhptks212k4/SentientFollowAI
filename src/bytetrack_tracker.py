"""
ByteTrack Implementation for Person Tracking
- Lightweight and fast (suitable for edge devices)
- Robust to occlusions
- Maintains consistent IDs
- No need for ReID model (unlike DeepSORT)
"""

import numpy as np
from collections import OrderedDict


class KalmanFilter:
    """
    Simple Kalman Filter for bounding box tracking
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va, vr]
    """
    
    def __init__(self):
        # State transition matrix
        self.dt = 1.0
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = self.dt
        
        # Measurement matrix
        self.H = np.eye(4, 8)
        
        # Process noise
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01
        
        # Measurement noise
        self.R = np.eye(4) * 10
        
        # State covariance
        self.P = np.eye(8) * 1000
        
        # State
        self.x = np.zeros((8, 1))
    
    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].flatten()
    
    def update(self, measurement):
        """Update state with measurement"""
        y = measurement.reshape(4, 1) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
    
    def init_state(self, measurement):
        """Initialize state with first measurement"""
        self.x[:4] = measurement.reshape(4, 1)
        self.x[4:] = 0


class Track:
    """
    Single track object
    Represents one person being tracked
    """
    
    _id_counter = 0
    
    def __init__(self, bbox, score):
        """
        Args:
            bbox: [x1, y1, x2, y2]
            score: confidence score
        """
        self.track_id = Track._id_counter
        Track._id_counter += 1
        
        self.bbox = bbox
        self.score = score
        
        # Kalman filter
        self.kf = KalmanFilter()
        self.kf.init_state(self._bbox_to_state(bbox))
        
        # Track state
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        # Track status
        self.state = 'tentative'  # tentative, confirmed, deleted
    
    def predict(self):
        """Predict next position"""
        self.age += 1
        self.time_since_update += 1
        
        state = self.kf.predict()
        self.bbox = self._state_to_bbox(state)
    
    def update(self, bbox, score):
        """Update track with new detection"""
        self.bbox = bbox
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        
        state = self._bbox_to_state(bbox)
        self.kf.update(state)
        
        # Confirm track after 3 consecutive hits
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'
    
    def mark_missed(self):
        """Mark track as missed"""
        if self.state == 'tentative':
            self.state = 'deleted'
        elif self.time_since_update > 30:  # Lost for 30 frames
            self.state = 'deleted'
    
    @staticmethod
    def _bbox_to_state(bbox):
        """Convert bbox [x1, y1, x2, y2] to state [cx, cy, area, ratio]"""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        ratio = (x2 - x1) / max(y2 - y1, 1e-6)
        return np.array([cx, cy, area, ratio])
    
    @staticmethod
    def _state_to_bbox(state):
        """Convert state [cx, cy, area, ratio] to bbox [x1, y1, x2, y2]"""
        cx, cy, area, ratio = state
        
        # Ensure valid values to prevent overflow
        area = max(area, 1.0)  # Minimum area of 1 pixel
        ratio = max(ratio, 1e-6)  # Minimum ratio
        
        h = np.sqrt(area / ratio)
        w = ratio * h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        # Ensure finite values
        bbox = np.array([x1, y1, x2, y2])
        bbox = np.nan_to_num(bbox, nan=0.0, posinf=1000.0, neginf=0.0)
        
        return bbox


class ByteTracker:
    """
    ByteTrack: Multi-Object Tracking by Associating Every Detection Box
    
    Key features:
    - Uses both high and low confidence detections
    - Robust to occlusions
    - Fast and lightweight
    """
    
    def __init__(self, 
                 track_thresh=0.5,      # High confidence threshold
                 track_buffer=30,       # Frames to keep lost tracks
                 match_thresh=0.8,      # IoU threshold for matching
                 min_box_area=100):     # Minimum box area
        
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area
        
        self.tracked_tracks = []  # Active tracks
        self.lost_tracks = []     # Lost tracks
        self.removed_tracks = []  # Removed tracks
        
        self.frame_id = 0
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of dicts with 'bbox' [x1,y1,x2,y2] and 'confidence'
        
        Returns:
            List of active tracks with track_id
        """
        self.frame_id += 1
        
        # Separate high and low confidence detections
        high_dets = []
        low_dets = []
        
        for det in detections:
            bbox = det['bbox']
            score = det['confidence']
            
            # Filter small boxes
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area < self.min_box_area:
                continue
            
            if score >= self.track_thresh:
                high_dets.append((bbox, score))
            else:
                low_dets.append((bbox, score))
        
        # Predict current positions of tracks
        for track in self.tracked_tracks:
            track.predict()
        
        # First association: high confidence detections with tracked tracks
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self.tracked_tracks, high_dets, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            track = self.tracked_tracks[track_idx]
            bbox, score = high_dets[det_idx]
            track.update(bbox, score)
        
        # Second association: low confidence detections with unmatched tracks
        unmatched_tracks_objs = [self.tracked_tracks[i] for i in unmatched_tracks]
        matched_low, unmatched_tracks_low, _ = self._associate(
            unmatched_tracks_objs, low_dets, 0.5
        )
        
        # Update tracks matched with low confidence detections
        for track_idx, det_idx in matched_low:
            track = unmatched_tracks_objs[track_idx]
            bbox, score = low_dets[det_idx]
            track.update(bbox, score)
        
        # Mark unmatched tracks as lost
        for track_idx in unmatched_tracks_low:
            track = unmatched_tracks_objs[track_idx]
            track.mark_missed()
        
        # Create new tracks for unmatched high confidence detections
        for det_idx in unmatched_dets:
            bbox, score = high_dets[det_idx]
            new_track = Track(bbox, score)
            self.tracked_tracks.append(new_track)
        
        # Remove deleted tracks
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state != 'deleted']
        
        # Return confirmed tracks
        output_tracks = []
        for track in self.tracked_tracks:
            if track.state == 'confirmed':
                # Ensure bbox is valid before adding to output
                bbox = track.bbox
                if np.any(np.isnan(bbox)) or np.any(np.isinf(bbox)):
                    continue  # Skip invalid bboxes
                
                # Ensure bbox coordinates are reasonable
                bbox_int = np.clip(bbox, 0, 10000).astype(int)
                
                output_tracks.append({
                    'track_id': track.track_id,
                    'bbox': bbox_int,
                    'score': track.score
                })
        
        return output_tracks
    
    def _associate(self, tracks, detections, iou_threshold):
        """
        Associate tracks with detections using IoU
        
        Returns:
            matched: List of (track_idx, det_idx) pairs
            unmatched_tracks: List of track indices
            unmatched_dets: List of detection indices
        """
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, (bbox, _) in enumerate(detections):
                iou_matrix[i, j] = self._iou(track.bbox, bbox)
        
        # Greedy matching
        matched = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        while True:
            if len(unmatched_tracks) == 0 or len(unmatched_dets) == 0:
                break
            
            # Find best match
            max_iou = 0
            max_i, max_j = -1, -1
            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_i, max_j = i, j
            
            if max_iou < iou_threshold:
                break
            
            matched.append((max_i, max_j))
            unmatched_tracks.remove(max_i)
            unmatched_dets.remove(max_j)
        
        return matched, unmatched_tracks, unmatched_dets
    
    @staticmethod
    def _iou(bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / max(union_area, 1e-6)
    
    def reset(self):
        """Reset tracker"""
        Track._id_counter = 0
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0


class TargetLocker:
    """
    Target Locking System
    - Locks onto first detected person
    - Ignores all other people
    - Maintains target even with occlusions
    """
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.locked_target_id = None
        self.is_locked = False
        self.frames_without_target = 0
        self.max_lost_frames = 60  # Reset after 60 frames without target
    
    def lock_target(self, detections):
        """
        Lock onto first detected person
        
        Args:
            detections: List of detections from YOLO
        
        Returns:
            True if target locked successfully
        """
        if len(detections) == 0:
            return False
        
        # Update tracker to get track IDs
        tracks = self.tracker.update(detections)
        
        if len(tracks) == 0:
            return False
        
        # Lock onto first track (highest confidence)
        tracks_sorted = sorted(tracks, key=lambda x: x['score'], reverse=True)
        self.locked_target_id = tracks_sorted[0]['track_id']
        self.is_locked = True
        self.frames_without_target = 0
        
        print(f"[TargetLocker] Locked onto target ID: {self.locked_target_id}")
        return True
    
    def update(self, detections):
        """
        Update tracker and return only locked target
        
        Args:
            detections: List of detections from YOLO
        
        Returns:
            Target dict with 'track_id', 'bbox', 'score' or None
        """
        if not self.is_locked:
            return None
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Find locked target
        target = None
        for track in tracks:
            if track['track_id'] == self.locked_target_id:
                target = track
                self.frames_without_target = 0
                break
        
        # Handle lost target
        if target is None:
            self.frames_without_target += 1
            
            if self.frames_without_target > self.max_lost_frames:
                print(f"[TargetLocker] Target lost for {self.max_lost_frames} frames. Unlocking.")
                self.unlock_target()
        
        return target
    
    def unlock_target(self):
        """Unlock current target"""
        self.locked_target_id = None
        self.is_locked = False
        self.frames_without_target = 0
        self.tracker.reset()
        print("[TargetLocker] Target unlocked")
    
    def get_status(self):
        """Get current status"""
        return {
            'is_locked': self.is_locked,
            'target_id': self.locked_target_id,
            'frames_without_target': self.frames_without_target
        }
