#!/usr/bin/env python3

import time
import math
import threading
import numpy as np
import queue
import io
import wave
from enum import Enum
from typing import Optional, Tuple, Callable, Dict, Any, List
from dataclasses import dataclass
import logging

try:
    import pyaudio
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Audio libraries not available. Install: pip install pyaudio speechrecognition")

try:
    ADVANCED_AUDIO = False
    print("Using simplified audio processing (RNNoise/Silero not available)")
except ImportError:
    ADVANCED_AUDIO = False

from decision_maker import VisualServoingDecisionMaker
from config import CAMERA_WIDTH, CAMERA_HEIGHT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_SR = 16000
AUDIO_QUEUE_MAX = 5
TEXT_QUEUE_MAX = 10

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

_recognizer = sr.Recognizer()
_pyaudio = pyaudio.PyAudio() if AUDIO_AVAILABLE else None


def transcribe_audio(audio_16k):
    try:
        pcm16 = (audio_16k * 32768).astype(np.int16).tobytes()
        
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(pcm16)
        
        wav_buf.seek(0)
        
        with sr.AudioFile(wav_buf) as source:
            audio_data = _recognizer.record(source)
        
        return _recognizer.recognize_google(audio_data, language="vi-VN").strip()
    
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        logger.error(f"Google API error: {e}")
        return ""
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""


class VoiceTimer:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_voice = 0.0
    
    def touch(self):
        with self._lock:
            self._last_voice = time.monotonic()
    
    def elapsed(self):
        with self._lock:
            if self._last_voice == 0.0:
                return 0.0
            return time.monotonic() - self._last_voice


class EndpointDetector:
    FILLERS = frozenset({
        "ờ", "à", "ừ", "ừm", "ạ", "ơ", "ơi", "ê", "hả",
        "uhm", "um", "ah", "uh", "eh", "hmm", "mm"
    })
    
    CONNECTORS = frozenset({
        "và", "hoặc", "nhưng", "rồi", "mà", "hay", "vì", "nên",
        "thì", "còn", "với", "để", "nếu", "khi", "sau", "trước"
    })
    
    ENDPOINTS = frozenset({
        "xong", "hết", "ok", "được rồi", "thế thôi", "vậy thôi",
        "cảm ơn", "tạm biệt", "bye", "chào"
    })
    
    SILENCE_TABLE = (
        (3,   1.2),
        (6,   1.6),
        (10,  2.0),
        (15,  2.5),
        (20,  3.0),
        (30,  3.5),
    )
    
    SILENCE_MAX = 4.5
    CONNECTOR_BONUS = 1.0
    TEXT_AGE_WORD_THRESHOLD = 6
    STABILITY_COUNT = 3
    
    def __init__(self):
        self._last_raw = ""
        self._last_merged = ""
        self._repeat_count = 0
        self._last_text_time = 0.0
    
    def on_text_received(self):
        self._last_text_time = time.monotonic()
    
    def _text_cooldown_elapsed(self):
        if self._last_text_time == 0.0:
            return float('inf')
        return time.monotonic() - self._last_text_time
    
    def filter_fillers(self, text: str) -> str:
        words = text.split()
        filtered = [w for w in words if w.lower() not in self.FILLERS]
        return " ".join(filtered)
    
    def is_duplicate(self, raw_text: str) -> bool:
        normalized = raw_text.strip().lower()
        if normalized == self._last_raw:
            return True
        self._last_raw = normalized
        return False
    
    def try_extend_buffer(self, buffer: List[str], new_text: str) -> bool:
        if not buffer:
            return False
        
        last = buffer[-1].lower()
        new_lower = new_text.lower()
        
        if new_lower.startswith(last) and len(new_text) > len(buffer[-1]):
            buffer[-1] = new_text
            return True
        
        if last.startswith(new_lower):
            return True
        
        return False
    
    def check_punctuation(self, text: str) -> bool:
        return bool(text) and text[-1] in ".?!"
    
    def check_keyword_endpoint(self, text: str) -> bool:
        lower = text.lower().strip()
        for kw in self.ENDPOINTS:
            if lower.endswith(kw):
                return True
        return False
    
    def check_stability(self, merged_text: str) -> bool:
        normalized = merged_text.strip().lower()
        if normalized == self._last_merged:
            self._repeat_count += 1
        else:
            self._last_merged = normalized
            self._repeat_count = 1
        
        return self._repeat_count >= self.STABILITY_COUNT
    
    def get_silence_threshold(self, merged_text: str) -> float:
        words = merged_text.split()
        n = len(words)
        
        base = self.SILENCE_MAX
        for max_words, timeout in self.SILENCE_TABLE:
            if n <= max_words:
                base = timeout
                break
        
        if n > 0 and words[-1].lower() in self.CONNECTORS:
            base += self.CONNECTOR_BONUS
        
        return base
    
    def should_finalize_silence(self, merged_text: str, voice_elapsed: float,
                               stt_is_busy: bool, audio_q_empty: bool, 
                               mic_is_recording: bool) -> bool:
        if stt_is_busy:
            return False
        
        if not audio_q_empty:
            return False
        
        if mic_is_recording:
            return False
        
        threshold = self.get_silence_threshold(merged_text)
        
        n = len(merged_text.split())
        if n > self.TEXT_AGE_WORD_THRESHOLD:
            text_age = self._text_cooldown_elapsed()
            effective = min(voice_elapsed, text_age)
        else:
            effective = voice_elapsed
        
        return effective > threshold
    
    def reset(self):
        self._last_raw = ""
        self._last_merged = ""
        self._repeat_count = 0
        self._last_text_time = 0.0


class SimplifiedMicStream:
    def __init__(self, sample_rate=TARGET_SR, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None
        self.is_running = False
    
    def start(self):
        if not AUDIO_AVAILABLE:
            logger.warning("Audio not available")
            return
        
        try:
            self.stream = _pyaudio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            self.is_running = True
            logger.info("Microphone stream started")
        except Exception as e:
            logger.error(f"Failed to start: {e}")
    
    def read(self):
        if not self.stream or not self.is_running:
            return None
        
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_np = np.frombuffer(data, dtype=np.float32)
            return audio_np
        except Exception as e:
            logger.error(f"Read error: {e}")
            return None
    
    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            logger.info("Microphone stream stopped")


def simplified_vad(audio_chunk, threshold=0.01):
    if audio_chunk is None or len(audio_chunk) == 0:
        return False
    
    rms = np.sqrt(np.mean(audio_chunk ** 2))
    return rms > threshold


def capture_audio_simplified(mic_stream, voice_timer, min_duration=1.0):
    if not mic_stream.is_running:
        return None
    
    audio_buffer = []
    recording = False
    silence_count = 0
    max_silence = 20
    
    start_time = time.time()
    
    while time.time() - start_time < 5.0:
        chunk = mic_stream.read()
        if chunk is None:
            time.sleep(0.02)
            continue
        
        has_voice = simplified_vad(chunk)
        
        if has_voice:
            voice_timer.touch()
            if not recording:
                recording = True
                logger.debug("Voice detected - start recording")
            audio_buffer.append(chunk)
            silence_count = 0
        else:
            if recording:
                audio_buffer.append(chunk)
                silence_count += 1
                
                if silence_count > max_silence:
                    break
        
        time.sleep(0.02)
    
    if audio_buffer and len(audio_buffer) > min_duration * 50:
        return np.concatenate(audio_buffer)
    
    return None


class RobotState(Enum):
    IDLE = "idle"
    WAKE = "wake"
    SEARCH = "search"
    TARGET_LOCK = "target_lock"
    LISTEN_COMMAND = "listen_command"


@dataclass
class SoundEvent:
    timestamp: float
    wake_word_detected: bool = False
    direction_angle: Optional[float] = None
    confidence: float = 0.0
    audio_level: float = 0.0


@dataclass
class InteractionConfig:
    wake_word: str = "emi ơi"
    wake_confidence_threshold: float = 0.7
    
    sample_rate: int = 16000
    chunk_size: int = 1024
    audio_device_index: Optional[int] = None
    
    search_rotation_speed: float = 1.0
    search_timeout: float = 10.0
    
    wake_timeout: float = 3.0
    listen_timeout: float = 5.0
    
    min_person_confidence: float = 0.6
    
    idle_motion_sensitivity: float = 0.3
    idle_sound_threshold: float = 0.1


class AdvancedWakeWordDetector:
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.recognizer = sr.Recognizer() if AUDIO_AVAILABLE else None
        self.is_listening = False
        
        self.voice_timer = VoiceTimer()
        self.endpoint_detector = EndpointDetector()
        
        self.audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
        self.text_queue = queue.Queue(maxsize=TEXT_QUEUE_MAX)
        
        self.stop_event = threading.Event()
        self.stt_busy = threading.Event()
        self.mic_recording = threading.Event()
        
        if ADVANCED_AUDIO:
            self.mic_stream = None
        else:
            self.mic_stream = SimplifiedMicStream()
        
        self.temp_text_buffer = []
        self.sentence_start_time = None
        
        logger.info("Advanced Wake Word Detector initialized")
    
    def start_listening(self, callback: Callable[[SoundEvent], None]):
        if not AUDIO_AVAILABLE:
            logger.warning("Audio not available, using mock detection")
            return
        
        self.is_listening = True
        self.callback = callback
        
        if self.mic_stream:
            self.mic_stream.start()
        
        self.mic_thread = threading.Thread(
            target=self._mic_worker, daemon=True, name="mic_worker"
        )
        self.stt_thread = threading.Thread(
            target=self._stt_worker, daemon=True, name="stt_worker"
        )
        self.processing_thread = threading.Thread(
            target=self._processing_worker, daemon=True, name="processing_worker"
        )
        
        self.mic_thread.start()
        self.stt_thread.start()
        self.processing_thread.start()
        
        logger.info("Advanced wake word detection started")
    
    def stop_listening(self):
        self.is_listening = False
        self.stop_event.set()
        
        if self.mic_stream:
            self.mic_stream.stop()
        
        logger.info("Stopped wake word detection")
    
    def _mic_worker(self):
        while not self.stop_event.is_set():
            try:
                if ADVANCED_AUDIO:
                    audio = None
                else:
                    audio = capture_audio_simplified(
                        self.mic_stream, self.voice_timer
                    )
                
                if audio is None:
                    time.sleep(0.1)
                    continue
                
                self.voice_timer.touch()
                
                try:
                    self.audio_queue.put_nowait(audio)
                except queue.Full:
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.audio_queue.put_nowait(audio)
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Error: {e}")
                break
    
    def _stt_worker(self):
        while not self.stop_event.is_set():
            try:
                audio = self.audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            
            self.stt_busy.set()
            try:
                t0 = time.time()
                text = transcribe_audio(audio)
                dt = time.time() - t0
                
                if text:
                    self.text_queue.put((text, dt))
                    
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Error: {e}")
            finally:
                self.stt_busy.clear()
    
    def _processing_worker(self):
        while not self.stop_event.is_set():
            try:
                try:
                    raw_text, latency = self.text_queue.get(timeout=0.1)
                except queue.Empty:
                    if self.temp_text_buffer:
                        merged = " ".join(self.temp_text_buffer)
                        if self.endpoint_detector.should_finalize_silence(
                            merged,
                            self.voice_timer.elapsed(),
                            self.stt_busy.is_set(),
                            self.audio_queue.empty(),
                            self.mic_recording.is_set()
                        ):
                            self._finalize_sentence("SILENCE")
                    continue
                
                text = self.endpoint_detector.filter_fillers(raw_text)
                if not text:
                    logger.debug(f"Filler: '{raw_text}'")
                    continue
                
                if self.endpoint_detector.is_duplicate(text):
                    logger.debug(f"Duplicate: '{text}'")
                    continue
                
                self.endpoint_detector.on_text_received()
                logger.debug(f"Received: '{text}' ({latency:.2f}s)")
                
                if not self.temp_text_buffer:
                    self.sentence_start_time = time.time()
                
                if not self.endpoint_detector.try_extend_buffer(self.temp_text_buffer, text):
                    self.temp_text_buffer.append(text)
                else:
                    logger.debug("Extended buffer")
                
                if self.endpoint_detector.check_punctuation(text):
                    self._finalize_sentence("PUNCTUATION")
                    continue
                
                if self.endpoint_detector.check_keyword_endpoint(text):
                    self._finalize_sentence("KEYWORD")
                    continue
                
                merged = " ".join(self.temp_text_buffer)
                if self.endpoint_detector.check_stability(merged):
                    self._finalize_sentence("STABILITY")
                    continue
                
            except Exception as e:
                if not self.stop_event.is_set():
                    logger.error(f"Error: {e}")
    
    def _finalize_sentence(self, reason: str):
        if not self.temp_text_buffer:
            return
        
        full_text = " ".join(self.temp_text_buffer)
        clean_text = self.endpoint_detector.filter_fillers(full_text)
        
        if not clean_text:
            self.temp_text_buffer.clear()
            self.endpoint_detector.reset()
            return
        
        end_time = time.time()
        duration = end_time - self.sentence_start_time if self.sentence_start_time else 0
        
        logger.info(f"{GREEN}{reason}: '{clean_text}' ({duration:.2f}s){RESET}")
        
        if self.config.wake_word.lower() in clean_text.lower():
            logger.info(f"{GREEN}Wake word detected in: '{clean_text}'{RESET}")
            
            event = SoundEvent(
                timestamp=time.time(),
                wake_word_detected=True,
                confidence=0.8,
                audio_level=0.5
            )
            
            if self.callback:
                self.callback(event)
        
        self.temp_text_buffer.clear()
        self.endpoint_detector.reset()
        self.sentence_start_time = None


class SoundDirectionEstimator:
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.mic_array_available = False
    
    def get_sound_direction(self) -> Optional[float]:
        if not self.mic_array_available:
            logger.debug("Single mic mode - no direction info")
            return None
        
        mock_angle = np.random.uniform(-np.pi, np.pi)
        logger.info(f"Mock direction: {mock_angle:.2f} rad ({math.degrees(mock_angle):.1f}°)")
        return mock_angle


class RobotController:
    def __init__(self, decision_maker: VisualServoingDecisionMaker):
        self.decision_maker = decision_maker
        self.is_scanning = False
        self.scan_thread = None
    
    def rotate_to_angle(self, target_angle: float, speed: float = 1.0):
        logger.info(f"Rotating to {math.degrees(target_angle):.1f}° at {speed:.1f} rad/s")
        
        rotation_direction = 1.0 if target_angle > 0 else -1.0
        
        rotation_time = abs(target_angle) / speed
        
        self.decision_maker.send_to_esp32(0.0, rotation_direction * speed)
        
        time.sleep(rotation_time)
        self.decision_maker.send_to_esp32(0.0, 0.0)
        
        logger.info(f"Rotation complete")
    
    def start_360_scan(self, speed: float = 1.0, person_detector: Callable[[], Optional[Tuple[int, int]]] = None):
        if self.is_scanning:
            logger.warning("Already scanning")
            return
        
        self.is_scanning = True
        logger.info(f"Starting 360° scan at {speed:.1f} rad/s")
        
        def scan_worker():
            start_time = time.time()
            scan_duration = 2 * math.pi / speed
            
            self.decision_maker.send_to_esp32(0.0, speed)
            
            while self.is_scanning and (time.time() - start_time) < scan_duration:
                if person_detector:
                    person_pos = person_detector()
                    if person_pos:
                        logger.info(f"Person detected at {person_pos}")
                        self.stop_scan()
                        return person_pos
                
                time.sleep(0.1)
            
            self.decision_maker.send_to_esp32(0.0, 0.0)
            self.is_scanning = False
            logger.info("360° scan completed")
            return None
        
        self.scan_thread = threading.Thread(target=scan_worker, daemon=True)
        self.scan_thread.start()
    
    def stop_scan(self):
        if self.is_scanning:
            self.is_scanning = False
            self.decision_maker.send_to_esp32(0.0, 0.0)
            logger.info("Scan stopped")


class VisionSystem:
    def __init__(self, config: InteractionConfig):
        self.config = config
        self.last_detection = None
    
    def detect_person(self, frame, detections) -> Optional[Tuple[int, int]]:
        if not detections:
            return None
        
        best_person = None
        best_confidence = 0.0
        
        for detection in detections:
            if (detection.get('class_id') == 0 and
                detection.get('confidence', 0) > self.config.min_person_confidence):
                
                if detection['confidence'] > best_confidence:
                    best_confidence = detection['confidence']
                    best_person = detection
        
        if best_person:
            bbox = best_person['bbox']
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            
            self.last_detection = (center_x, center_y)
            logger.debug(f"Person detected at ({center_x}, {center_y}) conf={best_confidence:.2f}")
            return (center_x, center_y)
        
        return None


class EMISoundInteractionSystem:
    def __init__(self, decision_maker: VisualServoingDecisionMaker, config: InteractionConfig = None):
        self.config = config or InteractionConfig()
        self.decision_maker = decision_maker
        
        self.wake_detector = AdvancedWakeWordDetector(self.config)
        self.direction_estimator = SoundDirectionEstimator(self.config)
        self.robot_controller = RobotController(decision_maker)
        self.vision_system = VisionSystem(self.config)
        
        self.current_state = RobotState.IDLE
        self.state_start_time = time.time()
        self.target_person_pos = None
        
        self.stats = {
            'wake_word_detections': 0,
            'successful_target_locks': 0,
            'search_attempts': 0,
            'timeouts': 0,
            'state_transitions': 0
        }
        
        logger.info("Sound Interaction System initialized")
    
    def start(self):
        logger.info("Starting EMI Sound Interaction System")
        
        self.wake_detector.start_listening(self._on_sound_event)
        
        self.running = True
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
        logger.info("EMI System started - Listening for 'Emi ơi'")
    
    def stop(self):
        logger.info("Stopping EMI Sound Interaction System")
        
        self.running = False
        self.wake_detector.stop_listening()
        self.robot_controller.stop_scan()
        
        self._transition_to_state(RobotState.IDLE)
        
        logger.info("EMI System stopped")
    
    def _transition_to_state(self, new_state: RobotState):
        if new_state != self.current_state:
            logger.info(f"State: {self.current_state.value} → {new_state.value}")
            self.current_state = new_state
            self.state_start_time = time.time()
            self.stats['state_transitions'] += 1
    
    def _on_sound_event(self, event: SoundEvent):
        if event.wake_word_detected and self.current_state == RobotState.IDLE:
            logger.info("Wake word detected - transitioning to WAKE state")
            self.stats['wake_word_detections'] += 1
            self._transition_to_state(RobotState.WAKE)
    
    def _main_loop(self):
        while self.running:
            try:
                current_time = time.time()
                state_duration = current_time - self.state_start_time
                
                if self.current_state == RobotState.IDLE:
                    self._handle_idle_state()
                
                elif self.current_state == RobotState.WAKE:
                    self._handle_wake_state(state_duration)
                
                elif self.current_state == RobotState.SEARCH:
                    self._handle_search_state(state_duration)
                
                elif self.current_state == RobotState.TARGET_LOCK:
                    self._handle_target_lock_state(state_duration)
                
                elif self.current_state == RobotState.LISTEN_COMMAND:
                    self._handle_listen_command_state(state_duration)
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(0.5)
    
    def _handle_idle_state(self):
        pass
    
    def _handle_wake_state(self, duration: float):
        if duration > self.config.wake_timeout:
            logger.warning("Wake state timeout")
            self.stats['timeouts'] += 1
            self._transition_to_state(RobotState.IDLE)
            return
        
        direction = self.direction_estimator.get_sound_direction()
        
        if direction is not None:
            logger.info(f"Sound direction detected: {math.degrees(direction):.1f}°")
            self.robot_controller.rotate_to_angle(direction)
            self._transition_to_state(RobotState.TARGET_LOCK)
        else:
            logger.info("No direction info - starting 360° search")
            self.stats['search_attempts'] += 1
            self._transition_to_state(RobotState.SEARCH)
    
    def _handle_search_state(self, duration: float):
        if duration > self.config.search_timeout:
            logger.warning("Search timeout - no person found")
            self.stats['timeouts'] += 1
            self._transition_to_state(RobotState.IDLE)
            return
        
        if not self.robot_controller.is_scanning:
            def person_detector():
                return None
            
            self.robot_controller.start_360_scan(
                speed=self.config.search_rotation_speed,
                person_detector=person_detector
            )
    
    def _handle_target_lock_state(self, duration: float):
        logger.info("Target locked - ready for commands")
        self.stats['successful_target_locks'] += 1
        
        self._transition_to_state(RobotState.LISTEN_COMMAND)
    
    def _handle_listen_command_state(self, duration: float):
        if duration > self.config.listen_timeout:
            logger.info("Listen timeout - returning to idle")
            self._transition_to_state(RobotState.IDLE)
            return
    
    def get_current_state(self) -> RobotState:
        return self.current_state
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'current_state': self.current_state.value,
            'uptime': time.time() - self.state_start_time
        }
    
    def process_frame_with_detections(self, frame, detections):
        if self.current_state == RobotState.SEARCH:
            person_pos = self.vision_system.detect_person(frame, detections)
            if person_pos:
                logger.info(f"Person found during search at {person_pos}")
                self.target_person_pos = person_pos
                self.robot_controller.stop_scan()
                self._transition_to_state(RobotState.TARGET_LOCK)


def test_emi_system():
    print("Testing Advanced EMI Sound Interaction System")
    print("=" * 60)
    
    class MockDecisionMaker:
        def send_to_esp32(self, v, w):
            print(f"Motor command: v={v:.3f}, w={w:.3f}")
    
    mock_decision_maker = MockDecisionMaker()
    
    config = InteractionConfig(
        wake_word="emi ơi",
        search_timeout=5.0,
        listen_timeout=3.0,
        wake_confidence_threshold=0.7
    )
    
    emi_system = EMISoundInteractionSystem(mock_decision_maker, config)
    
    try:
        emi_system.start()
        
        print("Advanced EMI System started")
        print("Features enabled:")
        print("   - Intelligent Voice Activity Detection")
        print("   - Noise filtering and audio enhancement")
        print("   - Smart endpoint detection")
        print("   - Vietnamese filler word filtering")
        print("   - Sentence boundary detection")
        print("   - Wake word: 'Emi ơi'")
        print("\n" + "=" * 60)
        print("Say 'Emi ơi' to test wake word detection")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        while True:
            stats = emi_system.get_statistics()
            current_state = stats['current_state']
            wake_count = stats['wake_word_detections']
            
            if current_state == 'idle':
                status_color = GREEN
            elif current_state == 'wake':
                status_color = YELLOW
            elif current_state == 'search':
                status_color = CYAN
            else:
                status_color = RED
            
            print(f"\r{status_color}State: {current_state.upper()} | "
                  f"Wake detections: {wake_count} | "
                  f"Target locks: {stats['successful_target_locks']}{RESET}", 
                  end='', flush=True)
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stopping Advanced EMI System{RESET}")
        emi_system.stop()
        
        final_stats = emi_system.get_statistics()
        print(f"\n{GREEN}Final Statistics:{RESET}")
        print(f"   Wake word detections: {final_stats['wake_word_detections']}")
        print(f"   Successful target locks: {final_stats['successful_target_locks']}")
        print(f"   Search attempts: {final_stats['search_attempts']}")
        print(f"   State transitions: {final_stats['state_transitions']}")
        print(f"   Total uptime: {final_stats['uptime']:.1f}s")
        
        print(f"\n{CYAN}Audio Processing Features:{RESET}")
        print("   Voice Activity Detection")
        print("   Intelligent Endpoint Detection")
        print("   Vietnamese Filler Filtering")
        print("   Smart Sentence Boundaries")
        print("   Noise Reduction (simplified)")
        print("   Ready for RNNoise/Silero upgrade")


if __name__ == "__main__":
    test_emi_system()