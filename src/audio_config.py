#!/usr/bin/env python3

from typing import Final

TARGET_SAMPLE_RATE: Final[int] = 16000
AUDIO_CHUNK_SIZE: Final[int] = 1024
AUDIO_CHANNELS: Final[int] = 1
AUDIO_FORMAT = "float32"

AUDIO_QUEUE_MAX: Final[int] = 5
TEXT_QUEUE_MAX: Final[int] = 10

VAD_ENERGY_THRESHOLD: Final[float] = 0.01
VAD_MIN_DURATION: Final[float] = 0.3
VAD_MAX_DURATION: Final[float] = 5.0
VAD_SILENCE_FRAMES: Final[int] = 20

SILERO_THRESHOLD: Final[float] = 0.5
SILERO_MIN_SPEECH_DURATION: Final[int] = 250
SILERO_MIN_SILENCE_DURATION: Final[int] = 100

DEFAULT_WAKE_WORD: Final[str] = "emi ơi"
WAKE_CONFIDENCE_THRESHOLD: Final[float] = 0.7
WAKE_WORD_TIMEOUT: Final[float] = 3.0

STT_LANGUAGE: Final[str] = "vi-VN"
STT_TIMEOUT: Final[float] = 5.0
STT_PHRASE_TIME_LIMIT: Final[float] = 10.0

VIETNAMESE_FILLERS = frozenset({
    "ờ", "à", "ừ", "ừm", "ạ", "ơ", "ơi", "ê", "hả", "hử", "hử",
    "uhm", "um", "ah", "uh", "eh", "hmm", "mm", "er", "erm"
})

VIETNAMESE_CONNECTORS = frozenset({
    "và", "hoặc", "nhưng", "rồi", "mà", "hay", "vì", "nên", "do",
    "thì", "còn", "với", "để", "nếu", "khi", "sau", "trước", "trong",
    "ngoài", "bên", "giữa", "theo", "như", "tại", "về", "từ", "đến"
})

VIETNAMESE_ENDPOINTS = frozenset({
    "xong", "hết", "ok", "được rồi", "thế thôi", "vậy thôi", "thôi",
    "cảm ơn", "tạm biệt", "bye", "chào", "hẹn gặp lại", "see you"
})

SILENCE_TIMEOUT_TABLE = (
    (1,   0.8),
    (3,   1.2),
    (6,   1.6),
    (10,  2.0),
    (15,  2.5),
    (20,  3.0),
    (30,  3.5),
)

SILENCE_MAX_TIMEOUT: Final[float] = 4.5
CONNECTOR_BONUS_TIME: Final[float] = 1.0
TEXT_AGE_THRESHOLD: Final[int] = 6
STABILITY_REPEAT_COUNT: Final[int] = 3

RNNOISE_ENABLED: Final[bool] = True
RNNOISE_FRAME_SIZE: Final[int] = 480

NOISE_GATE_THRESHOLD: Final[float] = 0.005
NOISE_GATE_RATIO: Final[float] = 0.1

SINGLE_MIC_MODE: Final[bool] = True
MIC_ARRAY_ENABLED: Final[bool] = False
MIC_ARRAY_COUNT: Final[int] = 4
MIC_ARRAY_SPACING: Final[float] = 0.05

TDOA_SOUND_SPEED: Final[float] = 343.0
DIRECTION_SMOOTHING: Final[float] = 0.3

AUDIO_THREAD_PRIORITY: Final[int] = 1
STT_THREAD_PRIORITY: Final[int] = 0

AUDIO_BUFFER_DURATION: Final[float] = 0.1
TEXT_BUFFER_MAX_AGE: Final[float] = 10.0

ENABLE_AUDIO_OPTIMIZATION: Final[bool] = True
MAX_CONCURRENT_STT: Final[int] = 1

DEBUG_AUDIO: Final[bool] = False
DEBUG_VAD: Final[bool] = False
DEBUG_ENDPOINT: Final[bool] = False
DEBUG_STT: Final[bool] = False

SAVE_AUDIO_FILES: Final[bool] = False
AUDIO_DEBUG_PATH: Final[str] = "/tmp/emi_audio"

CONSOLE_COLORS = {
    'GREEN': "\033[92m",
    'YELLOW': "\033[93m", 
    'CYAN': "\033[96m",
    'RED': "\033[91m",
    'BLUE': "\033[94m",
    'MAGENTA': "\033[95m",
    'WHITE': "\033[97m",
    'RESET': "\033[0m"
}

def validate_audio_config() -> bool:
    errors = []
    
    if TARGET_SAMPLE_RATE <= 0:
        errors.append("TARGET_SAMPLE_RATE must be positive")
    if AUDIO_CHUNK_SIZE <= 0:
        errors.append("AUDIO_CHUNK_SIZE must be positive")
    if VAD_ENERGY_THRESHOLD <= 0:
        errors.append("VAD_ENERGY_THRESHOLD must be positive")
    if WAKE_CONFIDENCE_THRESHOLD < 0 or WAKE_CONFIDENCE_THRESHOLD > 1:
        errors.append("WAKE_CONFIDENCE_THRESHOLD must be between 0 and 1")
    if SILENCE_MAX_TIMEOUT <= 0:
        errors.append("SILENCE_MAX_TIMEOUT must be positive")
    
    if errors:
        print("Audio configuration validation errors:")
        for error in errors:
            print(f"   - {error}")
        return False
    return True

def print_audio_config_summary() -> None:
    print("=== EMI Audio Configuration ===")
    print(f"Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"Chunk Size: {AUDIO_CHUNK_SIZE}")
    print(f"Wake Word: '{DEFAULT_WAKE_WORD}'")
    print(f"VAD Threshold: {VAD_ENERGY_THRESHOLD}")
    print(f"STT Language: {STT_LANGUAGE}")
    print(f"Single Mic Mode: {SINGLE_MIC_MODE}")
    print(f"RNNoise: {'Enabled' if RNNOISE_ENABLED else 'Disabled'}")
    print("===============================")

if __name__ == "__main__":
    if validate_audio_config():
        print_audio_config_summary()
        print("Audio configuration is valid")
    else:
        print("Audio configuration validation failed")
        exit(1)
else:
    if not validate_audio_config():
        raise ValueError("Invalid audio configuration parameters")