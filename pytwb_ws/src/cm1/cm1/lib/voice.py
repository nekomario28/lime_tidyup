"""Voice input, transcription, and navigation-command parsing."""

from array import array
import importlib
import math
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time

from .locations import location_pose

# 音声認識の共通設定
LANGUAGE = "ja"
MODEL = "base"
DEVICE_INDEX = -1
LISTEN_TIMEOUT = 10
PHRASE_TIME_LIMIT = 6
WHISPER_PROMPT = (
    "これはロボットへの短い日本語音声命令です。"
    "例: Aに行って。Bへ移動。Cに向かって。"
)

# 音声から移動先を判定するルール
_GO_KEYWORDS = re.compile(r"行|いって|移動|向か")
_DESTINATION_RULES = (
    (re.compile(r"[Aa]|えー|エー|えい"), "A"),
    (re.compile(r"[Bb]|びー|ビー|びい|ピー|ぴー|ぴい|ピイ|ヴィ|ヴィー"), "B"),
    (re.compile(r"[Cc]|しー|シー|しい|ちー|チー"), "C"),
)

# Whisperモデルの実行状態
_whisper_model = None
_whisper_model_lock = threading.Lock()


# エラー・中断などの共通処理
def _report_error(stage, message):
    print(f"[voice][{stage}] {message}")


def _is_cancelled(cancel_event):
    return cancel_event is not None and cancel_event.is_set()


def _run_cancellable(worker, cancel_event, poll_interval=0.1):
    """Run blocking work while allowing its caller to discard a late result."""
    if _is_cancelled(cancel_event):
        return None

    result_queue = queue.Queue(maxsize=1)

    def run_worker():
        try:
            result_queue.put((True, worker()))
        except Exception as exc:
            result_queue.put((False, exc))

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()
    while worker_thread.is_alive():
        worker_thread.join(poll_interval)
        if _is_cancelled(cancel_event):
            return None

    succeeded, result = result_queue.get()
    if not succeeded:
        raise result
    return result


def _device_index(value=DEVICE_INDEX):
    return None if int(value) < 0 else int(value)


# マイクの確認・診断
def pcm16_rms(data):
    """Calculate RMS volume from native-endian signed 16-bit PCM bytes."""
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0
    mean_square = sum(sample * sample for sample in samples) / len(samples)
    return int(math.sqrt(mean_square))


def print_microphones():
    """Print available input devices."""
    audio = None
    try:
        pyaudio = importlib.import_module("pyaudio")
        audio = pyaudio.PyAudio()
        for index in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                print(f"[{index}] {info['name']}")
        return True
    except Exception as exc:
        _report_error("microphone", exc)
        return False
    finally:
        if audio is not None:
            audio.terminate()


def test_microphone(seconds=3, device_index=DEVICE_INDEX):
    """Measure input volume for a few seconds."""
    audio = stream = None
    try:
        pyaudio = importlib.import_module("pyaudio")
        audio = pyaudio.PyAudio()
        index = _device_index(device_index)
        info = (
            audio.get_default_input_device_info()
            if index is None
            else audio.get_device_info_by_index(index)
        )
        rate = int(info["defaultSampleRate"])
        print(f"[voice] testing: {info['name']}")
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=rate,
            input=True,
            input_device_index=index,
            frames_per_buffer=1024,
        )
        end_time = time.time() + float(seconds)
        max_rms = 0
        while time.time() < end_time:
            data = stream.read(1024, exception_on_overflow=False)
            max_rms = max(max_rms, pcm16_rms(data))
        print(f"[voice] max_rms={max_rms}")
        return max_rms > 0
    except Exception as exc:
        _report_error("microphone-test", exc)
        return False
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if audio is not None:
            audio.terminate()


# 文字の読み上げ
def speak(text):
    """Speak text through espeak-ng."""
    if not text:
        return True
    try:
        subprocess.run(
            ["espeak-ng", "-v", LANGUAGE, "-s", "150", text],
            capture_output=True,
        )
    except FileNotFoundError:
        _report_error("tts", "espeak-ng not found - skipping")
    return True


# 録音とWhisperによる文字起こし
def _record_audio(sr, recognizer, cancel_event):
    """Record one utterance while allowing the caller to stop waiting."""
    def record():
        microphone = sr.Microphone(device_index=_device_index())
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[voice] listening")
            return recognizer.listen(
                source,
                timeout=LISTEN_TIMEOUT,
                phrase_time_limit=PHRASE_TIME_LIMIT,
            )

    try:
        audio = _run_cancellable(record, cancel_event)
    except sr.WaitTimeoutError:
        _report_error("timeout", "no speech detected")
        return None
    if audio is None:
        _report_error("cancelled", "recording result discarded")
    return audio


def _transcribe_audio(audio, cancel_event):
    """Run model loading and transcription behind a cancellation boundary."""
    def transcribe():
        global _whisper_model
        whisper = importlib.import_module("whisper")
        torch = importlib.import_module("torch")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with _whisper_model_lock:
            if _whisper_model is None:
                print(
                    f"[voice] loading whisper model: {MODEL} ({device})"
                )
                _whisper_model = whisper.load_model(
                    MODEL, device=device
                )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_file.write(audio.get_wav_data())
            wav_path = wav_file.name
        try:
            result = _whisper_model.transcribe(
                wav_path,
                language=LANGUAGE,
                verbose=None,
                fp16=device == "cuda",
                condition_on_previous_text=False,
                initial_prompt=WHISPER_PROMPT,
            )
            return result["text"].strip()
        finally:
            os.unlink(wav_path)

    text = _run_cancellable(transcribe, cancel_event)
    if text is None:
        _report_error("cancelled", "Whisper result discarded")
    return text


def transcribe_microphone(cancel_event=None):
    """Record one utterance and return its Whisper transcription."""
    if _is_cancelled(cancel_event):
        _report_error("cancelled", "voice command stopped")
        return None
    try:
        sr = importlib.import_module("speech_recognition")
        selected_index = _device_index()
        microphone = "default" if selected_index is None else selected_index
        print(f"[voice] mic={microphone} model={MODEL}")
        recognizer = sr.Recognizer()
        audio = _record_audio(sr, recognizer, cancel_event)
        if audio is None:
            return None
        text = _transcribe_audio(audio, cancel_event)
    except Exception as exc:
        _report_error("recognition", exc)
        return None
    if text is None:
        return None

    print(f"[voice] recognized: {text}")
    return text


# 認識した文字から移動先を決定
def parse_voice_command(text):
    """Return A, B, or C when text contains a supported movement command."""
    if not text or not _GO_KEYWORDS.search(text):
        return None
    for pattern, destination in _DESTINATION_RULES:
        if pattern.search(text):
            return destination
    return None


def recognize_navigation_command(cancel_event=None):
    """Transcribe one utterance and return its navigation pose."""
    text = transcribe_microphone(cancel_event=cancel_event)
    destination = parse_voice_command(text)
    if destination is None:
        _report_error("command", "no movement command detected")
        return None

    pose = location_pose(destination)
    print(
        f"[voice] command: go to {destination} "
        f"({pose[0]}, {pose[1]}, {math.degrees(pose[2]):.0f} deg)"
    )
    return pose
