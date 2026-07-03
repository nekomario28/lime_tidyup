try:
    import speech_recognition as sr
except ModuleNotFoundError:
    sr = None


def _require_speech_recognition():
    if sr is None:
        raise RuntimeError(
            "SpeechRecognition is not installed. Run this inside the Docker "
            "environment, or rebuild the image from the Dockerfile."
        )
    return sr


def list_microphones():
    sr = _require_speech_recognition()
    return sr.Microphone.list_microphone_names()


def print_microphones():
    names = list_microphones()
    if not names:
        print("[voice] no microphone found")
        return
    for i, name in enumerate(names):
        print(f"[{i}] {name}")


def select_microphone_index(device_index=None):
    names = list_microphones()
    if not names:
        raise RuntimeError("No microphone found")

    if device_index is not None and device_index >= 0:
        return int(device_index)

    # Prefer PulseAudio when available: this container is set up to talk to
    # the host's PulseAudio server (see PULSE_SERVER/PULSE_COOKIE in
    # docker-compose.yaml), and pulse transparently resamples/handles the
    # real input device, avoiding ALSA "Invalid sample rate" failures.
    for i, name in enumerate(names):
        if name.strip().lower() == "pulse":
            return i

    for i, name in enumerate(names):
        if name == "default":
            return i

    for i, name in enumerate(names):
        lower = name.lower()
        if "hdmi" in lower or "monitor" in lower:
            continue
        if (
            "analog" in lower
            or "mic" in lower
            or "input" in lower
            or "usb audio" in lower
        ):
            return i

    return None
