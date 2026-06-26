import math
import re

LOCATIONS = {
    "A": [-0.48, -1, 270],
    "B": [0.17, -1, 270],
    "C": [0.8, -1, 270],
}

_GO_KEYWORDS = re.compile(r"行|いって|移動|向か")
_DESTINATION_RULES = (
    (re.compile(r"[Aa]|えー|エー|えい"), "A"),
    (re.compile(r"[Bb]|びー|ビー"), "B"),
    (re.compile(r"[Cc]|しー|シー"), "C"),
)


def location_pose(destination):
    x, y, theta_deg = LOCATIONS[destination]
    return float(x), float(y), math.radians(theta_deg)


def parse_voice_command(text):
    """Return 'A'/'B'/'C' when text looks like a go command, else None."""
    if not text or not _GO_KEYWORDS.search(text):
        return None
    for pattern, destination in _DESTINATION_RULES:
        if pattern.search(text):
            return destination
    return None


def process_voice_text(text):
    destination = parse_voice_command(text)
    pose = location_pose(destination) if destination else None
    return {
        "text": text,
        "destination": destination,
        "pose": pose,
    }
