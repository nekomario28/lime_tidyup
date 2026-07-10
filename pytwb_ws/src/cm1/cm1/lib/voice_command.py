import math
import re

LOCATIONS = {
    "A": [-0.15631535649299622, -1.073258638381958, 270],
    "B": [0.49344485998153687, -1.0844091176986694, 270],
    "C": [1.0341026782989502, -1.0767114162445068, 270],
}

_GO_KEYWORDS = re.compile(r"行|いって|移動|向か")
_DESTINATION_RULES = (
    (re.compile(r"[Aa]|えー|エー|えい"), "A"),
    (re.compile(r"[Bb]|びー|ビー|びい|ピー|ぴー|ぴい|ピイ"), "B"),
    (re.compile(r"[Cc]|しー|シー|しい|ちー|チー"), "C"),
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
