"""Shared navigation destinations used by voice and QR commands."""

import math


DESTINATIONS = {
    "A": (-0.1330670714378357, -1.0817683935165405, 270.0),
    "B": (0.400907963514328, -1.0347225666046143, 270.0),
    "C": (1.0341026782989502, -1.0767114162445068, 270.0),
    # One C-to-A interval further left from A.
    "INIT": (-1.3002368211746216, -1.0868253707885742, 270.0),
}


def location_pose(destination):
    """Return an (x, y, yaw-radians) navigation pose."""
    x, y, yaw_degrees = DESTINATIONS[destination]
    return float(x), float(y), math.radians(yaw_degrees)
