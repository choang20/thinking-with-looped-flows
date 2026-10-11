"""Small numeric helpers used to keep the demo explicitly float32."""

from __future__ import annotations

import struct


def float32(value: float) -> float:
    """Round a Python number to IEEE-754 single precision."""

    return struct.unpack("=f", struct.pack("=f", float(value)))[0]
