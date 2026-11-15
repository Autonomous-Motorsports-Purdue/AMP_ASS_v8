"""Read image dimensions from file headers using only the stdlib.

Keeps scripts/inventory.py runnable on a bare Python install: no PIL, no numpy.
Returns (width, height) or None if the format isn't recognised / file is short.
"""
import struct

def _jpeg_size(f):
    f.seek(2)
    while True:
        b = f.read(1)
        while b and b != b"\xff":
            b = f.read(1)
        while b == b"\xff":
            b = f.read(1)
        if not b:
            return None
        marker = b[0]
        # SOF0..SOF15 carry the frame dims; DHT/JPG/DAC are not SOF despite range
        if marker in (0xC4, 0xC8, 0xCC) or not (0xC0 <= marker <= 0xCF):
            seg = f.read(2)
            if len(seg) < 2:
                return None
            length = struct.unpack(">H", seg)[0]
            f.seek(length - 2, 1)
            continue
        f.read(3)  # segment length (2) + sample precision (1)
        hw = f.read(4)
        if len(hw) < 4:
            return None
        h, w = struct.unpack(">HH", hw)
        return w, h

def _png_size(head, f):
    if head[12:16] == b"IHDR":
        w, h = struct.unpack(">II", head[16:24])
        return w, h
    return None

def _webp_size(head, f):
    fmt = head[12:16]
    f.seek(20)
    if fmt == b"VP8 ":
        d = f.read(10)
        if len(d) < 10:
            return None
        return struct.unpack("<HH", d[6:10])[0] & 0x3FFF, struct.unpack("<HH", d[6:10])[1] & 0x3FFF
    if fmt == b"VP8L":
        d = f.read(5)
        if len(d) < 5:
            return None
        bits = struct.unpack("<I", d[1:5])[0]
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if fmt == b"VP8X":
        d = f.read(10)
        if len(d) < 10:
            return None
        w = d[4] | (d[5] << 8) | (d[6] << 16)
        h = d[7] | (d[8] << 8) | (d[9] << 16)
        return w + 1, h + 1
    return None

def image_size(path):
    try:
        with open(path, "rb") as f:
            head = f.read(32)
            if len(head) < 24:
                return None
            if head[:2] == b"\xff\xd8":
                return _jpeg_size(f)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return _png_size(head, f)
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                return _webp_size(head, f)
            if head[:2] == b"BM":
                f.seek(18)
                d = f.read(8)
                w, h = struct.unpack("<ii", d)
                return abs(w), abs(h)
    except OSError:
        return None
    return None
