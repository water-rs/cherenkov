#!/usr/bin/env python3
"""Generate the deterministic Cherenkov sbix test font."""

from __future__ import annotations

import binascii
import io
import struct
import zlib
from pathlib import Path

from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables.sbixStrike import Glyph, Strike
from PIL import Image


HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "NotoColorEmojiSubset.ttf"
OUTPUT = HERE / "CherenkovSbixTest.ttf"
FIXED_TIMESTAMP = 3_155_328_000


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(tag + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def encode_png_rgba(image) -> bytes:
    """Encode an RGBA image as Pillow's PNG encoder does, deflating through
    the standard-library zlib rather than Pillow itself: macOS Pillow wheels
    bundle zlib-ng while Linux wheels use the system zlib, so Pillow writes
    different deflate streams on different hosts for identical input.
    """
    stride = image.width * 4
    px = image.tobytes()
    raw = bytearray()
    prev = bytes(stride)
    for y in range(image.height):
        row = px[y * stride : (y + 1) * stride]
        best, best_row = 0, row
        best_sum = sum(v if v < 128 else 256 - v for v in row)
        if best_sum > 0:
            filtered = bytes((row[x] - prev[x]) & 0xFF for x in range(stride))
            s = sum(v if v < 128 else 256 - v for v in filtered)
            if s < best_sum:
                best, best_sum, best_row = 2, s, filtered
        if best_sum > 0:
            filtered = bytes(
                (row[x] - (row[x - 4] if x >= 4 else 0)) & 0xFF
                for x in range(stride)
            )
            s = sum(v if v < 128 else 256 - v for v in filtered)
            if s < best_sum:
                best, best_sum, best_row = 1, s, filtered
        if best_sum > 0:
            filtered = bytes(
                (
                    row[x]
                    - _paeth(
                        row[x - 4] if x >= 4 else 0,
                        prev[x],
                        prev[x - 4] if x >= 4 else 0,
                    )
                )
                & 0xFF
                for x in range(stride)
            )
            s = sum(v if v < 128 else 256 - v for v in filtered)
            if s < best_sum:
                best, best_sum, best_row = 4, s, filtered
        raw.append(best)
        raw += best_row
        prev = row
    comp = zlib.compressobj(-1, zlib.DEFLATED, 15, 9, zlib.Z_FILTERED)
    idat = comp.compress(bytes(raw)) + comp.flush()
    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", idat)
        + _png_chunk(b"IEND", b"")
    )


def png_at_size(png: bytes, ppem: int) -> bytes:
    with Image.open(io.BytesIO(png)) as image:
        width = round(image.width * ppem / 109)
        height = round(image.height * ppem / 109)
        resized = image.convert("RGBA").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        return encode_png_rgba(resized)


def main() -> None:
    font = TTFont(SOURCE, recalcTimestamp=False)
    cbdt_glyphs = font["CBDT"].strikeData[0]
    del font["CBDT"]
    del font["CBLC"]

    for record in font["name"].names:
        if record.nameID in (1, 16):
            value = "Cherenkov Sbix Test"
        elif record.nameID == 4:
            value = "Cherenkov Sbix Test"
        elif record.nameID == 6:
            value = "CherenkovSbixTest-Regular"
        elif record.nameID == 3:
            value = "Cherenkov Sbix Test Regular"
        else:
            continue
        record.string = value.encode(record.getEncoding())

    for glyph_name, (advance, _lsb) in font["hmtx"].metrics.items():
        font["hmtx"].metrics[glyph_name] = (advance, 0)

    font["head"].created = FIXED_TIMESTAMP
    font["head"].modified = FIXED_TIMESTAMP
    font.recalcTimestamp = False

    sbix = newTable("sbix")
    sbix.version = 1
    sbix.flags = 1
    sbix.strikes = {}
    for ppem in (32, 96):
        strike = Strike(ppem=ppem, resolution=72)
        for glyph_name, cbdt_glyph in cbdt_glyphs.items():
            strike.glyphs[glyph_name] = Glyph(
                glyphName=glyph_name,
                graphicType="png ",
                originOffsetX=0,
                originOffsetY=round(-27 * ppem / 109),
                imageData=png_at_size(cbdt_glyph.imageData, ppem),
            )
        sbix.strikes[ppem] = strike
    font["sbix"] = sbix
    font.save(OUTPUT, reorderTables=True)


if __name__ == "__main__":
    main()
