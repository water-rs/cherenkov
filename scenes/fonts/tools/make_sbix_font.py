#!/usr/bin/env python3
"""Generate the deterministic Cherenkov sbix test font."""

from __future__ import annotations

import io
from pathlib import Path

from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables.sbixStrike import Glyph, Strike
from PIL import Image


HERE = Path(__file__).resolve().parents[1]
SOURCE = HERE / "NotoColorEmojiSubset.ttf"
OUTPUT = HERE / "CherenkovSbixTest.ttf"
FIXED_TIMESTAMP = 3_155_328_000


def png_at_size(png: bytes, ppem: int) -> bytes:
    with Image.open(io.BytesIO(png)) as image:
        width = round(image.width * ppem / 109)
        height = round(image.height * ppem / 109)
        resized = image.convert("RGBA").resize(
            (width, height), Image.Resampling.LANCZOS
        )
        output = io.BytesIO()
        resized.save(output, format="PNG", optimize=False)
        return output.getvalue()


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
