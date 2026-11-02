"""Draw the model extractor's icon: ``polylode/extractor/icon.png`` (256 px) and ``icon.ico`` (16–256 px).

    python tools/make_extractor_icon.py

A rounded square in the window's accent (blue into violet), a cube — the
model — in three shades of white, and an export arrow in a circle at the
lower right.  Drawn at 1024 px and reduced, so the small sizes stay clean.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parents[1] / "polylode" / "extractor"
SIZE = 1024
BLUE, VIOLET = (47, 111, 235), (124, 77, 255)


def _gradient(size: int) -> Image.Image:
    """Top-left blue to bottom-right violet."""
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            pixels[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(BLUE, VIOLET))
    return image


def draw() -> Image.Image:
    s = SIZE
    canvas = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle((40, 40, s - 40, s - 40), radius=220, fill=255)
    canvas.paste(_gradient(s), (0, 0), mask)

    # a soft shadow under the cube
    shadow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((250, 770, 700, 850), fill=(10, 14, 40, 110))
    canvas.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(30)))

    # the cube, isometric: top, left and right faces
    cx, top, edge = 470, 210, 300
    half = edge * 0.866
    apex, left, right, centre = (cx, top), (cx - half, top + edge / 2), (cx + half, top + edge / 2), (cx, top + edge)
    down = edge
    cube = ImageDraw.Draw(canvas)
    cube.polygon([apex, right, centre, left], fill=(255, 255, 255, 250))
    cube.polygon([left, centre, (centre[0], centre[1] + down), (left[0], left[1] + down)], fill=(225, 232, 255, 235))
    cube.polygon([centre, right, (right[0], right[1] + down), (centre[0], centre[1] + down)], fill=(190, 202, 250, 230))
    line = (47, 60, 140, 90)
    for a, b in ((left, centre), (centre, right), (centre, (centre[0], centre[1] + down))):
        cube.line([a, b], fill=line, width=10)

    # the export badge: a white circle with a downward arrow in the accent colour
    bx, by, r = 735, 735, 175
    badge = ImageDraw.Draw(canvas)
    badge.ellipse((bx - r - 14, by - r - 14, bx + r + 14, by + r + 14), fill=(255, 255, 255, 70))
    badge.ellipse((bx - r, by - r, bx + r, by + r), fill=(255, 255, 255, 255))
    stem = 34
    badge.rounded_rectangle((bx - stem, by - 105, bx + stem, by + 40), radius=stem, fill=VIOLET)
    badge.polygon([(bx - 105, by + 5), (bx + 105, by + 5), (bx, by + 115)], fill=VIOLET)
    badge.rounded_rectangle((bx - 110, by + 128, bx + 110, by + 158), radius=15, fill=BLUE)
    return canvas


def main() -> None:
    image = draw()
    image.resize((256, 256), Image.LANCZOS).save(OUT / "icon.png")
    image.resize((256, 256), Image.LANCZOS).save(OUT / "icon.ico",
                                                 sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128),
                                                        (256, 256)])
    print(OUT / "icon.png", OUT / "icon.ico")


if __name__ == "__main__":
    main()
