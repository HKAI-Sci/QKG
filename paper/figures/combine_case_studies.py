"""
Combine the two case-study figures into one vertically stacked figure.
The source images already contain their own titles, so no extra panel labels
are added here. Writes both PNG and PDF outputs.
"""

from __future__ import annotations

import os

from PIL import Image


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE1 = os.path.join(OUT_DIR, "case_study_1.png")
CASE2 = os.path.join(OUT_DIR, "case_study_2.png")
PNG_PATH = os.path.join(OUT_DIR, "case_studies_combined.png")
PDF_PATH = os.path.join(OUT_DIR, "case_studies_combined.pdf")

GAP = 28
PAD = 14
BG = (255, 255, 255)


def main() -> None:
    im1 = Image.open(CASE1).convert("RGB")
    im2 = Image.open(CASE2).convert("RGB")

    width = max(im1.width, im2.width) + PAD * 2
    height = im1.height + im2.height + GAP + PAD * 2
    canvas = Image.new("RGB", (width, height), BG)

    x1 = (width - im1.width) // 2
    x2 = (width - im2.width) // 2
    y1 = PAD
    y2 = PAD + im1.height + GAP

    canvas.paste(im1, (x1, y1))
    canvas.paste(im2, (x2, y2))

    canvas.save(PNG_PATH)
    canvas.save(PDF_PATH, "PDF", resolution=150.0)
    print(f"Saved: {PNG_PATH}")
    print(f"Saved: {PDF_PATH}")


if __name__ == "__main__":
    main()
