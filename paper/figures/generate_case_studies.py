"""
Generate polished case study visualizations for the analysis section.

Case 1: Fluoroquinolone tendinopathy risk amplified by patient factors
Case 2: Absolute contraindication to tPA identified from patient-specific lab value
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Design tokens ──────────────────────────────────────────────────────────────
FONT_FAMILY = "DejaVu Sans"
BG          = "#F8F9FB"      # figure background

# Card accent colors  (border / header bg)
ACT = {
    "patient":   ("#2563EB", "#DBEAFE", "#1E40AF"),   # blue
    "reasoner":  ("#D97706", "#FEF3C7", "#92400E"),   # amber
    "evidence":  ("#0D9488", "#CCFBF1", "#065F46"),   # teal
    "validator": ("#7C3AED", "#EDE9FE", "#4C1D95"),   # violet
    "correct":   ("#059669", "#D1FAE5", "#064E3B"),   # emerald
}

BADGE_CONTRA  = ("#FEE2E2", "#DC2626")   # red  fill / text
BADGE_SUPPORT = ("#D1FAE5", "#065F46")   # green fill / text
BADGE_WRONG   = ("#FEE2E2", "#B91C1C")
BADGE_OK      = ("#D1FAE5", "#065F46")


def shadow_box(ax, x, y, w, h, offset=0.004):
    """Draw a subtle drop shadow then a white card on top."""
    # shadow layer
    shadow = FancyBboxPatch(
        (x + offset, y - offset), w, h,
        boxstyle="round,pad=0.012",
        facecolor="#CBD5E1", edgecolor="none", alpha=0.55,
        transform=ax.transAxes, clip_on=False, zorder=1,
    )
    ax.add_patch(shadow)
    # white card
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012",
        facecolor="white", edgecolor="#E2E8F0", linewidth=1.2,
        transform=ax.transAxes, clip_on=False, zorder=2,
    )
    ax.add_patch(card)


def header_strip(ax, x, y, w, h, color, text, step_num=None):
    """Colored header strip at top of a card, with optional step circle."""
    strip = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012",
        facecolor=color, edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=3,
    )
    ax.add_patch(strip)

    tx = x + w / 2
    if step_num is not None:
        circ = Circle((x + 0.022, y + h / 2), 0.019,
                      facecolor="white", edgecolor="none",
                      transform=ax.transAxes, clip_on=False, zorder=4)
        ax.add_patch(circ)
        ax.text(x + 0.022, y + h / 2, str(step_num),
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color=color,
                zorder=5, clip_on=False)
        tx = x + 0.050 + (w - 0.050) / 2

    ax.text(tx, y + h / 2, text,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white",
            zorder=4, clip_on=False)


def card_text(ax, x, y, lines, fontsize=9.5, color="#1E293B",
              linespacing=1.3, fontweight="normal"):
    ax.text(x, y, lines,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=fontsize, color=color, linespacing=linespacing,
            fontweight=fontweight, clip_on=False, zorder=3)


def badge(ax, x, y, label, fc, tc, fontsize=9.0):
    ax.text(x, y, f"  {label}  ",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=fontsize, fontweight="bold", color=tc,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=fc,
                      edgecolor=tc, linewidth=1.2),
            clip_on=False, zorder=5)


def answer_box(ax, x, y, w, h, text, fc, ec, fontsize=9.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
        transform=ax.transAxes, clip_on=False, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize, color=ec, fontweight="bold",
            linespacing=1.4, clip_on=False, zorder=4)


def connector_arrow(ax, x0, y, x1, color="#94A3B8"):
    """Horizontal arrow between two x positions at height y (axes fraction)."""
    from matplotlib.patches import FancyArrowPatch
    arr = FancyArrowPatch(
        posA=(x0, y), posB=(x1, y),
        arrowstyle="-|>",
        mutation_scale=14,
        facecolor=color, edgecolor=color,
        linewidth=1.6,
        transform=ax.transAxes,
        clip_on=False,
        zorder=6,
    )
    ax.add_patch(arr)


def divider(ax, x, y, w, color="#E2E8F0"):
    ax.plot([x, x + w], [y, y], transform=ax.transAxes,
            color=color, lw=0.8, zorder=3, clip_on=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared layout constants
# ═══════════════════════════════════════════════════════════════════════════════
GAP    = 0.013   # gap between cards
CARDS  = 5
CW     = (1.0 - (CARDS + 1) * GAP) / CARDS   # card width
CH     = 0.78    # card height
CARD_Y = 0.06    # bottom y of all cards
HDR_H  = 0.090   # header strip height
PAD    = 0.020   # inner padding
ARR_Y  = CARD_Y + CH / 2


def card_x(i):
    return GAP + i * (CW + GAP)


# ═══════════════════════════════════════════════════════════════════════════════
# Case 1
# ═══════════════════════════════════════════════════════════════════════════════

def make_case1():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── Title bar ─────────────────────────────────────────────────────────────
    title_strip = FancyBboxPatch(
        (0, 0.90), 1.0, 0.095,
        boxstyle="round,pad=0", facecolor="#1E3A5F", edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=1,
    )
    ax.add_patch(title_strip)
    ax.text(0.5, 0.952,
            "Case 1 · Fluoroquinolone Tendinopathy Risk Amplified by Patient Factors",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", clip_on=False, zorder=2)

    # ── Cards ─────────────────────────────────────────────────────────────────
    labels = ["Patient Context", "Reasoner  (Initial)",
              "Evidence Retrieval", "Validator Judgment", "Reasoner  (Revised)"]
    keys   = ["patient", "reasoner", "evidence", "validator", "correct"]

    for i, (lbl, key) in enumerate(zip(labels, keys)):
        cx = card_x(i)
        shadow_box(ax, cx, CARD_Y, CW, CH)
        header_strip(ax, cx, CARD_Y + CH - HDR_H, CW, HDR_H,
                     ACT[key][0], lbl, step_num=i + 1)

    # ── Arrows ────────────────────────────────────────────────────────────────
    for i in range(4):
        connector_arrow(ax, card_x(i) + CW + 0.003, ARR_Y,
                            card_x(i + 1) - 0.003)

    # ── Content helpers ───────────────────────────────────────────────────────
    def cy(i):   return CARD_Y + CH - HDR_H - PAD   # content top y
    def cx2(i):  return card_x(i) + PAD

    # ── Card 0: Patient ───────────────────────────────────────────────────────
    i = 0
    card_text(ax, cx2(i), cy(i),
              "42F, left ankle pain 2 days\n"
              "after starting ciprofloxacin\n"
              "(Salmonella gastroenteritis)")
    divider(ax, cx2(i), cy(i) - 0.150, CW - 2 * PAD)
    card_text(ax, cx2(i), cy(i) - 0.175,
              "Risk factors",
              fontsize=8.5, color="#64748B")
    card_text(ax, cx2(i), cy(i) - 0.225,
              "● Age > 40\n"
              "● Smoking  2 PPD × 25 yr\n"
              "● Alcohol  2–3 beers/day\n"
              "● BMI 30")

    # ── Card 1: Reasoner (Initial) ────────────────────────────────────────────
    i = 1
    card_text(ax, cx2(i), cy(i),
              "Most likely cause\nof ankle pain?")
    card_text(ax, cx2(i), cy(i) - 0.125,
              "Initial answer",
              fontsize=8.5, color="#64748B")
    answer_box(ax, cx2(i), cy(i) - 0.38, CW - 2 * PAD, 0.20,
               "B: Reactive arthritis\n(triggered by\ngastroenteritis)",
               BADGE_WRONG[0], BADGE_WRONG[1])
    card_text(ax, cx2(i), cy(i) - 0.435,
              "Plausible: temporal\nassociation with\nbacterial infection.",
              fontsize=8.5, color="#64748B")

    # ── Card 2: Evidence Retrieval ────────────────────────────────────────────
    i = 2
    card_text(ax, cx2(i), cy(i),
              "KG relations",
              fontsize=8.5, color="#64748B")
    card_text(ax, cx2(i), cy(i) - 0.050,
              "ciprofloxacin\n  → contraindication\n     → tendinitis")
    card_text(ax, cx2(i), cy(i) - 0.215,
              "Risk amplifiers: age >40,\nsmoking, alcohol use")
    divider(ax, cx2(i), cy(i) - 0.305, CW - 2 * PAD)
    card_text(ax, cx2(i), cy(i) - 0.330,
              "Patient match",
              fontsize=8.5, color="#64748B")
    card_text(ax, cx2(i), cy(i) - 0.380,
              "Age > 40      ✓\n"
              "Smoking       ✓\n"
              "Alcohol use   ✓\n"
              "Onset 2d after cipro\n→ drug-induced timing")

    # ── Card 3: Validator Judgment ────────────────────────────────────────────
    i = 3
    card_text(ax, cx2(i), cy(i), "Claim B", fontsize=9.5, fontweight="bold")
    badge(ax, cx2(i), cy(i) - 0.085, "CONTRADICTED",
          BADGE_CONTRA[0], BADGE_CONTRA[1])
    card_text(ax, cx2(i), cy(i) - 0.140,
              "Onset 2d after cipro.\n"
              "All 3 risk amplifiers present.\n"
              "Drug-induced tendinopathy\n"
              "better explains the picture.",
              color="#374151")
    divider(ax, cx2(i), cy(i) - 0.345, CW - 2 * PAD)
    card_text(ax, cx2(i), cy(i) - 0.370, "Claim A", fontsize=9.5, fontweight="bold")
    badge(ax, cx2(i), cy(i) - 0.455, "SUPPORTED",
          BADGE_SUPPORT[0], BADGE_SUPPORT[1])
    card_text(ax, cx2(i), cy(i) - 0.510,
              "Adverse medication effect\nconsistent with KG evidence\n+ patient risk profile.",
              color="#374151")

    # ── Card 4: Reasoner (Revised) ────────────────────────────────────────────
    i = 4
    card_text(ax, cx2(i), cy(i),
              "Revised answer",
              fontsize=8.5, color="#64748B")
    answer_box(ax, cx2(i), cy(i) - 0.34, CW - 2 * PAD, 0.20,
               "A: Adverse medication\neffect (ciprofloxacin\ntendinopathy)",
               BADGE_OK[0], BADGE_OK[1])
    ax.text(card_x(i) + CW / 2, CARD_Y + 0.055,
            "✓  Correct",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=ACT["correct"][0],
            clip_on=False, zorder=4)

    plt.tight_layout(rect=[0, 0, 1, 1])
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"case_study_1.{ext}")
        plt.savefig(out, bbox_inches="tight", dpi=180, facecolor=BG)
        print(f"Saved: {out}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Case 2
# ═══════════════════════════════════════════════════════════════════════════════

def make_case2():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ── Title bar ─────────────────────────────────────────────────────────────
    title_strip = FancyBboxPatch(
        (0, 0.90), 1.0, 0.095,
        boxstyle="round,pad=0", facecolor="#1E3A5F", edgecolor="none",
        transform=ax.transAxes, clip_on=False, zorder=1,
    )
    ax.add_patch(title_strip)
    ax.text(0.5, 0.952,
            "Case 2 · Absolute Contraindication to tPA Identified from Patient-Specific Lab Value",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", clip_on=False, zorder=2)

    # ── Cards ─────────────────────────────────────────────────────────────────
    labels = ["Patient Context", "Reasoner  (Initial)",
              "Evidence Retrieval", "Validator Judgment", "Reasoner  (Revised)"]
    keys   = ["patient", "reasoner", "evidence", "validator", "correct"]

    for i, (lbl, key) in enumerate(zip(labels, keys)):
        cx = card_x(i)
        shadow_box(ax, cx, CARD_Y, CW, CH)
        header_strip(ax, cx, CARD_Y + CH - HDR_H, CW, HDR_H,
                     ACT[key][0], lbl, step_num=i + 1)

    # ── Arrows ────────────────────────────────────────────────────────────────
    for i in range(4):
        connector_arrow(ax, card_x(i) + CW + 0.003, ARR_Y,
                            card_x(i + 1) - 0.003)

    # ── Content helpers ───────────────────────────────────────────────────────
    def cy(i):  return CARD_Y + CH - HDR_H - PAD
    def cx2(i): return card_x(i) + PAD

    # ── Card 0: Patient ───────────────────────────────────────────────────────
    i = 0
    card_text(ax, cx2(i), cy(i),
              "71M, acute ischemic stroke\n"
              "(within thrombolytic\n"
              "window)")
    divider(ax, cx2(i), cy(i) - 0.150, CW - 2 * PAD)
    card_text(ax, cx2(i), cy(i) - 0.175,
              "History", fontsize=8.5, color="#64748B")
    card_text(ax, cx2(i), cy(i) - 0.225,
              "● ITP (on prednisone)\n"
              "● Plt  95,000/mm³\n"
              "● BP   175/105 mmHg")

    # ── Card 1: Reasoner (Initial) ────────────────────────────────────────────
    i = 1
    card_text(ax, cx2(i), cy(i),
              "Which factor absolutely\ncontraindicates IV tPA?")
    card_text(ax, cx2(i), cy(i) - 0.125,
              "Initial answer",
              fontsize=8.5, color="#64748B")
    answer_box(ax, cx2(i), cy(i) - 0.38, CW - 2 * PAD, 0.20,
               "A: BP 175/105 mmHg\n(hypertension as\ncontraindication)",
               BADGE_WRONG[0], BADGE_WRONG[1])
    card_text(ax, cx2(i), cy(i) - 0.435,
              "BP appeared most\nsalient hazard for\nischemic stroke.",
              fontsize=8.5, color="#64748B")

    # ── Card 2: Evidence Retrieval ────────────────────────────────────────────
    i = 2
    card_text(ax, cx2(i), cy(i),
              "KG relations",
              fontsize=8.5, color="#64748B")
    card_text(ax, cx2(i), cy(i) - 0.050,
              "low platelet count\n  → increases risk\n     → bleeding")
    card_text(ax, cx2(i), cy(i) - 0.215,
              "tPA eligibility threshold:\nplt < 100,000/mm³")
    divider(ax, cx2(i), cy(i) - 0.305, CW - 2 * PAD)
    card_text(ax, cx2(i), cy(i) - 0.330,
              "Patient match",
              fontsize=8.5, color="#64748B")
    card_text(ax, cx2(i), cy(i) - 0.380,
              "Plt 95k < 100k  ✓\n"
              "ITP diagnosis   ✓\n"
              "BP 175/105:\n"
              "  below SBP >185\n"
              "  threshold      ✗")

    # ── Card 3: Validator Judgment ────────────────────────────────────────────
    i = 3
    card_text(ax, cx2(i), cy(i), "Claim A", fontsize=9.5, fontweight="bold")
    badge(ax, cx2(i), cy(i) - 0.085, "CONTRADICTED",
          BADGE_CONTRA[0], BADGE_CONTRA[1])
    card_text(ax, cx2(i), cy(i) - 0.140,
              "BP 175/105 is below\nabsolute cutoff\n(SBP >185 / DBP >110).\nNot an absolute CI.",
              color="#374151")
    divider(ax, cx2(i), cy(i) - 0.345, CW - 2 * PAD)
    card_text(ax, cx2(i), cy(i) - 0.370, "Claim C", fontsize=9.5, fontweight="bold")
    badge(ax, cx2(i), cy(i) - 0.455, "SUPPORTED",
          BADGE_SUPPORT[0], BADGE_SUPPORT[1])
    card_text(ax, cx2(i), cy(i) - 0.510,
              "Plt 95k < 100k threshold.\n"
              "ITP + thrombocytopenia\n= absolute CI to tPA.",
              color="#374151")

    # ── Card 4: Reasoner (Revised) ────────────────────────────────────────────
    i = 4
    card_text(ax, cx2(i), cy(i),
              "Revised answer",
              fontsize=8.5, color="#64748B")
    answer_box(ax, cx2(i), cy(i) - 0.34, CW - 2 * PAD, 0.20,
               "C: Platelet count\n95,000/mm³\n(absolute CI to IV tPA)",
               BADGE_OK[0], BADGE_OK[1])
    ax.text(card_x(i) + CW / 2, CARD_Y + 0.055,
            "✓  Correct",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=ACT["correct"][0],
            clip_on=False, zorder=4)

    plt.tight_layout(rect=[0, 0, 1, 1])
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"case_study_2.{ext}")
        plt.savefig(out, bbox_inches="tight", dpi=180, facecolor=BG)
        print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    make_case1()
    make_case2()
    print("Done.")
