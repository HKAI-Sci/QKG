#!/usr/bin/env python3

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SVG_PATH = Path(__file__).with_name("figure1_qkg_overview.svg")

CANVAS_W = 1200
CANVAS_H = 720


@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    rx: float = 0
    ry: float = 0
    fill: str = "#ffffff"
    stroke: str = "#000000"
    stroke_width: float = 1.0
    label: str = ""

    def intersects(self, other: "Rect", padding: float = 0) -> bool:
        return not (
            self.x + self.w + padding <= other.x
            or other.x + other.w + padding <= self.x
            or self.y + self.h + padding <= other.y
            or other.y + other.h + padding <= self.y
        )

    def contains(self, other: "Rect", padding: float = 0) -> bool:
        return (
            other.x >= self.x + padding
            and other.y >= self.y + padding
            and other.x + other.w <= self.x + self.w - padding
            and other.y + other.h <= self.y + self.h - padding
        )


@dataclass
class TextBlock:
    x: float
    y: float
    lines: list[str]
    style: str
    anchor: str = "start"
    fill: str | None = None
    label: str = ""

    def estimate_box(self) -> Rect:
        style_map = {
            "panel-title": (21, 0.58, 1.25),
            "body": (18, 0.54, 1.30),
            "small": (16, 0.53, 1.25),
            "mini": (14, 0.54, 1.22),
            "mono": (18, 0.60, 1.30),
        }
        font_size, width_factor, line_height = style_map[self.style]
        max_chars = max(len(line) for line in self.lines) if self.lines else 0
        width = max_chars * font_size * width_factor
        height = len(self.lines) * font_size * line_height
        if self.anchor == "middle":
            x = self.x - width / 2
        elif self.anchor == "end":
            x = self.x - width
        else:
            x = self.x
        y = self.y - font_size
        return Rect(x=x, y=y, w=width, h=height, label=self.label or "|".join(self.lines))

    def to_svg(self) -> str:
        fill_attr = f' fill="{self.fill}"' if self.fill else ""
        if len(self.lines) == 1:
            return f'<text x="{self.x}" y="{self.y}" class="{self.style}" text-anchor="{self.anchor}"{fill_attr}>{escape(self.lines[0])}</text>'
        tspans = []
        for i, line in enumerate(self.lines):
            dy = "0" if i == 0 else "1.25em"
            tspans.append(
                f'<tspan x="{self.x}" dy="{dy}">{escape(line)}</tspan>'
            )
        return (
            f'<text x="{self.x}" y="{self.y}" class="{self.style}" '
            f'text-anchor="{self.anchor}"{fill_attr}>'
            + "".join(tspans)
            + "</text>"
        )


@dataclass
class SvgDoc:
    parts: list[str] = field(default_factory=list)
    elements: list[tuple[Rect, str | None, str, str | None]] = field(default_factory=list)

    def add_rect(self, rect: Rect, panel: str | None = None, parent: str | None = None) -> None:
        self.parts.append(
            f'<rect x="{rect.x}" y="{rect.y}" width="{rect.w}" height="{rect.h}" '
            f'rx="{rect.rx}" ry="{rect.ry}" fill="{rect.fill}" '
            f'stroke="{rect.stroke}" stroke-width="{rect.stroke_width}"/>'
        )
        self.elements.append((rect, panel, "rect", parent))

    def add_text(self, text: TextBlock, panel: str | None = None, parent: str | None = None) -> None:
        self.parts.append(text.to_svg())
        box = text.estimate_box()
        self.elements.append((box, panel, "text", parent))

    def add_line(self, x1: float, y1: float, x2: float, y2: float, width: float = 2.5, color: str = "#6b7280", marker: str = "arrow") -> None:
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{width}" marker-end="url(#{marker})"/>'
        )

    def add_path(self, d: str, width: float = 2.5, color: str = "#6b7280", marker: str = "arrow") -> None:
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"/>'
        )

    def validate(self, panel_rects: dict[str, Rect]) -> list[str]:
        issues: list[str] = []
        panel_items: dict[str, list[tuple[Rect, str]]] = {}
        for rect, panel, kind, parent in self.elements:
            if panel is None:
                continue
            panel_items.setdefault(panel, []).append((rect, parent or ""))
            if rect.label == f"panel {panel}":
                continue
            if not panel_rects[panel].contains(rect, padding=10):
                issues.append(f"{panel}: item escapes panel bounds: {rect.label}")

        for rect, panel, kind, parent in self.elements:
            if kind != "text" or not panel:
                continue
            for candidate, p, candidate_kind, candidate_parent in self.elements:
                if p != panel or candidate_kind != "rect":
                    continue
                if candidate.label.startswith("panel "):
                    continue
                if parent and candidate.label == parent:
                    continue
                if rect.intersects(candidate, padding=2):
                    issues.append(f"{panel}: text collides with '{candidate.label}': {rect.label}")

        for panel, items in panel_items.items():
            rects = [
                (rect, parent)
                for rect, parent in items
                if rect.label
                and not rect.label.startswith("panel ")
                and any(
                    rect is candidate and kind == "rect"
                    for candidate, p, kind, parent_name in self.elements
                    if p == panel
                )
            ]
            for i, (a, a_parent) in enumerate(rects):
                for b, b_parent in rects[i + 1 :]:
                    if a_parent == b.label or b_parent == a.label:
                        continue
                    if a.intersects(b, padding=0):
                        issues.append(f"{panel}: overlap between '{a.label}' and '{b.label}'")
        return issues

    def render(self) -> str:
        return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" viewBox="0 0 {CANVAS_W} {CANVAS_H}">
  <defs>
    <style>
      .panel-title {{ font-family: Helvetica, Arial, sans-serif; font-size: 21px; font-weight: 700; fill: #1f2937; }}
      .body {{ font-family: Helvetica, Arial, sans-serif; font-size: 18px; font-weight: 400; fill: #374151; }}
      .small {{ font-family: Helvetica, Arial, sans-serif; font-size: 16px; font-weight: 700; font-style: italic; fill: #4b5563; }}
      .mini {{ font-family: Helvetica, Arial, sans-serif; font-size: 14px; font-weight: 700; font-style: italic; fill: #374151; }}
      .mono {{ font-family: Consolas, Menlo, monospace; font-size: 18px; font-weight: 400; fill: #1f2937; }}
    </style>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#6b7280"/>
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#0891b2"/>
    </marker>
  </defs>
  <rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#ffffff"/>
  {' '.join(self.parts)}
</svg>
"""


def escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def add_panel_a(doc: SvgDoc, panel: Rect) -> None:
    doc.add_rect(panel, panel="A")
    doc.add_text(TextBlock(panel.x + 30, panel.y + 38, ["A. Context-Insensitive KG"], "panel-title", label="panel-title"), panel="A")
    doc.add_text(TextBlock(panel.x + panel.w - 30, panel.y + 38, ["P(τ|C) ∈ {0,1}"], "mini", anchor="end", label="A title math"), panel="A")
    doc.add_text(TextBlock(panel.x + 30, panel.y + 60, ["unable to model context-dependent validity"], "mini", label="A subtitle"), panel="A")

    triple = Rect(panel.x + 88, panel.y + 92, 314, 52, rx=12, ry=12, fill="#ffffff", stroke="#9ca3af", stroke_width=1.7, label="A triple")
    ctx1 = Rect(panel.x + 18, panel.y + 182, 214, 90, rx=12, ry=12, fill="#ffffff", stroke="#9ca3af", stroke_width=1.6, label="A context1")
    ctx2 = Rect(panel.x + 258, panel.y + 182, 214, 90, rx=12, ry=12, fill="#ffffff", stroke="#9ca3af", stroke_width=1.6, label="A context2")
    for box in (triple, ctx1, ctx2):
        doc.add_rect(box, panel="A")

    doc.add_text(TextBlock(triple.x + 14, triple.y + 33, ["(drug A, treats, disease B)"], "mono", label="A triple text"), panel="A", parent="A triple")
    doc.add_text(TextBlock(ctx1.x + ctx1.w / 2, ctx1.y + 26, ["Context 1: patient without", "renal failure"], "small", anchor="middle", label="A ctx1 line1"), panel="A", parent="A context1")
    doc.add_text(TextBlock(ctx1.x + ctx1.w / 2, ctx1.y + 76, ["holds"], "body", anchor="middle", label="A ctx1 line2"), panel="A", parent="A context1")
    doc.add_text(TextBlock(ctx2.x + ctx2.w / 2, ctx2.y + 26, ["Context 2: patient with", "renal failure"], "small", anchor="middle", label="A ctx2 line1"), panel="A", parent="A context2")
    doc.add_text(TextBlock(ctx2.x + ctx2.w / 2, ctx2.y + 76, ["does not hold"], "body", anchor="middle", label="A ctx2 line2"), panel="A", parent="A context2")

    center_x = triple.x + triple.w / 2
    branch_y = ctx1.y - 10
    doc.add_line(center_x, triple.y + triple.h, center_x, branch_y, width=2.0)
    doc.add_path(f"M {center_x} {branch_y} C {center_x - 45} {branch_y}, {ctx1.x + ctx1.w / 2} {branch_y}, {ctx1.x + ctx1.w / 2} {ctx1.y}", width=1.8, marker="arrow")
    doc.add_path(f"M {center_x} {branch_y} C {center_x + 45} {branch_y}, {ctx2.x + ctx2.w / 2} {branch_y}, {ctx2.x + ctx2.w / 2} {ctx2.y}", width=1.8, marker="arrow")


def add_panel_b(doc: SvgDoc, panel: Rect) -> None:
    doc.add_rect(panel, panel="B")
    doc.add_text(TextBlock(panel.x + 30, panel.y + 38, ["B. Quantum Knowledge Graph"], "panel-title", label="panel-title"), panel="B")
    doc.add_text(TextBlock(panel.x + panel.w - 30, panel.y + 38, ["P(τ|C) = Fτ(C)"], "mini", anchor="end", label="B title math"), panel="B")
    doc.add_text(TextBlock(panel.x + 30, panel.y + 60, ["models context-dependent validity with LLMs"], "mini", label="B subtitle"), panel="B")

    triple = Rect(panel.x + 84, panel.y + 92, 322, 52, rx=12, ry=12, fill="#ffffff", stroke="#38bdf8", stroke_width=1.7, label="B triple")
    constraints = Rect(panel.x + 44, panel.y + 184, 418, 76, rx=14, ry=14, fill="#ffffff", stroke="#0891b2", stroke_width=1.8, label="B constraints")
    for box in (triple, constraints):
        doc.add_rect(box, panel="B")

    doc.add_text(TextBlock(triple.x + 18, triple.y + 33, ["(drug A, treats, disease B)"], "mono", label="B triple text"), panel="B", parent="B triple")
    doc.add_text(
        TextBlock(
            constraints.x + constraints.w / 2,
            constraints.y + 32,
            [
                "patient with eGFR >= 60 mL/min/1.73 m^2,",
                "UACR < 30 mg/g, and no other clues of renal failure",
            ],
            "small",
            anchor="middle",
            label="B constraints text",
        ),
        panel="B",
        parent="B constraints",
    )
    center_x = triple.x + triple.w / 2
    doc.add_line(center_x, triple.y + triple.h, center_x, constraints.y, width=2.0, color="#0891b2", marker="arrow-blue")
    doc.add_text(
        TextBlock(center_x + 14, triple.y + triple.h + 22, ["Validity condition"], "mini", label="B arrow label"),
        panel="B",
    )


def add_panel_c(doc: SvgDoc, panel: Rect) -> None:
    doc.add_rect(panel, panel="C")
    doc.add_text(TextBlock(panel.x + 30, panel.y + 38, ["C. Reasoner-Validator System"], "panel-title", label="panel-title"), panel="C")

    question = Rect(panel.x + 40, panel.y + 84, 188, 58, rx=6, ry=6, fill="#ffffff", stroke="#9ca3af", stroke_width=1.7, label="C question")
    reasoner = Rect(panel.x + 284, panel.y + 81, 230, 64, rx=14, ry=14, fill="#ffffff", stroke="#ea580c", stroke_width=1.8, label="C reasoner")
    answer = Rect(panel.x + 570, panel.y + 84, 150, 58, rx=6, ry=6, fill="#ffffff", stroke="#16a34a", stroke_width=1.8, label="C answer")
    validator = Rect(panel.x + 284, panel.y + 211, 230, 64, rx=14, ry=14, fill="#eff6ff", stroke="#2563eb", stroke_width=1.8, label="C validator")
    for box in (question, reasoner, validator, answer):
        doc.add_rect(box, panel="C")

    doc.add_text(TextBlock(question.x + question.w / 2, question.y + 35, ["Medical question"], "body", anchor="middle", label="C question text"), panel="C", parent="C question")
    doc.add_text(TextBlock(reasoner.x + reasoner.w / 2, reasoner.y + 24, ["Reasoner"], "panel-title", anchor="middle", label="C reasoner title"), panel="C", parent="C reasoner")
    doc.add_text(TextBlock(reasoner.x + reasoner.w / 2, reasoner.y + 49, ["LLM only"], "small", anchor="middle", label="C reasoner text"), panel="C", parent="C reasoner")
    doc.add_text(TextBlock(validator.x + validator.w / 2, validator.y + 24, ["Validator"], "panel-title", anchor="middle", label="C validator title"), panel="C", parent="C validator")
    doc.add_text(TextBlock(validator.x + validator.w / 2, validator.y + 49, ["QKG"], "small", anchor="middle", label="C validator text"), panel="C", parent="C validator")
    doc.add_text(TextBlock(answer.x + answer.w / 2, answer.y + 35, ["final answer"], "body", anchor="middle", label="C answer text"), panel="C", parent="C answer")

    doc.add_line(question.x + question.w, question.y + question.h / 2, reasoner.x, reasoner.y + reasoner.h / 2, width=2.0)
    doc.add_line(reasoner.x + reasoner.w, reasoner.y + reasoner.h / 2, answer.x, answer.y + answer.h / 2, width=2.0)
    doc.add_path(
        f"M {question.x + question.w / 2} {question.y + question.h} "
        f"L {question.x + question.w / 2} {validator.y + validator.h / 2} "
        f"L {validator.x} {validator.y + validator.h / 2}",
        width=1.8,
    )
    doc.add_text(
        TextBlock(question.x + question.w / 2 + 18, validator.y + validator.h / 2 - 10, ["patient context"], "small", label="C patient context label"),
        panel="C",
    )
    left_arrow_x = reasoner.x + reasoner.w * 0.32
    right_arrow_x = reasoner.x + reasoner.w * 0.68
    doc.add_line(left_arrow_x, reasoner.y + reasoner.h, left_arrow_x, validator.y, width=2.0)
    doc.add_text(
        TextBlock(left_arrow_x - 128, (reasoner.y + reasoner.h + validator.y) / 2 + 6, ["answer + claims"], "small", label="C claims label"),
        panel="C",
    )
    doc.add_line(right_arrow_x, validator.y, right_arrow_x, reasoner.y + reasoner.h, width=1.8)
    doc.add_text(
        TextBlock(right_arrow_x + 24, (reasoner.y + reasoner.h + validator.y) / 2 + 6, ["validation report"], "small", label="C validation label"),
        panel="C",
    )


def add_panel_d(doc: SvgDoc, panel: Rect) -> None:
    doc.add_rect(panel, panel="D")
    doc.add_text(TextBlock(panel.x + 30, panel.y + 38, ["D. Outcome"], "panel-title", label="panel-title"), panel="D")
    base_x = panel.x + 28
    base_y = panel.y + 234
    bar_w = 52
    gap = 28
    bars = [
        (["LLM", "only"], 46, "#d1d5db", "D llm"),
        (["LLM +", "KG"], 72, "#9ca3af", "D kg"),
        (["LLM +", "QKG"], 92, "#22c55e", "D qkg"),
    ]
    doc.add_path(f"M {base_x - 14} {base_y} L {base_x + 3 * (bar_w + gap) - gap + 14} {base_y}", width=1.4, color="#9ca3af", marker="arrow")
    doc.add_path(f"M {base_x - 18} {base_y} L {base_x - 18} {panel.y + 86}", width=1.4, color="#9ca3af", marker="arrow")
    doc.add_text(TextBlock(base_x - 2, panel.y + 92, ["accuracy"], "small", label="D y axis label"), panel="D")

    x = base_x
    for label_lines, height, color, tag in bars:
        bar = Rect(x, base_y - height, bar_w, height, rx=6, ry=6, fill=color, stroke=color, stroke_width=1.0, label=tag)
        doc.add_rect(bar, panel="D")
        doc.add_text(TextBlock(x + bar_w / 2, base_y + 18, label_lines, "mini", anchor="middle", label=f"{tag} text"), panel="D")
        x += bar_w + gap


def build_svg() -> tuple[str, list[str]]:
    doc = SvgDoc()

    panel_rects = {
        "A": Rect(70, 52, 490, 282, rx=18, ry=18, fill="#f3f4f6", stroke="#6b7280", stroke_width=2.5, label="panel A"),
        "B": Rect(640, 52, 490, 282, rx=18, ry=18, fill="#ecfeff", stroke="#0891b2", stroke_width=2.5, label="panel B"),
        "C": Rect(70, 386, 760, 302, rx=18, ry=18, fill="#fff7ed", stroke="#ea580c", stroke_width=2.5, label="panel C"),
        "D": Rect(860, 386, 270, 302, rx=18, ry=18, fill="#ecfdf5", stroke="#16a34a", stroke_width=2.5, label="panel D"),
    }

    add_panel_a(doc, panel_rects["A"])
    add_panel_b(doc, panel_rects["B"])
    add_panel_c(doc, panel_rects["C"])
    add_panel_d(doc, panel_rects["D"])

    doc.add_line(panel_rects["A"].x + panel_rects["A"].w, 193, panel_rects["B"].x, 193, width=2.8)
    doc.add_path(f"M 885 334 L 885 360 L 315 360 L 315 386", width=2.8)

    return doc.render(), doc.validate(panel_rects)


def main() -> int:
    svg, issues = build_svg()
    if issues:
        print("layout validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1
    SVG_PATH.write_text(svg, encoding="utf-8")
    print(f"wrote {SVG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
