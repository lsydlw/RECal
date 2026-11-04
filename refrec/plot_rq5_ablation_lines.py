#!/usr/bin/env python3
"""Generate a four-panel line chart for the RQ5 ablation study.

Each panel corresponds to one backbone.  The x axis contains all eight
datasets and the y axis is the absolute NDCG@10 change (percentage points)
relative to the same frozen Base.  Seven ablations and the complete
one-sided model are shown; the unguarded full-projection variant is excluded.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from plot_rq5_ablation import DATA, DATASETS, VARIANTS


DISPLAY = {
    "Fixed boost": "Fixed boost",
    "Fixed global $g$": "Fixed global $g$",
    "Entropy conf.": "Entropy conf.",
    "$q$ only": "$q$ only",
    "Global repeat": "Global repeat",
    "w/o seq. stats": "w/o seq. stats",
    "Bidirectional": "Bidirectional",
    "One-sided (Ours)": "One-sided (Ours)",
}

COLORS = {
    "Fixed boost": "boost",
    "Fixed global $g$": "fixedg",
    "Entropy conf.": "entropy",
    "$q$ only": "qonly",
    "Global repeat": "globalr",
    "w/o seq. stats": "nostats",
    "Bidirectional": "bidir",
    "One-sided (Ours)": "ours",
}

LINE_STYLES = {
    "Fixed boost": "densely dashed",
    "Fixed global $g$": "densely dotted",
    "Entropy conf.": "dash pattern=on 3pt off 1.5pt on 0.7pt off 1.5pt",
    "$q$ only": "dashed",
    "Global repeat": "dotted",
    "w/o seq. stats": "solid",
    "Bidirectional": "dash pattern=on 5pt off 1.6pt",
    "One-sided (Ours)": "solid",
}

MARKERS = {
    "Fixed boost": "circle",
    "Fixed global $g$": "square",
    "Entropy conf.": "triangle",
    "$q$ only": "diamond",
    "Global repeat": "cross",
    "w/o seq. stats": "circle",
    "Bidirectional": "square",
    "One-sided (Ours)": "circle",
}


def marker_tex(kind: str, x: float, y: float, color: str, ours: bool) -> str:
    r = 1.65 if ours else 1.15
    lw = 0.8 if ours else 0.45
    if kind == "square":
        return rf"\filldraw[fill=white,draw={color},line width={lw}pt] ({x-r:.2f},{y-r:.2f}) rectangle ({x+r:.2f},{y+r:.2f});"
    if kind == "triangle":
        return rf"\filldraw[fill=white,draw={color},line width={lw}pt] ({x:.2f},{y+r:.2f}) -- ({x-r:.2f},{y-r:.2f}) -- ({x+r:.2f},{y-r:.2f}) -- cycle;"
    if kind == "diamond":
        return rf"\filldraw[fill=white,draw={color},line width={lw}pt] ({x:.2f},{y+r:.2f}) -- ({x+r:.2f},{y:.2f}) -- ({x:.2f},{y-r:.2f}) -- ({x-r:.2f},{y:.2f}) -- cycle;"
    if kind == "cross":
        return rf"\draw[{color},line width={lw}pt] ({x-r:.2f},{y-r:.2f}) -- ({x+r:.2f},{y+r:.2f}) ({x-r:.2f},{y+r:.2f}) -- ({x+r:.2f},{y-r:.2f});"
    return rf"\filldraw[fill=white,draw={color},line width={lw}pt] ({x:.2f},{y:.2f}) circle ({r:.2f}pt);"


def build_tex() -> str:
    canvas_w, canvas_h = 500.0, 360.0
    panel_w, panel_h = 202.0, 91.0
    panels = {
        "BERT4Rec": (36.0, 182.0, "(a) BERT4Rec"),
        "GRU4Rec": (286.0, 182.0, "(b) GRU4Rec"),
        "RepeatNet": (36.0, 32.0, "(c) RepeatNet"),
        "SASRec": (286.0, 32.0, "(d) SASRec"),
    }
    y_min, y_max = -6.5, 6.5

    def px(left: float, j: int) -> float:
        return left + j * panel_w / 7.0

    def py(bottom: float, value: float) -> float:
        return bottom + (value - y_min) / (y_max - y_min) * panel_h

    lines = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage{lmodern}",
        r"\usepackage[T1]{fontenc}",
        r"\definecolor{ink}{HTML}{243447}",
        r"\definecolor{muted}{HTML}{64748B}",
        r"\definecolor{grid}{HTML}{DCE4EC}",
        r"\definecolor{boost}{HTML}{D95F59}",
        r"\definecolor{fixedg}{HTML}{E59A2F}",
        r"\definecolor{entropy}{HTML}{8C6BB1}",
        r"\definecolor{qonly}{HTML}{59A14F}",
        r"\definecolor{globalr}{HTML}{B07C59}",
        r"\definecolor{nostats}{HTML}{2A9D8F}",
        r"\definecolor{bidir}{HTML}{7A8793}",
        r"\definecolor{ours}{HTML}{1769AA}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1pt,y=1pt,font=\sffamily]",
        rf"\path[use as bounding box] (0,0) rectangle ({canvas_w:.1f},{canvas_h:.1f});",
        r"\node[anchor=west,text=ink,font=\bfseries\fontsize{8.5}{9.5}\selectfont] at (2,350) {RQ5: per-setting ablation effects};",
        r"\node[anchor=west,text=muted,font=\fontsize{6}{7}\selectfont] at (2,339) {Absolute NDCG@10 change over the same frozen Base (percentage points)};",
    ]

    # Two-row legend: four entries per row.
    legend_x = [8, 132, 256, 380]
    legend_y = [322, 307]
    for idx, variant in enumerate(VARIANTS):
        row, col = divmod(idx, 4)
        x, y = legend_x[col], legend_y[row]
        color = COLORS[variant]
        style = LINE_STYLES[variant]
        ours = variant == "One-sided (Ours)"
        lw = 1.55 if ours else 0.85
        lines.append(rf"\draw[{color},{style},line width={lw}pt] ({x},{y}) -- ({x+17},{y});")
        lines.append(marker_tex(MARKERS[variant], x + 8.5, y, color, ours))
        weight = r"\bfseries" if ours else ""
        lines.append(rf"\node[anchor=west,text={color},font=\fontsize{{5.7}}{{6.3}}\selectfont{weight}] at ({x+21},{y}) {{{DISPLAY[variant]}}};")

    for backbone, (left, bottom, title) in panels.items():
        # Panel background, horizontal grid, and zero line.
        lines.append(rf"\fill[grid!18,rounded corners=2pt] ({left-3},{bottom-3}) rectangle ({left+panel_w+3},{bottom+panel_h+3});")
        for tick in [-6, -3, 0, 3, 6]:
            y = py(bottom, tick)
            style = "ink, line width=0.75pt" if tick == 0 else "grid, line width=0.4pt"
            lines.append(rf"\draw[{style}] ({left:.2f},{y:.2f}) -- ({left+panel_w:.2f},{y:.2f});")
            lines.append(rf"\node[anchor=east,text=muted,font=\fontsize{{5}}{{5.5}}\selectfont] at ({left-3},{y:.2f}) {{{tick:+d}}};")
        for j, dataset in enumerate(DATASETS):
            x = px(left, j)
            lines.append(rf"\draw[grid,line width=0.25pt] ({x:.2f},{bottom:.2f}) -- ({x:.2f},{bottom+panel_h:.2f});")
            lines.append(rf"\node[anchor=east,rotate=35,text=muted,font=\fontsize{{4.8}}{{5.2}}\selectfont] at ({x+2:.2f},{bottom-7}) {{{dataset}}};")
        lines.append(rf"\node[anchor=south,text=ink,font=\bfseries\fontsize{{7}}{{8}}\selectfont] at ({left+panel_w/2:.2f},{bottom+panel_h+5}) {{{title}}};")

        # Plot all seven variants plus the complete one-sided method.
        base = DATA[backbone]["Base"]
        for variant in VARIANTS:
            color = COLORS[variant]
            style = LINE_STYLES[variant]
            ours = variant == "One-sided (Ours)"
            lw = 1.65 if ours else 0.75
            coords = []
            for j, value in enumerate(DATA[backbone][variant]):
                delta = value - base[j]
                coords.append((px(left, j), py(bottom, delta)))
            path = " -- ".join(f"({x:.2f},{y:.2f})" for x, y in coords)
            lines.append(rf"\draw[{color},{style},line width={lw}pt] {path};")
            for x, y in coords:
                lines.append(marker_tex(MARKERS[variant], x, y, color, ours))

    # Shared y-axis descriptions for the left column.
    lines.append(r"\node[rotate=90,anchor=south,text=muted,font=\fontsize{5.4}{6}\selectfont] at (5,227.5) {$\Delta$NDCG@10 (pp)};")
    lines.append(r"\node[rotate=90,anchor=south,text=muted,font=\fontsize{5.4}{6}\selectfont] at (5,77.5) {$\Delta$NDCG@10 (pp)};")
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/figures/rq5_ablation_lines")
    parser.add_argument("--latex", default="pdflatex")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / "rq5_ablation_lines.tex"
    tex_path.write_text(build_tex(), encoding="utf-8")
    subprocess.run(
        [args.latex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=output_dir,
        check=True,
    )
    print(f"PDF: {output_dir / 'rq5_ablation_lines.pdf'}")


if __name__ == "__main__":
    main()
