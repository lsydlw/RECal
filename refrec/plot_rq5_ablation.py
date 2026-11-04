#!/usr/bin/env python3
"""Generate the RQ5 ablation summary as a publication-ready vector PDF.

The figure intentionally excludes the 'full closed-form projection' variant.
It summarizes absolute NDCG@10 gains (percentage points) over the frozen Base
across 8 datasets x 4 backbones = 32 settings.

Only Python's standard library and a LaTeX installation are required.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from pathlib import Path


DATASETS = ["ML-100K", "DF", "BM", "LastFM", "AM", "FS", "OR", "BN"]

DATA = {
    "BERT4Rec": {
        "Base": [6.20, 8.87, 2.86, 8.46, 2.69, 31.78, 9.01, 63.72],
        "Fixed boost": [3.19, 9.47, 2.84, 9.76, 2.45, 32.29, 9.29, 66.29],
        "Fixed global $g$": [6.96, 8.59, 2.86, 9.46, 2.71, 31.19, 9.26, 65.67],
        "Entropy conf.": [8.97, 8.37, 2.92, 9.32, 2.86, 30.84, 9.44, 66.98],
        "$q$ only": [8.97, 7.36, 2.92, 7.93, 2.86, 32.22, 9.54, 66.29],
        "Global repeat": [8.97, 6.17, 2.92, 9.58, 2.86, 28.53, 8.94, 67.23],
        "w/o seq. stats": [8.97, 8.97, 2.92, 10.33, 2.86, 31.89, 9.32, 67.83],
        "Bidirectional": [8.97, 8.85, 2.92, 10.25, 2.86, 31.90, 9.81, 68.24],
        "One-sided (Ours)": [8.97, 9.02, 2.92, 11.38, 2.86, 31.90, 9.95, 68.30],
    },
    "GRU4Rec": {
        "Base": [6.46, 8.86, 10.06, 8.17, 2.44, 31.57, 7.96, 61.27],
        "Fixed boost": [4.79, 10.09, 10.00, 9.14, 2.31, 32.36, 8.37, 64.76],
        "Fixed global $g$": [6.73, 10.67, 10.06, 9.64, 2.45, 31.64, 8.32, 65.88],
        "Entropy conf.": [8.13, 11.02, 10.11, 9.78, 2.54, 31.69, 8.42, 66.77],
        "$q$ only": [8.13, 5.38, 10.11, 7.91, 2.54, 32.26, 8.41, 65.22],
        "Global repeat": [8.13, 11.69, 10.11, 10.74, 2.54, 31.70, 8.21, 67.42],
        "w/o seq. stats": [8.13, 11.75, 10.11, 11.29, 2.54, 32.13, 8.54, 67.42],
        "Bidirectional": [8.13, 9.33, 10.11, 10.13, 2.54, 32.16, 9.17, 67.47],
        "One-sided (Ours)": [8.13, 11.83, 10.11, 11.70, 2.54, 32.20, 9.27, 67.58],
    },
    "RepeatNet": {
        "Base": [3.66, 12.74, 0.03, 16.32, 0.13, 36.21, 6.84, 76.51],
        "Fixed boost": [1.55, 12.79, 0.03, 16.32, 0.12, 36.21, 6.53, 76.48],
        "Fixed global $g$": [4.38, 12.68, 0.04, 16.19, 0.13, 36.21, 7.05, 76.50],
        "Entropy conf.": [9.61, 12.25, 0.08, 15.56, 0.17, 36.21, 8.58, 76.50],
        "$q$ only": [9.61, 12.27, 0.07, 16.18, 0.17, 36.21, 8.87, 76.47],
        "Global repeat": [9.61, 7.36, 0.08, 13.54, 0.17, 36.21, 8.20, 76.36],
        "w/o seq. stats": [9.61, 12.70, 0.08, 15.90, 0.17, 36.21, 6.82, 76.50],
        "Bidirectional": [9.61, 12.71, 0.08, 16.08, 0.17, 36.21, 7.85, 76.53],
        "One-sided (Ours)": [9.61, 12.71, 0.08, 16.08, 0.17, 36.21, 7.85, 76.53],
    },
    "SASRec": {
        "Base": [5.37, 7.89, 4.78, 9.05, 2.47, 32.21, 8.51, 68.36],
        "Fixed boost": [3.71, 8.36, 4.71, 10.14, 2.27, 32.72, 8.82, 69.88],
        "Fixed global $g$": [5.74, 7.43, 4.79, 10.15, 2.49, 31.48, 8.76, 69.69],
        "Entropy conf.": [7.63, 7.43, 4.83, 10.09, 2.71, 31.12, 8.89, 70.43],
        "$q$ only": [7.63, 7.02, 4.83, 8.34, 2.71, 32.68, 9.07, 69.87],
        "Global repeat": [7.63, 5.67, 4.83, 10.75, 2.71, 28.63, 8.42, 70.56],
        "w/o seq. stats": [7.63, 8.04, 4.83, 11.55, 2.71, 32.30, 8.85, 70.97],
        "Bidirectional": [7.63, 7.86, 4.83, 10.72, 2.71, 32.29, 9.36, 71.27],
        "One-sided (Ours)": [7.63, 8.17, 4.83, 12.21, 2.71, 32.30, 9.44, 71.31],
    },
}

VARIANTS = [
    "Fixed boost",
    "Fixed global $g$",
    "Entropy conf.",
    "$q$ only",
    "Global repeat",
    "w/o seq. stats",
    "Bidirectional",
    "One-sided (Ours)",
]


def quantile(values: list[float], p: float) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize():
    deltas = {variant: [] for variant in VARIANTS}
    tagged = {variant: [] for variant in VARIANTS}
    best_counts = {variant: 0 for variant in VARIANTS}
    for backbone, table in DATA.items():
        for j, dataset in enumerate(DATASETS):
            candidates = [table["Base"][j]] + [table[v][j] for v in VARIANTS]
            maximum = max(candidates)
            for variant in VARIANTS:
                delta = table[variant][j] - table["Base"][j]
                deltas[variant].append(delta)
                tagged[variant].append((backbone, dataset, delta))
                if abs(table[variant][j] - maximum) < 1e-12:
                    best_counts[variant] += 1
    rows = []
    for variant in VARIANTS:
        xs = deltas[variant]
        rows.append({
            "variant": variant,
            "mean": sum(xs) / len(xs),
            "median": quantile(xs, 0.5),
            "q1": quantile(xs, 0.25),
            "q3": quantile(xs, 0.75),
            "minimum": min(xs),
            "maximum": max(xs),
            "improved": sum(x > 1e-12 for x in xs),
            "best_or_tied": best_counts[variant],
        })
    return rows, tagged


def esc(text: str) -> str:
    return text.replace("%", r"\%").replace("&", r"\&")


def build_tex(rows, tagged) -> str:
    width = 248.0
    left = 91.0
    right = 242.0
    x_min, x_max = -6.5, 6.8

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def bx(count: int) -> float:
        return left + count / 32.0 * (right - left)

    lines = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage{lmodern}",
        r"\usepackage[T1]{fontenc}",
        r"\definecolor{ink}{HTML}{243447}",
        r"\definecolor{muted}{HTML}{64748B}",
        r"\definecolor{grid}{HTML}{DCE4EC}",
        r"\definecolor{soft}{HTML}{E8EEF4}",
        r"\definecolor{blue}{HTML}{62A9E8}",
        r"\definecolor{blueDark}{HTML}{2474B5}",
        r"\definecolor{orange}{HTML}{F3A65A}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1pt,y=1pt,font=\sffamily]",
        rf"\path[use as bounding box] (0,0) rectangle ({width:.1f},520);",
        r"\node[anchor=west,text=ink,font=\sffamily\bfseries\fontsize{8}{9}\selectfont] at (0,509) {(a) NDCG@10 gain over Base};",
        r"\node[anchor=west,text=muted,font=\sffamily\fontsize{5.7}{6.5}\selectfont] at (0,498) {8 datasets $\times$ 4 backbones; absolute percentage-point change};",
    ]

    # Panel (a) grid and axis.
    axis_y = 247
    for tick in [-6, -3, 0, 3, 6]:
        x = sx(tick)
        style = "blueDark, line width=0.8pt" if tick == 0 else "grid, line width=0.35pt"
        lines.append(rf"\draw[{style}] ({x:.2f},{axis_y}) -- ({x:.2f},482);")
        lines.append(rf"\node[anchor=north,text=muted,font=\fontsize{{5.5}}{{6}}\selectfont] at ({x:.2f},{axis_y-2}) {{{tick:+d}}};")
    lines.append(rf"\node[anchor=north,text=muted,font=\fontsize{{5.5}}{{6}}\selectfont] at ({(left+right)/2:.2f},{axis_y-12}) {{$\Delta$NDCG@10 (pp)}};")

    # Deterministic tiny jitter; points stay legible at one-column width.
    jitter = [-4.2, -3.1, -2.0, -0.9, 0.3, 1.5, 2.7, 3.9]
    row_ys = [466 - 27 * i for i in range(len(VARIANTS))]
    row_by_name = {row["variant"]: row for row in rows}
    for idx, variant in enumerate(VARIANTS):
        y = row_ys[idx]
        row = row_by_name[variant]
        label_color = "blueDark" if variant == "One-sided (Ours)" else "ink"
        label_weight = r"\bfseries" if variant == "One-sided (Ours)" else ""
        if variant == "One-sided (Ours)":
            lines.append(rf"\fill[blue!10,rounded corners=2pt] (0,{y-11}) rectangle ({width},{y+11});")
        lines.append(rf"\node[anchor=east,text={label_color},font=\fontsize{{5.8}}{{6.4}}\selectfont{label_weight}] at ({left-5},{y}) {{{esc(variant)}}};")
        # Raw points, grouped by backbone and jittered by dataset.
        for point_index, (_, _, delta) in enumerate(tagged[variant]):
            dataset_index = point_index % 8
            backbone_index = point_index // 8
            offset = jitter[dataset_index] * 0.55 + (backbone_index - 1.5) * 0.45
            lines.append(rf"\fill[muted,opacity=0.34] ({sx(delta):.2f},{y+offset:.2f}) circle (0.72pt);")
        # Whisker, IQR box, median, and mean diamond.
        fill = "blue!32" if variant == "One-sided (Ours)" else ("orange!30" if variant == "Bidirectional" else "soft")
        outline = "blueDark" if variant == "One-sided (Ours)" else "muted"
        lines.append(rf"\draw[{outline},line width=0.55pt] ({sx(row['minimum']):.2f},{y}) -- ({sx(row['maximum']):.2f},{y});")
        lines.append(rf"\draw[{outline},line width=0.55pt] ({sx(row['minimum']):.2f},{y-2.6}) -- ({sx(row['minimum']):.2f},{y+2.6});")
        lines.append(rf"\draw[{outline},line width=0.55pt] ({sx(row['maximum']):.2f},{y-2.6}) -- ({sx(row['maximum']):.2f},{y+2.6});")
        lines.append(rf"\filldraw[fill={fill},draw={outline},line width=0.55pt] ({sx(row['q1']):.2f},{y-4.0}) rectangle ({sx(row['q3']):.2f},{y+4.0});")
        lines.append(rf"\draw[ink,line width=0.85pt] ({sx(row['median']):.2f},{y-4.1}) -- ({sx(row['median']):.2f},{y+4.1});")
        mean_x = sx(row["mean"])
        lines.append(rf"\filldraw[fill=white,draw={outline},line width=0.7pt] ({mean_x:.2f},{y+4.2}) -- ({mean_x+2.1:.2f},{y}) -- ({mean_x:.2f},{y-4.2}) -- ({mean_x-2.1:.2f},{y}) -- cycle;")

    lines.extend([
        r"\fill[muted,opacity=0.34] (98,226) circle (0.8pt);",
        r"\node[anchor=west,text=muted,font=\fontsize{5.3}{6}\selectfont] at (102,226) {one setting};",
        r"\filldraw[fill=white,draw=muted,line width=0.6pt] (148,230) -- (150,226) -- (148,222) -- (146,226) -- cycle;",
        r"\node[anchor=west,text=muted,font=\fontsize{5.3}{6}\selectfont] at (154,226) {mean};",
        r"\node[anchor=west,text=ink,font=\sffamily\bfseries\fontsize{8}{9}\selectfont] at (0,202) {(b) Cross-setting robustness};",
        r"\node[anchor=west,text=muted,font=\sffamily\fontsize{5.7}{6.5}\selectfont] at (0,191) {Counts out of 32 settings after excluding full projection};",
    ])

    # Panel (b) x grid, rows, paired horizontal bars.
    b_axis_y = 11
    for tick in [0, 8, 16, 24, 32]:
        x = bx(tick)
        lines.append(rf"\draw[grid,line width=0.35pt] ({x:.2f},{b_axis_y+13}) -- ({x:.2f},178);")
        lines.append(rf"\node[anchor=north,text=muted,font=\fontsize{{5.5}}{{6}}\selectfont] at ({x:.2f},{b_axis_y+11}) {{{tick}}};")
    b_row_ys = [169 - 18 * i for i in range(len(VARIANTS))]
    for idx, variant in enumerate(VARIANTS):
        y = b_row_ys[idx]
        row = row_by_name[variant]
        label_color = "blueDark" if variant == "One-sided (Ours)" else "ink"
        label_weight = r"\bfseries" if variant == "One-sided (Ours)" else ""
        if variant == "One-sided (Ours)":
            lines.append(rf"\fill[blue!10,rounded corners=2pt] (0,{y-8}) rectangle ({width},{y+8});")
        lines.append(rf"\node[anchor=east,text={label_color},font=\fontsize{{5.8}}{{6.4}}\selectfont{label_weight}] at ({left-5},{y}) {{{esc(variant)}}};")
        improved_x = bx(row["improved"])
        best_x = bx(row["best_or_tied"])
        lines.append(rf"\fill[blue!48,rounded corners=0.8pt] ({left:.2f},{y+0.6}) rectangle ({improved_x:.2f},{y+5.2});")
        lines.append(rf"\fill[blueDark!88,rounded corners=0.8pt] ({left:.2f},{y-5.2}) rectangle ({best_x:.2f},{y-0.6});")
        lines.append(rf"\node[anchor=west,text=ink,font=\fontsize{{5.1}}{{5.5}}\selectfont] at ({min(improved_x+2,right-1):.2f},{y+2.9}) {{{row['improved']}}};")
        lines.append(rf"\node[anchor=west,text=ink,font=\fontsize{{5.1}}{{5.5}}\selectfont] at ({min(best_x+2,right-1):.2f},{y-2.9}) {{{row['best_or_tied']}}};")
    lines.extend([
        r"\fill[blue!48] (91,8) rectangle (97,12);",
        r"\node[anchor=west,text=muted,font=\fontsize{5.3}{6}\selectfont] at (100,10) {improved};",
        r"\fill[blueDark!88] (147,8) rectangle (153,12);",
        r"\node[anchor=west,text=muted,font=\fontsize{5.3}{6}\selectfont] at (156,10) {best / tied};",
        r"\end{tikzpicture}",
        r"\end{document}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/figures/rq5_ablation")
    parser.add_argument("--latex", default="pdflatex")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, tagged = summarize()
    tex_path = output_dir / "rq5_ablation_summary.tex"
    tex_path.write_text(build_tex(rows, tagged), encoding="utf-8")

    csv_path = output_dir / "rq5_ablation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    subprocess.run(
        [args.latex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=output_dir,
        check=True,
    )
    print(f"PDF: {output_dir / 'rq5_ablation_summary.pdf'}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
