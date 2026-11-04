#!/usr/bin/env python3
"""Generate the RECal RQ3 comparison figure as a vector PDF.

This script has no third-party Python dependency.  It writes a standalone
TikZ source and invokes the server's pdflatex executable.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DATASETS = [
    "Bookmarks",
    "LastFM",
    "Amazon Music",
    "Baidu News v2",
    "DataFoundation",
    "Foursquare",
    "ML-100K",
    "Online Retail",
]

METHODS = ["Hard Mask", "CalRec", "PRM", "ReCODE-style", "RECal"]
BASE = [4.78, 9.05, 2.47, 68.36, 7.89, 32.21, 5.37, 8.51]
SCORES = [
    [4.83, 4.78, 4.73, 4.83, 4.83],
    [1.37, 9.21, 9.31, 9.54, 12.21],
    [2.71, 2.46, 2.56, 2.71, 2.71],
    [1.58, 68.37, 67.84, 66.39, 71.31],
    [0.43, 8.26, 7.29, 7.02, 8.17],
    [0.90, 32.20, 32.34, 33.13, 32.30],
    [7.63, 5.40, 7.90, 7.63, 7.63],
    [5.81, 8.59, 9.14, 9.40, 9.44],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/alternative_postprocessing_heatmap.pdf"),
    )
    return parser.parse_args()


def relative_changes() -> list[list[float]]:
    return [
        [(score - base) / base * 100.0 for score in row]
        for base, row in zip(BASE, SCORES)
    ]


def format_change(value: float) -> str:
    if abs(value) < 0.05:
        return "0.0"
    return f"{value:+.1f}"


def cell_style(value: float) -> tuple[str, str]:
    strength = min(abs(value) / 20.0, 1.0)
    mix = int(round(12 + 73 * strength))
    if value > 0.05:
        fill = f"pos!{mix}!white"
    elif value < -0.05:
        fill = f"neg!{mix}!white"
    else:
        fill = "gray!5"
    text = "white" if abs(value) >= 11.0 else "ink"
    return fill, text


def tikz_source(relative: list[list[float]], improved: list[int]) -> str:
    lines = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{helvet}",
        r"\renewcommand{\familydefault}{\sfdefault}",
        r"\usepackage{xcolor}",
        r"\definecolor{pos}{HTML}{2B7DBC}",
        r"\definecolor{neg}{HTML}{E07A3F}",
        r"\definecolor{recal}{HTML}{087F8C}",
        r"\definecolor{ink}{HTML}{1F2937}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm]",
        r"\node[anchor=west,font=\fontsize{8.3}{9.4}\selectfont\bfseries,text=ink] at (0,0) {(a) Relative NDCG@10 change vs. Base (\%)};",
        r"\node[anchor=west,font=\fontsize{8.3}{9.4}\selectfont\bfseries,text=ink] at (12.05,0) {(b) Improved over Base};",
    ]

    cell_x = 3.25
    cell_w = 1.62
    row_top = -0.90
    row_h = 0.58

    for col, method in enumerate(METHODS):
        x = cell_x + (col + 0.5) * cell_w
        weight = r"\bfseries" if method == "RECal" else ""
        lines.append(
            rf"\node[anchor=south,font=\fontsize{{7.0}}{{8.1}}\selectfont{weight},text=ink] "
            rf"at ({x:.3f},{row_top + 0.12:.3f}) {{{method}}};"
        )

    for row, dataset in enumerate(DATASETS):
        y_top = row_top - row * row_h
        y_bottom = y_top - row_h
        y_mid = (y_top + y_bottom) / 2
        lines.append(
            rf"\node[anchor=east,font=\fontsize{{7.1}}{{8.2}}\selectfont,text=ink] "
            rf"at ({cell_x - 0.12:.3f},{y_mid:.3f}) {{{dataset}}};"
        )
        for col, value in enumerate(relative[row]):
            x_left = cell_x + col * cell_w
            x_right = x_left + cell_w
            fill, text = cell_style(value)
            lines.append(
                rf"\path[fill={fill},draw=white,line width=0.55pt] "
                rf"({x_left:.3f},{y_top:.3f}) rectangle ({x_right:.3f},{y_bottom:.3f});"
            )
            weight = r"\bfseries" if col == len(METHODS) - 1 else ""
            lines.append(
                rf"\node[font=\fontsize{{7.0}}{{7.8}}\selectfont{weight},text={text}] "
                rf"at ({(x_left + x_right) / 2:.3f},{y_mid:.3f}) {{{format_change(value)}}};"
            )

    last_left = cell_x + (len(METHODS) - 1) * cell_w
    bottom = row_top - len(DATASETS) * row_h
    lines.append(
        rf"\draw[recal,line width=1.25pt] ({last_left:.3f},{row_top:.3f}) "
        rf"rectangle ({last_left + cell_w:.3f},{bottom:.3f});"
    )

    # Compact diverging legend; the numeric annotations retain exact values.
    legend_y = bottom - 0.55
    legend_x = cell_x + 0.45
    legend_w = 1.10
    legend_colors = ["neg!85!white", "neg!48!white", "gray!5", "pos!48!white", "pos!85!white"]
    legend_labels = [r"$\leq-20$", "-10", "0", "+10", r"$\geq+20$"]
    for idx, (color, label) in enumerate(zip(legend_colors, legend_labels)):
        x0 = legend_x + idx * legend_w
        lines.append(
            rf"\path[fill={color},draw=white,line width=0.4pt] "
            rf"({x0:.3f},{legend_y:.3f}) rectangle ({x0 + legend_w:.3f},{legend_y - 0.25:.3f});"
        )
        lines.append(
            rf"\node[anchor=north,font=\fontsize{{6.2}}{{7.0}}\selectfont,text=ink] "
            rf"at ({x0 + legend_w / 2:.3f},{legend_y - 0.29:.3f}) {{{label}}};"
        )
    lines.append(
        rf"\node[anchor=east,font=\fontsize{{6.3}}{{7.2}}\selectfont,text=ink] "
        rf"at ({legend_x - 0.15:.3f},{legend_y - 0.12:.3f}) {{Change}};"
    )

    # Right-side summary bars.
    label_x = 13.52
    bar_x = 13.72
    bar_width = 2.55
    count_y0 = -1.18
    count_step = 0.78
    for idx, (method, count) in enumerate(zip(METHODS, improved)):
        y = count_y0 - idx * count_step
        fill = "recal" if method == "RECal" else "gray!50"
        weight = r"\bfseries" if method == "RECal" else ""
        lines.append(
            rf"\node[anchor=east,font=\fontsize{{7.0}}{{8.0}}\selectfont{weight},text=ink] "
            rf"at ({label_x:.3f},{y:.3f}) {{{method}}};"
        )
        lines.append(
            rf"\path[fill=gray!12,rounded corners=1.1pt] "
            rf"({bar_x:.3f},{y + 0.17:.3f}) rectangle ({bar_x + bar_width:.3f},{y - 0.17:.3f});"
        )
        width = bar_width * count / 8.0
        lines.append(
            rf"\path[fill={fill},rounded corners=1.1pt] "
            rf"({bar_x:.3f},{y + 0.17:.3f}) rectangle ({bar_x + width:.3f},{y - 0.17:.3f});"
        )
        lines.append(
            rf"\node[anchor=west,font=\fontsize{{7.0}}{{8.0}}\selectfont{weight},text=ink] "
            rf"at ({bar_x + bar_width + 0.13:.3f},{y:.3f}) {{{count}/8}};"
        )

    lines.extend(
        [
            r"\node[anchor=west,font=\fontsize{6.3}{7.2}\selectfont,text=ink] at (12.05,-5.05) {Same frozen SASRec checkpoint};",
            r"\node[anchor=west,font=\fontsize{6.3}{7.2}\selectfont,text=ink] at (12.05,-5.34) {Blue: gain; orange: degradation};",
            r"\end{tikzpicture}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    relative = relative_changes()
    improved = [
        sum(row[col] > base + 1e-12 for row, base in zip(SCORES, BASE))
        for col in range(len(METHODS))
    ]

    tex_path = output.with_suffix(".tex")
    tex_path.write_text(tikz_source(relative, improved), encoding="utf-8")
    subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output.parent}",
            str(tex_path),
        ],
        check=True,
    )

    generated = output.parent / f"{tex_path.stem}.pdf"
    if generated != output:
        generated.replace(output)
    for suffix in (".aux", ".log"):
        candidate = output.parent / f"{tex_path.stem}{suffix}"
        candidate.unlink(missing_ok=True)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
