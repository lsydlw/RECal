#!/usr/bin/env python3
"""Create single-column vector figures for RQ6 sensitivity and efficiency."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def sensitivity_tex() -> str:
    gammas = [1, 4, 8, 12, 24]
    panels = [
        ("LastFM", [11.62, 11.75, 11.82, 11.83, 11.86],
         [10.91, 11.03, 11.09, 11.11, 11.14], 10.8, 12.0, [11.0, 11.4, 11.8]),
        ("DataFoundation", [11.18, 11.18, 11.18, 11.18, 11.18],
         [7.23, 7.89, 8.15, 8.28, 8.39], 7.0, 11.5, [7, 9, 11]),
        ("Online Retail", [4.59, 4.60, 4.60, 4.60, 4.60],
         [4.51, 4.52, 4.52, 4.52, 4.52], 4.50, 4.62, [4.50, 4.55, 4.60]),
    ]
    left, right, height = 38.0, 240.0, 56.0
    bottoms = [201.0, 116.0, 31.0]

    lines = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage{lmodern}",
        r"\usepackage[T1]{fontenc}",
        r"\definecolor{ink}{HTML}{25364A}",
        r"\definecolor{muted}{HTML}{66788A}",
        r"\definecolor{grid}{HTML}{DCE5EE}",
        r"\definecolor{gru}{HTML}{2878B5}",
        r"\definecolor{sas}{HTML}{F39C55}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1pt,y=1pt,font=\sffamily]",
        r"\path[use as bounding box] (0,0) rectangle (248,305);",
        r"\node[anchor=west,text=ink,font=\bfseries\fontsize{8}{9}\selectfont] at (0,296) {Risk-exponent sensitivity};",
        r"\draw[gru,line width=1.3pt] (65,280) -- (82,280);",
        r"\filldraw[fill=white,draw=gru,line width=0.8pt] (73.5,280) circle (1.5pt);",
        r"\node[anchor=west,text=gru,font=\fontsize{5.8}{6.3}\selectfont] at (86,280) {GRU4Rec};",
        r"\draw[sas,line width=1.3pt] (146,280) -- (163,280);",
        r"\filldraw[fill=white,draw=sas,line width=0.8pt] (154.5,280) rectangle +(2.5pt,2.5pt);",
        r"\node[anchor=west,text=sas,font=\fontsize{5.8}{6.3}\selectfont] at (168,280) {SASRec};",
    ]

    def sx(gamma: int) -> float:
        return left + (gamma - 1) / 23.0 * (right - left)

    for idx, (name, gru_vals, sas_vals, ymin, ymax, ticks) in enumerate(panels):
        bottom = bottoms[idx]

        def sy(v: float) -> float:
            return bottom + (v - ymin) / (ymax - ymin) * height

        lines.append(rf"\fill[grid!18,rounded corners=2pt] ({left-3},{bottom-3}) rectangle ({right+3},{bottom+height+3});")
        for tick in ticks:
            y = sy(tick)
            label = f"{tick:.2f}" if ymax - ymin < 0.2 else (f"{tick:.1f}" if ymax - ymin < 2 else f"{tick:g}")
            lines.append(rf"\draw[grid,line width=0.4pt] ({left},{y:.2f}) -- ({right},{y:.2f});")
            lines.append(rf"\node[anchor=east,text=muted,font=\fontsize{{5}}{{5.5}}\selectfont] at ({left-3},{y:.2f}) {{{label}}};")
        default_x = sx(12)
        lines.append(rf"\draw[muted,densely dashed,line width=0.55pt] ({default_x:.2f},{bottom}) -- ({default_x:.2f},{bottom+height});")
        lines.append(rf"\node[anchor=south,text=muted,font=\fontsize{{4.8}}{{5.2}}\selectfont] at ({default_x:.2f},{bottom+height+2}) {{$\gamma=12$}};")
        lines.append(rf"\node[anchor=west,text=ink,font=\bfseries\fontsize{{6.2}}{{7}}\selectfont] at ({left},{bottom+height+9}) {{{name}}};")

        for color, vals, square in [("gru", gru_vals, False), ("sas", sas_vals, True)]:
            coords = [(sx(g), sy(v)) for g, v in zip(gammas, vals)]
            path = " -- ".join(f"({x:.2f},{y:.2f})" for x, y in coords)
            lines.append(rf"\draw[{color},line width=1.25pt] {path};")
            for x, y in coords:
                if square:
                    lines.append(rf"\filldraw[fill=white,draw={color},line width=0.7pt] ({x-1.35:.2f},{y-1.35:.2f}) rectangle ({x+1.35:.2f},{y+1.35:.2f});")
                else:
                    lines.append(rf"\filldraw[fill=white,draw={color},line width=0.7pt] ({x:.2f},{y:.2f}) circle (1.35pt);")
        for gamma in gammas:
            x = sx(gamma)
            lines.append(rf"\node[anchor=north,text=muted,font=\fontsize{{5}}{{5.5}}\selectfont] at ({x:.2f},{bottom-4}) {{{gamma}}};")
        lines.append(rf"\node[rotate=90,anchor=south,text=muted,font=\fontsize{{4.8}}{{5.2}}\selectfont] at (1,{bottom+height/2:.2f}) {{NDCG@10 (\%)}};")

    lines.append(r"\node[anchor=north,text=muted,font=\fontsize{5.4}{6}\selectfont] at (139,12) {Risk exponent $\gamma$};")
    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines) + "\n"


def efficiency_tex() -> str:
    datasets = ["ML-100K", "Online Retail", "LastFM"]
    panels = [
        ("Parameter increase (\\%)", [3.23, 1.18, 0.91], [1.53, 0.84, 0.69], 3.6, [0, 1, 2, 3]),
        ("Latency increase (ms/user)", [0.084, 0.085, 0.082], [0.089, 0.081, 0.083], 0.10, [0, 0.05, 0.10]),
        ("Peak-memory increase (MiB)", [8.21, 8.19, 8.16], [0.09, 0.07, 0.04], 9.0, [0, 4, 8]),
    ]
    left, right, height = 36.0, 240.0, 50.0
    bottoms = [181.0, 103.0, 25.0]
    group_x = [72.0, 139.0, 206.0]
    bar_w = 9.0
    gap = 1.5

    lines = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage{lmodern}",
        r"\usepackage[T1]{fontenc}",
        r"\definecolor{ink}{HTML}{25364A}",
        r"\definecolor{muted}{HTML}{66788A}",
        r"\definecolor{grid}{HTML}{DCE5EE}",
        r"\definecolor{gru}{HTML}{6AADE4}",
        r"\definecolor{gruDark}{HTML}{2878B5}",
        r"\definecolor{sas}{HTML}{F7B477}",
        r"\definecolor{sasDark}{HTML}{DB7D2F}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1pt,y=1pt,font=\sffamily]",
        r"\path[use as bounding box] (0,0) rectangle (248,283);",
        r"\node[anchor=west,text=ink,font=\bfseries\fontsize{8}{9}\selectfont] at (0,274) {Inference overhead};",
        r"\fill[gru] (62,257) rectangle (70,263);",
        r"\node[anchor=west,text=gruDark,font=\fontsize{5.8}{6.3}\selectfont] at (75,260) {GRU4Rec};",
        r"\fill[sas] (145,257) rectangle (153,263);",
        r"\node[anchor=west,text=sasDark,font=\fontsize{5.8}{6.3}\selectfont] at (158,260) {SASRec};",
    ]

    for idx, (title, gru_vals, sas_vals, ymax, ticks) in enumerate(panels):
        bottom = bottoms[idx]

        def sy(v: float) -> float:
            return bottom + v / ymax * height

        lines.append(rf"\fill[grid!18,rounded corners=2pt] ({left-3},{bottom-3}) rectangle ({right+3},{bottom+height+3});")
        for tick in ticks:
            y = sy(tick)
            if ymax <= 0.1:
                label = f"{tick:.2f}"
            else:
                label = f"{tick:g}"
            lines.append(rf"\draw[grid,line width=0.4pt] ({left},{y:.2f}) -- ({right},{y:.2f});")
            lines.append(rf"\node[anchor=east,text=muted,font=\fontsize{{5}}{{5.5}}\selectfont] at ({left-3},{y:.2f}) {{{label}}};")
        lines.append(rf"\node[anchor=west,text=ink,font=\bfseries\fontsize{{6}}{{6.7}}\selectfont] at ({left},{bottom+height+8}) {{{title}}};")
        for j, center in enumerate(group_x):
            gx0 = center - gap / 2 - bar_w
            sx0 = center + gap / 2
            gy = sy(gru_vals[j])
            syv = sy(sas_vals[j])
            lines.append(rf"\fill[gru,rounded corners=0.7pt] ({gx0:.2f},{bottom}) rectangle ({gx0+bar_w:.2f},{gy:.2f});")
            lines.append(rf"\fill[sas,rounded corners=0.7pt] ({sx0:.2f},{bottom}) rectangle ({sx0+bar_w:.2f},{syv:.2f});")
            fmt = ".3f" if ymax <= 0.1 else ".2f"
            # Similar latency values place adjacent labels too close; stagger
            # that panel vertically while retaining exact values.
            gru_label_offset = 7.0 if idx == 1 else 1.0
            sas_label_offset = 1.0
            lines.append(rf"\node[anchor=south,text=gruDark,font=\fontsize{{4.5}}{{5}}\selectfont] at ({gx0+bar_w/2:.2f},{gy+gru_label_offset:.2f}) {{{format(gru_vals[j], fmt)}}};")
            lines.append(rf"\node[anchor=south,text=sasDark,font=\fontsize{{4.5}}{{5}}\selectfont] at ({sx0+bar_w/2:.2f},{syv+sas_label_offset:.2f}) {{{format(sas_vals[j], fmt)}}};")
            if idx == 2:
                lines.append(rf"\node[anchor=north,text=muted,font=\fontsize{{5}}{{5.5}}\selectfont] at ({center:.2f},{bottom-5}) {{{datasets[j]}}};")

    lines.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(lines) + "\n"


def compile_tex(output_dir: Path, stem: str, source: str, latex: str) -> None:
    tex_path = output_dir / f"{stem}.tex"
    tex_path.write_text(source, encoding="utf-8")
    subprocess.run(
        [latex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
        cwd=output_dir,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/figures/rq6")
    parser.add_argument("--latex", default="pdflatex")
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compile_tex(output_dir, "rq6_sensitivity", sensitivity_tex(), args.latex)
    compile_tex(output_dir, "rq6_efficiency", efficiency_tex(), args.latex)
    print(f"Sensitivity: {output_dir / 'rq6_sensitivity.pdf'}")
    print(f"Efficiency: {output_dir / 'rq6_efficiency.pdf'}")


if __name__ == "__main__":
    main()
