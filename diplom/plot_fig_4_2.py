#!/usr/bin/env python3
"""
Рисунок 4.2: столбчатая диаграмма RMSE по строкам benchmark в result.txt
и отрезки порога threshold из того же протокола (совпадает с gp_settings.json).

Ось Y — логарифмическая: для exponential.xlsx абсолютный RMSE велик при малой
относительной ошибке (см. табл. 4.2 и §4.2 текста ВКР).
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def parse_benchmark_row_line(line: str) -> tuple[str, str, str, str, str] | None:
    s = line.strip()
    if ": RMSE=" not in s or ", rel=" not in s or ", threshold=" not in s or "status=" not in s:
        return None
    try:
        name, rest = s.split(": RMSE=", 1)
        rmse, rest = rest.split(", rel=", 1)
        rel, rest = rest.split(", threshold=", 1)
        threshold, rest = rest.split(", rel≤", 1)
        status = rest.split("status=", 1)[1].strip()
        return (name.strip(), rmse.strip(), rel.strip(), threshold.strip(), status)
    except ValueError:
        return None


def parse_benchmark_rows(result_path: Path) -> list[tuple[str, str, str, str, str]]:
    if not result_path.exists():
        return []
    rows: list[tuple[str, str, str, str, str]] = []
    for line in result_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_benchmark_row_line(line)
        if parsed:
            rows.append(parsed)
    return rows


def to_float(s: str) -> float:
    x = s.replace(",", ".").strip()
    return float(x)


def plot_benchmark(out_path: Path, result_path: Path) -> None:
    rows = parse_benchmark_rows(result_path)
    if not rows:
        raise SystemExit(f"Нет строк benchmark в {result_path}")

    names = [r[0] for r in rows]
    rmse = np.array([max(to_float(r[1]), 1e-30) for r in rows])
    thr = np.array([max(to_float(r[3]), 1e-30) for r in rows])
    status = [r[4] for r in rows]
    x = np.arange(len(names))

    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=150)
    bars = ax.bar(
        x,
        rmse,
        width=0.62,
        color="#4a7ebb",
        edgecolor="#2f5582",
        linewidth=0.6,
        label="RMSE (протокол)",
        zorder=3,
    )

    for i in range(len(names)):
        ax.hlines(
            thr[i],
            i - 0.31,
            i + 0.31,
            colors="#c62828",
            linewidth=2.8,
            zorder=5,
        )
        if not re.search(r"\bPASS\b", status[i], re.I):
            bars[i].set_color("#b0bec5")
            bars[i].set_edgecolor("#546e7a")

    ax.plot([], [], color="#c62828", linewidth=2.8, label="Порог RMSE")

    ax.set_yscale("log")
    ax.set_ylabel("RMSE и порог (логарифмическая ось)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=48, ha="right", fontsize=8.5)
    ax.grid(True, which="major", axis="y", alpha=0.35, linestyle="--")
    ax.grid(True, which="minor", axis="y", alpha=0.15, linestyle=":")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.92)

    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=ROOT / "kursovaia" / "kursovaia" / "result.txt",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "diagrams" / "fig_4_2_rmse_vs_threshold.png",
    )
    args = ap.parse_args()
    plot_benchmark(args.output, args.result)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
