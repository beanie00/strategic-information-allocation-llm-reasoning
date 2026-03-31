"""Plot Section 3.4.3 results with a single merged control panel.

The script reads summary JSON files produced by eval_section_3_4_3.py and
renders one chart per model with the unified control order:
baseline / suppress / insert_wait / induce_1shot / induce_2shot.

By default, the plot targets DeepSeek-R1-Distill-Qwen-1.5B so the plotting
setup matches the lightweight runner configuration in run_section_3_4_3.sh.
Missing settings are shown as placeholders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
]

COMBINED_ORDER = ["baseline", "suppress", "insert_wait", "induce_1shot", "induce_2shot"]

COMBINED_LABELS = {
    "baseline": "Baseline",
    "suppress": "Suppress",
    "insert_wait": "Insert\nWait",
    "induce_1shot": "1-shot\nExample",
    "induce_2shot": "2-shot\nExample",
}

COLOR_MAP = {
    "baseline": "#A7A9AC",
    "suppress": "#D16D6A",
    "insert_wait": "#E195A6",
    "induce_1shot": "#6FAF9A",
    "induce_2shot": "#2E7D66",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=os.path.join(SCRIPT_DIR, "outputs_section_3_4_3"),
        type=str,
        help="Directory containing summary JSONs from eval_section_3_4_3.py",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(SCRIPT_DIR, "plots", "figure6_section_3_4_3.png"),
        type=str,
        help="Output image path",
    )
    parser.add_argument("--data_name", default=None, type=str, help="Optional dataset filter")
    parser.add_argument("--split", default=None, type=str, help="Optional split filter")
    parser.add_argument("--start_idx", default=None, type=int, help="Optional start_idx filter")
    parser.add_argument("--end_idx", default=None, type=int, help="Optional end_idx filter")
    parser.add_argument(
        "--deepseek_models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Deprecated compatibility option; merged with --qwen_models into one ordered model list",
    )
    parser.add_argument(
        "--qwen_models",
        nargs="*",
        default=[],
        help="Deprecated compatibility option; merged with --deepseek_models into one ordered model list",
    )
    parser.add_argument("--dpi", default=200, type=int)
    parser.add_argument("--title", default="Section 3.4.3 Test-Time Control Results", type=str)
    return parser.parse_args()


def model_display_name(model_name: str) -> str:
    short = model_name.split("/")[-1]
    if short.startswith("DeepSeek-R1-Distill-Qwen-"):
        size = short.split("-")[-1]
        return f"DeepSeek-{size}"
    return short


def summary_control_key(summary: Dict) -> str:
    if summary["control_mode"] == "induce":
        return f'induce_{summary["induction_shots"]}shot'
    return summary["control_mode"]


def load_summaries(output_dir: str) -> List[Dict]:
    summaries = []
    for path in sorted(Path(output_dir).rglob("*_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["_path"] = str(path)
        summaries.append(summary)
    return summaries


def filter_summaries(summaries: Iterable[Dict], args) -> List[Dict]:
    filtered = []
    for summary in summaries:
        if args.data_name is not None and summary.get("data_name") != args.data_name:
            continue
        if args.split is not None and summary.get("split") != args.split:
            continue
        if args.start_idx is not None and summary.get("start_idx") != args.start_idx:
            continue
        if args.end_idx is not None and summary.get("end_idx") != args.end_idx:
            continue
        filtered.append(summary)
    return filtered


def build_lookup(summaries: Iterable[Dict]) -> Dict[Tuple[str, str], Dict]:
    lookup: Dict[Tuple[str, str], Dict] = {}
    for summary in summaries:
        key = (summary["model_name_or_path"], summary_control_key(summary))
        lookup[key] = summary
    return lookup


def merged_model_list(args) -> List[str]:
    ordered_models: List[str] = []
    for model_name in list(args.deepseek_models) + list(args.qwen_models):
        if model_name not in ordered_models:
            ordered_models.append(model_name)
    return ordered_models or list(DEFAULT_MODELS)


def pick_ylim(values: List[float]) -> float:
    present = [value for value in values if value is not None]
    if not present:
        return 1.0
    max_val = max(present)
    if max_val <= 0.10:
        return 0.15
    if max_val <= 0.35:
        return 0.40
    if max_val <= 0.65:
        return 0.75
    if max_val <= 0.85:
        return 1.00
    return min(1.05, math.ceil((max_val + 0.05) * 10) / 10)


def percent_formatter(_: float, pos: int) -> str:
    del pos
    return f"{_ * 100:.0f}"


def annotate_bar(ax, x: float, value: float):
    ax.text(
        x,
        value + max(ax.get_ylim()[1] * 0.02, 0.01),
        f"{value * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def plot_model_panel(ax, model_name: str, control_order: List[str], labels: Dict[str, str], lookup: Dict[Tuple[str, str], Dict]):
    values = []
    for control in control_order:
        summary = lookup.get((model_name, control))
        values.append(None if summary is None else summary["accuracy"])

    x = np.arange(len(control_order))
    y = [0.0 if value is None else value for value in values]
    colors = [COLOR_MAP[control] for control in control_order]
    bars = ax.bar(x, y, width=0.68, color=colors, edgecolor="#444444", linewidth=1.0)

    ymax = pick_ylim(values)
    ax.set_ylim(0, ymax)
    ax.set_xticks(x, [labels[control] for control in control_order], fontsize=10)
    ax.set_title(model_display_name(model_name), fontsize=13, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)

    for idx, (bar, value) in enumerate(zip(bars, values)):
        if value is None:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ymax * 0.08,
                "missing",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#777777",
                rotation=90,
            )
            continue
        annotate_bar(ax, idx, value)


def plot_figure(args, lookup: Dict[Tuple[str, str], Dict]):
    models = merged_model_list(args)
    num_models = max(1, len(models))

    fig, axes = plt.subplots(1, num_models, figsize=(4.6 * num_models + 0.8, 5.8), squeeze=False)
    axes = list(axes[0])

    for ax, model_name in zip(axes, models):
        plot_model_panel(ax, model_name, COMBINED_ORDER, COMBINED_LABELS, lookup)

    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].set_ylabel("Accuracy (%)", fontsize=12)

    fig.suptitle(args.title, fontsize=17, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.02,
        "Control order: baseline, suppress, insert_wait, 1-shot induction, 2-shot induction.",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    fig.subplots_adjust(top=0.84, bottom=0.18, wspace=0.35)

    return fig


def main():
    args = parse_args()
    summaries = load_summaries(args.output_dir)
    summaries = filter_summaries(summaries, args)
    lookup = build_lookup(summaries)

    if not summaries:
        raise FileNotFoundError(
            "No matching Section 3.4.3 summary JSON files were found. "
            "Run eval_section_3_4_3.py first or adjust --output_dir / filters."
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig = plot_figure(args, lookup)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    pdf_output = str(Path(args.output).with_suffix(".pdf"))
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {args.output}")
    print(f"Saved plot to {pdf_output}")


if __name__ == "__main__":
    main()
