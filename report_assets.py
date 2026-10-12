"""Report assets: builds every table and figure for the report from the stored evaluation outputs.

Reads the JSON and CSV files that the other scripts store in the outputs directory and writes
tables (CSV) and figures (PNG) to outputs/report. No number is typed by hand, so the report
always matches the stored results.

Usage:
    python -m src.report_assets
"""
import glob
import json
import logging
from pathlib import Path

import pandas as pd

from src import utils

logger = logging.getLogger(__name__)

# Fixed method order, display names and colours. The colours follow one validated categorical order.
METHOD_ORDER = ["rules", "climatebert", "llm"]
METHOD_NAMES = {"rules": "Rule based baseline", "climatebert": "Fine tuned ClimateBERT", "llm": "Zero shot LLM"}
METHOD_COLOURS = {"rules": "#2a78d6", "climatebert": "#eb6834", "llm": "#1baf7a"}
CLASS_ORDER = ["none", "net_zero", "reduction"]
FLAG_NAMES = {"long_text": "Long sentence\n(over 500 characters)", "has_line_break": "Contains\nline breaks",
              "non_english": "Not English\n(rough check)"}
# One hue sequential ramp (light to dark) used for the confusion matrices.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]
INK, INK_SECONDARY, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _read_json(path: Path) -> dict:
    """Reads one JSON file and returns its content."""
    return json.loads(Path(path).read_text(encoding="utf8"))


def split_table(summary: dict) -> pd.DataFrame:
    """Returns one row per split with class counts, class shares and sentence length statistics."""
    rows = []
    for name in ("train", "validation", "test"):
        s = summary["splits"][name]
        row = {"split": name, "sentences": s["n"]}
        for label in CLASS_ORDER:
            row[f"{label}_count"] = s["class_counts"][label]
        for label in CLASS_ORDER:
            row[f"{label}_share_percent"] = round(100 * s["class_share"][label], 1)
        row.update({"median_characters": s["chars_median"], "p95_characters": s["chars_p95"],
                    "max_characters": s["chars_max"], "rows_with_line_breaks": s["rows_with_line_breaks"]})
        rows.append(row)
    return pd.DataFrame(rows)


def results_tables(metrics: dict) -> tuple:
    """Returns the overall results table and the per class results table for the available methods."""
    overall, per_class = [], []
    for method in [m for m in METHOD_ORDER if m in metrics]:
        m = metrics[method]
        overall.append({"method": METHOD_NAMES[method], "accuracy": m["accuracy"],
                        "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
                        "macro_f1": m["macro_f1"]})
        for label in CLASS_ORDER:
            c = m["per_class"][label]
            per_class.append({"method": METHOD_NAMES[method], "class": label, "precision": c["precision"],
                              "recall": c["recall"], "f1": c["f1"], "support": c["support"]})
    return pd.DataFrame(overall), pd.DataFrame(per_class)


def bootstrap_table(out_dir: Path) -> pd.DataFrame:
    """Returns one row per stored paired bootstrap result, or an empty table when none exist."""
    rows = []
    for path in sorted(glob.glob(str(out_dir / "bootstrap_*_vs_*.json"))):
        r = _read_json(path)
        rows.append({"method_a": METHOD_NAMES.get(r["method_a"], r["method_a"]),
                     "method_b": METHOD_NAMES.get(r["method_b"], r["method_b"]),
                     "macro_f1_difference_b_minus_a": round(r["observed_diff"], 4),
                     "ci95_low": round(r["ci95_low"], 4), "ci95_high": round(r["ci95_high"], 4),
                     "p_value": round(r["p_value"], 4)})
    return pd.DataFrame(rows)


def signal_tables(out_dir: Path) -> tuple:
    """Returns the per signal scores for every stored method and the labeller agreement table."""
    rows = []
    for method in METHOD_ORDER:
        path = out_dir / f"signal_metrics_{method}.json"
        if not path.exists():
            continue
        data = _read_json(path)
        for signal, m in data.items():
            if signal == "macro_f1":
                continue
            rows.append({"method": METHOD_NAMES[method], "signal": signal, "precision": m["precision"],
                         "recall": m["recall"], "f1": m["f1"], "present_in_key": m["support_present"]})
        rows.append({"method": METHOD_NAMES[method], "signal": "macro average", "precision": None,
                     "recall": None, "f1": data["macro_f1"], "present_in_key": None})
    agreement_path = out_dir / "signal_agreement.json"
    agreement = []
    if agreement_path.exists():
        for signal, a in _read_json(agreement_path).items():
            agreement.append({"signal": signal, **a})
    return pd.DataFrame(rows), pd.DataFrame(agreement)


def _style_axes(ax) -> None:
    """Applies the shared chart style: recessive grid, no top or right frame, secondary ink for ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def confusion_figure(metrics: dict, method: str, path: Path) -> None:
    """Draws one confusion matrix with counts and row shares, coloured by the share of each gold class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cm = metrics["confusion_matrix"]
    labels, matrix = cm["labels"], pd.DataFrame(cm["matrix"]).to_numpy()
    shares = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=200, facecolor=SURFACE)
    ax.imshow(shares, cmap=LinearSegmentedColormap.from_list("seq", SEQUENTIAL), vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            colour = "#ffffff" if shares[i, j] > 0.5 else INK
            ax.text(j, i, f"{matrix[i, j]}\n{100 * shares[i, j]:.1f}%", ha="center", va="center",
                    color=colour, fontsize=10)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted label", color=INK_SECONDARY)
    ax.set_ylabel("Gold label", color=INK_SECONDARY)
    ax.set_title(METHOD_NAMES.get(method, method), color=INK, fontsize=11)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def macro_f1_figure(overall: pd.DataFrame, path: Path) -> None:
    """Draws one bar per method for macro F1 with the value labelled on each bar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = [METHOD_COLOURS[m] for m in METHOD_ORDER if METHOD_NAMES[m] in set(overall["method"])]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200, facecolor=SURFACE)
    bars = ax.bar(overall["method"], overall["macro_f1"], color=colours, width=0.5)
    for bar, value in zip(bars, overall["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", color=INK, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Macro F1 on the shared test set", color=INK_SECONDARY)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def error_rate_figure(summary: dict, path: Path) -> None:
    """Draws error rates for sentences with and without each property, one panel per method.

    Each panel is labelled with the method name. The lighter bar is the error rate for sentences
    without the property, and the number under each group is the count of sentences that have it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    methods = [m for m in METHOD_ORDER if m in summary["per_method"]]
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 3.8), dpi=200,
                             facecolor=SURFACE, sharey=True, squeeze=False)
    flags = list(FLAG_NAMES)
    for ax, method in zip(axes[0], methods):
        by_flag = summary["per_method"][method]["by_flag"]
        colour = METHOD_COLOURS[method]
        for k, flag in enumerate(flags):
            for offset, key, alpha in ((-0.19, "error_rate_with_flag", 1.0), (0.19, "error_rate_without_flag", 0.35)):
                value = by_flag[flag][key]
                if value is None:
                    continue
                ax.bar(k + offset, value, width=0.34, color=colour, alpha=alpha)
                ax.text(k + offset, value + 0.012, f"{100 * value:.0f}%", ha="center", color=INK, fontsize=9)
        ax.set_xticks(range(len(flags)), [f"{FLAG_NAMES[f]}\nn = {by_flag[f]['rows_with_flag']}" for f in flags])
        ax.set_title(METHOD_NAMES[method], color=INK, fontsize=11)
        ax.set_ylim(0, 0.8)
        _style_axes(ax)
    axes[0][0].set_ylabel("Error rate", color=INK_SECONDARY)
    axes[0][0].yaxis.set_major_formatter(lambda v, _: f"{100 * v:.0f}%")
    handles = [Patch(color=INK_SECONDARY, label="Sentences with the property"),
               Patch(color=INK_SECONDARY, alpha=0.35, label="Sentences without it")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def generate_all(out_dir: Path, report_dir: Path) -> list:
    """Builds every available table and figure and returns the list of files written."""
    report_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def save_table(frame: pd.DataFrame, name: str) -> None:
        if frame.empty:
            return
        frame.to_csv(report_dir / name, index=False)
        written.append(name)

    if (out_dir / "split_summary.json").exists():
        save_table(split_table(_read_json(out_dir / "split_summary.json")), "table_split.csv")
    metrics = {m: _read_json(out_dir / f"{m}_metrics.json") for m in METHOD_ORDER
               if (out_dir / f"{m}_metrics.json").exists()}
    if metrics:
        overall, per_class = results_tables(metrics)
        save_table(overall, "table_results.csv")
        save_table(per_class, "table_per_class.csv")
        macro_f1_figure(overall, report_dir / "figure_macro_f1.png")
        written.append("figure_macro_f1.png")
        for method, m in metrics.items():
            confusion_figure(m, method, report_dir / f"figure_confusion_{method}.png")
            written.append(f"figure_confusion_{method}.png")
    save_table(bootstrap_table(out_dir), "table_bootstrap.csv")
    signals, agreement = signal_tables(out_dir)
    save_table(signals, "table_signals.csv")
    save_table(agreement, "table_signal_agreement.csv")
    if (out_dir / "error_summary.json").exists():
        error_rate_figure(_read_json(out_dir / "error_summary.json"), report_dir / "figure_error_rates.png")
        written.append("figure_error_rates.png")
    return written


def main() -> None:
    """Generates all report tables and figures into outputs/report."""
    utils.setup_logging()
    cfg = utils.load_config()
    out_dir = Path(cfg["paths"]["predictions_dir"])
    written = generate_all(out_dir, out_dir / "report")
    logger.info("Wrote %d files to %s: %s", len(written), out_dir / "report", written)


if __name__ == "__main__":
    main()
