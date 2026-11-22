"""Publication-style plotting helpers for credit-banking experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MECHANISM_LABELS = {
    "linucb": "Per-arm LinUCB",
    "one_sided": "One-sided OptGTM",
    "bidirectional": "Bidirectional OptGTM",
    "windowed_w200": "Windowed bidirectional (W=200)",
}

CONDITION_LABELS = {
    "truthful": "Truthful",
    "under_reporter": "Under-reporting",
    "over_reporter": "Over-reporting",
    "time_delayed": "Time-delayed over-reporting",
    "time_banker": "Time-based banker",
    "pull_delayed": "Pull-delayed over-reporting",
    "pull_banker": "Pull-count banker",
}


def _prepare_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_grouped_estimates(
    rows,
    category_key,
    series_key,
    metric,
    path,
    title,
    ylabel,
    category_order=None,
    series_order=None,
    category_labels=None,
    series_labels=None,
    horizontal_zero=True,
    zero_count_annotations=False,
    censor_metric=None,
):
    """Plot grouped means with 95% confidence-interval error bars.

    Zero-valued rate estimates can be labelled as ``0/n`` so their upper
    confidence limits are not mistaken for observed events.  If
    ``censor_metric`` is supplied, censored estimates are hatched and labelled
    with the corresponding censoring rate.
    """
    if not rows:
        return

    category_labels = category_labels or {}
    series_labels = series_labels or {}
    category_order = category_order or sorted({r[category_key] for r in rows})
    series_order = series_order or sorted({r[series_key] for r in rows})
    lookup = {(r[category_key], r[series_key]): r for r in rows}

    x = np.arange(len(category_order), dtype=float)
    width = 0.8 / max(len(series_order), 1)
    fig, ax = plt.subplots(figsize=(max(8.0, 1.4 * len(category_order)), 5.2))
    any_censored = False

    for index, series in enumerate(series_order):
        means = []
        lower = []
        upper = []
        sample_sizes = []
        censor_rates = []
        for category in category_order:
            row = lookup.get((category, series))
            if row is None or row.get(f"{metric}_mean") is None:
                means.append(np.nan)
                lower.append(np.nan)
                upper.append(np.nan)
                sample_sizes.append(None)
                censor_rates.append(None)
                continue
            mean = float(row[f"{metric}_mean"])
            means.append(mean)
            lower.append(mean - float(row[f"{metric}_ci_low"]))
            upper.append(float(row[f"{metric}_ci_high"]) - mean)
            sample_sizes.append(row.get(f"{metric}_n"))
            censor_value = (
                row.get(f"{censor_metric}_mean") if censor_metric else None
            )
            censor_rates.append(
                None if censor_value is None else float(censor_value)
            )

        offset = (index - (len(series_order) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            yerr=np.asarray([lower, upper]),
            capsize=3,
            label=series_labels.get(series, series),
            alpha=0.88,
        )
        for bar, mean, sample_size, censor_rate in zip(
            bars, means, sample_sizes, censor_rates
        ):
            if not np.isfinite(mean):
                continue
            center = bar.get_x() + bar.get_width() / 2
            if zero_count_annotations and mean == 0.0 and sample_size:
                ax.annotate(
                    f"0/{int(sample_size)}",
                    xy=(center, 0.0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6.5,
                    color="0.25",
                )
            if censor_rate is not None and censor_rate > 0.0:
                any_censored = True
                bar.set_hatch("///")
                offset_points = 4 if mean >= 0 else -4
                ax.annotate(
                    f"{censor_rate:.0%} censored",
                    xy=(center, mean),
                    xytext=(0, offset_points),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if mean >= 0 else "top",
                    rotation=90,
                    fontsize=6.5,
                    color="0.20",
                )

    if horizontal_zero:
        ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [category_labels.get(value, value) for value in category_order],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.22)
    if any_censored:
        fig.text(
            0.01,
            0.01,
            "Hatched bars are censored paired estimates; labels show the censoring rate.",
            fontsize=7.5,
            color="0.25",
        )
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(_prepare_path(path), dpi=180)
    plt.close(fig)


def plot_calibration(calibration_rows, selected_scales, path):
    """Plot truthful false-elimination rate against radius scale."""
    if not calibration_rows:
        return

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    mechanisms = sorted({row["mechanism"] for row in calibration_rows})
    for mechanism in mechanisms:
        subset = sorted(
            (row for row in calibration_rows if row["mechanism"] == mechanism),
            key=lambda row: row["radius_scale"],
        )
        selected = selected_scales.get(mechanism)
        selected_text = "" if selected is None else f" (selected c={selected:g})"
        line, = ax.plot(
            [row["radius_scale"] for row in subset],
            [row["run_false_elimination_rate"] for row in subset],
            marker="o",
            linewidth=1.8,
            label=MECHANISM_LABELS.get(mechanism, mechanism) + selected_text,
        )
        if selected is not None:
            ax.axvline(
                selected,
                color=line.get_color(),
                linestyle=":",
                linewidth=1.0,
                alpha=0.55,
            )

    ax.set_xlabel("Confidence-radius scale")
    ax.set_ylabel("Fraction of truthful runs with any elimination")
    ax.set_title("Truthful-only confidence-radius calibration")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(_prepare_path(path), dpi=180)
    plt.close(fig)


def plot_attacker_trigger_trace(result, path, title):
    """Plot attacker credit, reverse slack, boundaries, and credit cap."""
    traces = result.get("traces")
    if traces is None or len(traces["one_sided_slack"]) == 0:
        return

    summary = result["summary"]
    pulls = np.arange(1, len(traces["one_sided_slack"]) + 1)
    boundary = np.asarray(traces["reward_boundary"], dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.plot(
        pulls,
        traces["one_sided_slack"],
        color="C0",
        linewidth=1.8,
        label=r"One-sided credit $R-L$",
    )
    ax.plot(
        pulls,
        traces["reverse_slack"],
        color="C3",
        linewidth=1.6,
        label=r"Reverse slack $R-U$",
    )
    ax.plot(pulls, boundary, color="C3", linestyle="--", label=r"$+B$")
    ax.plot(pulls, -boundary, color="C0", linestyle="--", label=r"$-B$")

    credit_cap = np.asarray(traces["credit_cap"], dtype=float)
    if np.isfinite(credit_cap).any():
        ax.plot(
            pulls,
            credit_cap,
            color="C2",
            linestyle="-.",
            linewidth=1.5,
            label=r"Effective credit cap $B+(U-L)$",
        )

    switch_pull = summary.get("switch_trace_pull")
    if switch_pull is not None:
        ax.axvline(
            switch_pull,
            color="black",
            linestyle=":",
            linewidth=1.4,
            label="End of banking phase",
        )

    elimination_pull = summary.get("elimination_pull")
    if elimination_pull is not None:
        ax.axvline(
            elimination_pull,
            color="purple",
            linestyle="--",
            linewidth=1.4,
            label=f"Eliminated ({summary.get('elimination_side')})",
        )

    ax.axhline(0.0, color="black", linewidth=0.7)
    ax.set_xlabel("Attacker pull count")
    ax.set_ylabel("Trigger statistic")
    ax.set_title(title)
    ax.grid(alpha=0.20)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(_prepare_path(path), dpi=180)
    plt.close(fig)
