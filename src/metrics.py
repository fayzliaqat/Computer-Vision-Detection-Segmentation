"""Measured frame telemetry and reproducible figures."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "figure.facecolor": "#0b1220",
        "axes.facecolor": "#141f30",
        "text.color": "#e5edf7",
        "axes.labelcolor": "#c5d1e3",
        "xtick.color": "#9dacbf",
        "ytick.color": "#9dacbf",
        "axes.edgecolor": "#344358",
        "font.size": 10,
        "savefig.facecolor": "#0b1220",
    }
)


def summarizeMetrics(df, uniqueTracks):
    times = df.processing_time_ms
    return {
        "total_frames": len(df),
        "successful_frames": len(df),
        "average_processing_time_ms": float(times.mean()),
        "median_processing_time_ms": float(times.median()),
        "minimum_processing_time_ms": float(times.min()),
        "maximum_processing_time_ms": float(times.max()),
        "p95_processing_time_ms": float(times.quantile(0.95)),
        "average_fps": float(df.fps.mean()),
        "minimum_fps": float(df.fps.min()),
        "maximum_fps": float(df.fps.max()),
        "aggregate_processing_fps": float(1000 / times.mean()),
        "average_objects_per_frame": float(df.objects_detected.mean()),
        "maximum_objects_detected": int(df.objects_detected.max()),
        "unique_tracks": uniqueTracks,
    }


def createCharts(df, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    items = [
        ("processing_time_by_frame", "processing_time_ms", "Processing latency", "ms"),
        ("fps_by_frame", "fps", "Processing throughput", "frames / second"),
        (
            "object_count_by_frame",
            "objects_detected",
            "Objects and active tracks",
            "count",
        ),
    ]
    for filename, column, title, ylabel in items:
        fig, ax = plt.subplots(figsize=(10, 3.3), layout="constrained")
        ax.plot(df.frame_number, df[column], color="#36c9b0", lw=1.2, label=column)
        if column == "objects_detected":
            ax.plot(
                df.frame_number,
                df.active_tracks,
                color="#ffc373",
                ls="--",
                label="active_tracks",
            )
            ax.legend()
        ax.set(title=title, xlabel="Frame", ylabel=ylabel)
        ax.grid(alpha=0.12)
        fig.savefig(folder / f"{filename}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.3), layout="constrained")
    counts = pd.DataFrame([json.loads(value) for value in df.class_counts]).fillna(0)
    for column in counts:
        ax.plot(df.frame_number, counts[column], label=column)
    ax.set(title="Class counts over time", xlabel="Frame", ylabel="count")
    if len(counts.columns):
        ax.legend()
    fig.savefig(folder / "shape_counts_by_frame.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
    ax.hist(df.processing_time_ms, bins=25, color="#36c9b0")
    ax.set(title="Processing latency distribution", xlabel="ms", ylabel="Frames")
    fig.savefig(folder / "latency_distribution.png", dpi=160)
    plt.close(fig)


def pipelineFigure(stages, path):
    names = [
        "original",
        "grayscale",
        "blurred",
        "threshold",
        "mask",
        "segmented",
        "tracked",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    for ax, name in zip(axes.flat, names):
        data = stages[name]
        ax.imshow(
            data[..., ::-1] if data.ndim == 3 else data, cmap="gray", vmin=0, vmax=255
        )
        ax.set_title(name.replace("_", " ").title(), loc="left", pad=10)
        ax.axis("off")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(
        0.03,
        0.65,
        "VISION INTELLIGENCE\n\nGaussian + adaptive threshold\nExternal contours + pixel bounds\nCentroid IDs + live telemetry",
        fontsize=14,
        linespacing=1.7,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
