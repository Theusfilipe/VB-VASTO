import csv
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def plot_training(monitor_csv: Path, destination: Path) -> bool:
    """Plot finished episodes from a training run; return false if there are none."""
    with monitor_csv.open(encoding="utf-8", newline="") as stream:
        stream.readline()  # SB3 Monitor metadata
        rows = list(csv.DictReader(stream))
    if not rows:
        return False

    figure = Figure(figsize=(9, 6), layout="constrained")
    FigureCanvasAgg(figure)
    series = (
        ("Episode reward", [float(row["r"]) for row in rows]),
        ("Survival (steps)", [int(row["l"]) for row in rows]),
        ("Food eaten (units)", [int(row["food_eaten"]) for row in rows]),
        ("Sleep (fraction of steps)", [int(row["sleep_steps"]) / int(row["l"]) for row in rows]),
    )
    for axis, (label, values) in zip(figure.subplots(2, 2).flat, series, strict=True):
        axis.plot(range(1, len(values) + 1), values)
        axis.set(xlabel="Episode", ylabel=label)
    figure.savefig(destination)
    return True
