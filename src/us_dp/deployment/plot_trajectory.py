"""Save a world-frame policy plan and render it with Matplotlib."""
from pathlib import Path

import numpy as np


def trajectory_figure(start, positions, times, figure=None):
    from matplotlib.figure import Figure

    points = np.vstack((np.asarray(start).reshape(1, 3), positions))
    times = np.concatenate(([0.0], times))
    if figure is None:
        figure = Figure(figsize=(9, 7), layout="constrained")
    ax = figure.add_subplot(111, projection="3d")
    ax.plot(*points.T, color="0.5", linewidth=1, alpha=0.7)
    scatter = ax.scatter(*points.T, c=times, cmap="viridis", s=18)
    ax.scatter(*points[0], color="green", marker="o", s=90, label="Measured TCP at start")
    ax.scatter(*points[-1], color="red", marker="X", s=90, label="Predicted endpoint")
    center = (points.min(0) + points.max(0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * 0.55, 0.005)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="World X [m]", ylabel="World Y [m]", zlabel="World Z [m]",
           title=f"Predicted trajectory | {times[-1]:.2f} s | {len(positions)} waypoints")
    ax.legend(loc="upper left")
    figure.colorbar(scatter, ax=ax, shrink=0.65, label="Time from plan start [s]")
    return figure


def save_trajectory(path, start, positions, times):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), start=start, positions=positions, times=times)
    figure = trajectory_figure(start, positions, times)
    FigureCanvasAgg(figure).print_png(str(path.with_suffix(".png")))
    return path.with_suffix(".png")


def main():
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Trajectory NPZ saved during policy execution")
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as data:
        trajectory_figure(data["start"], data["positions"], data["times"],
                          figure=plt.figure(figsize=(9, 7), layout="constrained"))
    plt.show()


if __name__ == "__main__":
    main()
