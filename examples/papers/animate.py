"""Animate saved paper-demo trajectories using only NumPy/JSON and Matplotlib.

Example:
    python examples/papers/animate.py outputs/papers/jump_avoid_demo \
        --output outputs/animations/jump_avoid.gif
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt
import numpy as np


CASES = ("jump_avoid", "satellite_reach_avoid")
COLORS = ("#167c80", "#b45d2d")


def read_run(directory):
    directory = Path(directory)
    configuration = json.loads((directory/"configuration.json").read_text())
    reports = json.loads((directory/"rollouts.json").read_text())
    case = configuration["name"]
    if case not in CASES:
        raise ValueError(f"Animation supports {CASES}, received {case!r}")
    names = (("standard_filtered", "standard_nominal") if case == "jump_avoid"
             else ("low_velocity_filtered", "high_velocity_filtered"))
    trajectories = {}
    inputs = {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
              for name in ("configuration.json", "rollouts.json")}
    for name in names:
        path = directory/f"{name}.npz"
        inputs[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as data:
            trajectory = {key: data[key] for key in
                ("time", "state", "mode", "control", "continuous_intervention")}
        times, states = trajectory["time"], trajectory["state"]
        if (times.ndim != 1 or len(times) < 2 or times[0] != 0
                or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                or states.ndim != 2 or states.shape[0] != len(times) or states.shape[1] < 2
                or not np.isfinite(states).all()):
            raise ValueError(f"Invalid saved trajectory {name}")
        if (trajectory["mode"].shape != times.shape
                or trajectory["control"].shape != (len(times)-1, 2)
                or trajectory["continuous_intervention"].shape != (len(times)-1,)
                or not np.isfinite(trajectory["control"]).all()):
            raise ValueError(f"Invalid saved controls/modes for {name}")
        report = reports[name]
        if not np.isclose(times[-1], report["final_time"], rtol=0, atol=1e-8):
            raise ValueError(f"Saved endpoint time differs from report for {name}")
        if not np.allclose(states[-1], report["final_state"], rtol=0, atol=1e-8):
            raise ValueError(f"Saved endpoint state differs from report for {name}")
        trajectory["report"] = report
        trajectories[name] = trajectory
    return configuration, trajectories, inputs


def displayed_sample(trajectory, time):
    """Interpolate saved physical positions; retain the final point after stopping."""
    times, states = trajectory["time"], trajectory["state"]
    sample_time = min(time, float(times[-1]))
    count = int(np.searchsorted(times, sample_time, side="right"))
    position = np.array([np.interp(sample_time, times, states[:, axis]) for axis in (0, 1)])
    path = np.vstack((states[:count, :2], position))
    index = min(max(0, count-1), len(times)-1)
    stopped = time >= float(times[-1])
    return position, path, index, stopped


def jumper_figure(configuration, trajectories):
    fig, (ax, inputs_ax) = plt.subplots(2, 1, figsize=(8, 5.2),
        gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    p = configuration["parameters"]
    for left, right, top in p["pillars"]:
        ax.add_patch(Rectangle((left, 0), right-left, top,
            facecolor="#e8b2ab", edgecolor="#9d4339", linewidth=1.4))
    ax.axhline(0, color="#475569", linewidth=1.2)
    finish = configuration.get("finish_x")
    if finish is not None:
        ax.axvline(finish, color="#258653", linestyle=":", linewidth=1.4)
        ax.text(finish+.06, .08, "finish marker", va="bottom", rotation=90,
                color="#258653", fontsize=8)
    all_positions = np.concatenate([tr["state"][:, :2] for tr in trajectories.values()])
    ax.set(xlim=(0., max(5.7, float(all_positions[:, 0].max())+.3)),
        ylim=(-.04, max(1.16, float(all_positions[:, 1].max())+.25)),
        xlabel="Horizontal position [m]", ylabel="Height [m]")
    ax.grid(alpha=.12)
    artists = []
    for i, (name, tr) in enumerate(trajectories.items()):
        filtered = name.endswith("filtered")
        color = COLORS[i] if filtered else "#8b5365"
        line, = ax.plot([], [], color=color, linewidth=2.2,
            linestyle="-" if filtered else "--", label="Safety filtered" if filtered else "Nominal")
        point, = ax.plot([], [], "o", color=color, markersize=8, markeredgewidth=2)
        artists.append((name, tr, line, point, color))
    ax.legend(loc="upper left", framealpha=.92, fontsize=9)
    status = ax.text(.99, .95, "", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    title = ax.set_title("")
    filtered = trajectories["standard_filtered"]
    force_lines = [inputs_ax.plot([], [], color=c, drawstyle="steps-post", linewidth=1.1,
                                 label=label)[0]
                   for c, label in (("#167c80", "Applied horizontal force"),
                                    ("#9766ab", "Applied vertical force"))]
    force_max = float(np.max(p["force_max"]))
    inputs_ax.set(xlim=(0, float(filtered["time"][-1])), ylim=(-1, force_max+2),
                  xlabel="Physical time [s]", ylabel="Force [N]")
    inputs_ax.legend(loc="upper right", fontsize=8, ncol=2)
    inputs_ax.grid(alpha=.15)

    def draw(time):
        descriptions = []
        for name, tr, line, point, color in artists:
            position, path, index, stopped = displayed_sample(tr, time)
            line.set_data(path[:, 0], path[:, 1])
            point.set_data([position[0]], [position[1]])
            failed = stopped and tr["report"]["status"] == "failure"
            point.set_marker("x" if failed else "o")
            point.set_color("#ae2735" if failed else color)
            filtered_path = name.endswith("filtered")
            intervention = (filtered_path and not stopped
                            and bool(tr["continuous_intervention"][index]))
            point.set_markeredgecolor("#f0a126" if intervention else ("#ae2735" if failed else color))
            if stopped:
                label = "collision" if failed else "passed pillars"
            else:
                label = ("stance", "flight")[int(tr["mode"][index])]
                if intervention:
                    label += " · safety intervention"
            descriptions.append(f"{'Filtered' if filtered_path else 'Nominal'}: {label}")
        status.set_text("\n".join(descriptions))
        count = min(int(np.searchsorted(filtered["time"][:-1], time, side="right")),
                    len(filtered["control"]))
        for axis, line in enumerate(force_lines):
            line.set_data(filtered["time"][:count], filtered["control"][:count, axis])
        title.set_text(f"Jumping robot · hybrid safety filtering\nt = {time:.2f} s  |  orange outline = intervention")
        return ()

    return fig, draw


def satellite_figure(configuration, trajectories):
    fig, ax = plt.subplots(figsize=(8, 5.2))
    p = configuration["parameters"]
    for center in p["obstacle_centers"]:
        ax.add_patch(Circle(np.array(center)/1000, p["obstacle_radius"]/1000,
            facecolor="#e8b2ab", edgecolor="#9d4339", linewidth=1.2))
    goal_center, half_width = np.array(p["goal_center"])/1000, np.array(p["goal_half_width"])/1000
    ax.add_patch(Rectangle(goal_center-half_width, *(2*half_width),
                           facecolor="#9ed9b7", edgecolor="#258653", linewidth=1.5))
    ax.annotate("Goal", xy=goal_center, xytext=(1.1, .55), fontsize=9, color="#258653",
                arrowprops=dict(arrowstyle="->", color="#258653"))
    ax.axhline(0, color="#475569", linewidth=1.2)
    positions = np.concatenate([tr["state"][:, :2] for tr in trajectories.values()])/1000
    centers = np.asarray(p["obstacle_centers"])/1000
    radius = p["obstacle_radius"]/1000
    lo = np.minimum(positions.min(0), (centers-radius).min(0))
    hi = np.maximum(positions.max(0), (centers+radius).max(0))
    ax.set(xlim=(lo[0]-.65, hi[0]+.65), ylim=(-.2, hi[1]+.65),
           xlabel="Along-track position [km]", ylabel="Radial position [km]")
    ax.set_aspect("equal")
    ax.grid(alpha=.12)
    artists = []
    for i, (name, tr) in enumerate(trajectories.items()):
        label = "Start A" if name.startswith("low") else "Start B"
        line, = ax.plot([], [], color=COLORS[i], linewidth=1.8, label=label)
        point, = ax.plot([], [], "D", color=COLORS[i], markersize=7, markeredgewidth=2)
        text = ax.text(1.03, .63-i*.19, "", transform=ax.transAxes, fontsize=9, va="top")
        artists.append((label, tr, line, point, text))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    ax.text(1.03, .78, "Orange outline:\nthrust active", transform=ax.transAxes, fontsize=8, color="#996317")
    title = ax.set_title("")
    fig.subplots_adjust(left=.09, right=.74, bottom=.12, top=.84)

    def draw(time):
        title.set_text(f"Spacecraft · reach-avoid\nt = {time:.0f} s  |  two saved initial conditions")
        for label, tr, line, point, text in artists:
            position, path, index, stopped = displayed_sample(tr, time)
            line.set_data(path[:, 0]/1000, path[:, 1]/1000)
            point.set_data([position[0]/1000], [position[1]/1000])
            mode = int(tr["mode"][index])
            point.set_markeredgecolor("#f0a126" if mode == 1 and not stopped else line.get_color())
            if stopped:
                mode_text = "goal reached" if tr["report"]["status"] == "goal_reached" else tr["report"]["status"]
                text.set_text(f"{label}: {mode_text}\nat {tr['time'][-1]:.1f} s")
            else:
                text.set_text(f"{label}: {('idle', 'thrust', 'cooldown')[mode]}")
        return ()

    return fig, draw


def render(directory, output, *, frames=121, fps=20, hold=1.):
    output = Path(output)
    sidecar = output.with_suffix(".json")
    if output.suffix.lower() != ".gif" or frames < 2 or fps <= 0 or not np.isfinite(hold) or hold < 0:
        raise ValueError("Use a GIF output, at least two frames, positive fps and a nonnegative hold")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Choose a fresh animation output path")
    config, trajectories, inputs = read_run(directory)
    fig, draw = (jumper_figure if config["name"] == "jump_avoid" else satellite_figure)(config, trajectories)
    duration = max(float(tr["time"][-1]) for tr in trajectories.values())
    frame_times = np.r_[np.linspace(0., duration, frames), np.repeat(duration, round(hold*fps))]
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        animation = FuncAnimation(fig, draw, frames=frame_times, blit=False)
        animation.save(output, writer=PillowWriter(fps=fps), dpi=100)
    finally:
        plt.close(fig)
    provenance = dict(case=config["name"], preset=config.get("preset"),
        rendering="Linear interpolation of saved physical positions; recorded modes/controls held between samples",
        physical_duration_seconds=duration, rendered_frames=len(frame_times), fps=fps,
        source_sha256=inputs, animation_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        outcomes={name: dict(status=tr["report"]["status"], final_time=float(tr["time"][-1]))
                  for name, tr in trajectories.items()})
    sidecar.write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Rendered saved {config['name']} trajectories to {output}")
    return provenance


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Completed run with configuration.json, rollouts.json and trajectory NPZs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=121)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--hold", type=float, default=1., help="Hold the final frame for this many playback seconds")
    args = parser.parse_args()
    render(args.directory, args.output, frames=args.frames, fps=args.fps, hold=args.hold)
