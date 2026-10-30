"""Figures and GIFs for the falling-piece lesson; no solver or policy logic."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Patch, Rectangle
import numpy as np

from hybrid_reach.examples.falling_piece import MODE_NAMES


def scene(ax, parameters, kind, *, labels=True):
    p = parameters
    ax.axhspan(p.y_high, p.y_bounds[1], color="#e9f2fc", zorder=-5)
    ax.axhspan(p.y_low, p.y_high, color="#fff5df", zorder=-5)
    ax.axhspan(p.y_bounds[0], p.y_low, color="#edf0f3", zorder=-5)
    for y in (p.y_low, p.y_high):
        ax.axhline(y, color="#485563", linestyle="--", linewidth=1.2)
    if kind != "reach":
        for left, right, bottom, top in p.obstacles:
            ax.add_patch(Rectangle((left, bottom), right-left, top-bottom,
                                  facecolor="#e87573", edgecolor="#9a2323", zorder=3))
        for x in p.x_bounds:
            ax.axvline(x, color="#9a2323", linewidth=2)
    if kind != "avoid":
        left, right, bottom, top = p.target
        ax.add_patch(Rectangle((left, bottom), right-left, top-bottom,
                              facecolor="#47bc83", edgecolor="#146440", linewidth=1.5, zorder=3))
    if labels:
        for y, text in (((p.y_high+p.y_bounds[1])/2, "STEER: move sideways, fall slowly"),
                        (p.y_high-.22, "DROP: fixed column, fall quickly"),
                        ((p.y_bounds[0]+p.y_low)/2, "REST: no motion")):
            ax.text(.5, y, text, transform=ax.get_yaxis_transform(), ha="center", va="center",
                    fontsize=9, color="#263b50", bbox=dict(facecolor="white", alpha=.85, edgecolor="none"))
    ax.set(xlim=p.x_bounds, ylim=p.y_bounds, xlabel="Horizontal position x", ylabel="Height y")
    ax.set_aspect("equal")


def physical_field(result, p):
    y = result.coordinates[1]
    return np.where(y[None, :] <= p.y_low, result.values[-1, 2],
        np.where(y[None, :] <= p.y_high, result.values[-1, 1], result.values[-1, 0]))


def display_label(name, kind):
    if kind == "reach":
        return {"A": "A: inside BRT aligned", "B": "B: inside BRT steer",
                "C": "C: outside BRT guaranteed fail"}.get(name[:1], name)
    return name


def reachable_background(ax, result, p):
    field = physical_field(result, p)
    colors = ["#d5eddf", "#f0f1f3"] if result.kind != "avoid" else ["#f7d9d6", "#d5eddf"]
    ax.contourf(*result.coordinates, (field > 0).T, levels=[-.5, .5, 1.5], colors=colors, alpha=.85)
    if field.min() < 0 < field.max():
        ax.contour(*result.coordinates, field.T, levels=[0.], colors="#263b50", linewidths=1.4)
    return field


def analytic_reach_boundary(ax, p, horizon):
    """Reach-only column alignment boundary, independent of the computed field."""
    left, right, bottom, top = p.target
    drop_time = (p.y_high-top)/p.fast_speed
    top_y = min(p.y_bounds[1], p.y_high+p.slow_speed*(horizon-drop_time))
    if top_y > p.y_high:
        y = np.linspace(p.y_high, top_y, 100)
        steering = p.u_max*(y-p.y_high)/p.slow_speed
        ax.plot(left-steering, y, "--", color="#5750a3", linewidth=1.2,
                label="Reach-only boundary (analytic)")
        ax.plot(right+steering, y, "--", color="#5750a3", linewidth=1.2)
    upper = min(p.y_high, top+p.fast_speed*horizon)
    if upper > bottom:
        ax.plot([left, left], [bottom, upper], "--", color="#5750a3", linewidth=1.2)
        ax.plot([right, right], [bottom, upper], "--", color="#5750a3", linewidth=1.2)


def plot_setup(p, output):
    fig, ax = plt.subplots(figsize=(7, 6))
    scene(ax, p, "reach_avoid")
    ax.annotate("", xy=(.8, 6.2), xytext=(-.8, 6.2),
                arrowprops=dict(arrowstyle="<->", color="#2567b4", linewidth=2))
    ax.plot(0., 6.2, "s", color="#2567b4", markersize=9)
    ax.text(.1, 6.45, f"u in [{-p.u_max:g}, {p.u_max:g}]", fontsize=10)
    ax.annotate("", xy=(0., 1.2), xytext=(0., 3.4),
                arrowprops=dict(arrowstyle="->", color="#9a6a1b", linewidth=2))
    ax.set_title("Choose your column before steering disappears")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_brt(result, p, trajectories, output):
    fig, ax = plt.subplots(figsize=(8, 7))
    reachable_background(ax, result, p)
    scene(ax, p, result.kind, labels=False)
    if result.kind != "avoid":
        analytic_reach_boundary(ax, p, float(result.horizons[-1]))
    for i, (name, trajectory) in enumerate(trajectories.items()):
        color = plt.get_cmap("tab10")(i)
        x, y = trajectory["state"].T
        label = display_label(name, result.kind)
        ax.plot(x, y, color=color, linewidth=1.6,
                label=label if result.kind == "reach" else f"{label}: {trajectory['status']}")
        ax.plot(x[0], y[0], "s", color=color, markersize=5)
    meaning = "Green: computed safe set; red: failure BRT" if result.kind == "avoid" else "Green: computed winning BRT"
    ax.set_title(f"{result.kind.replace('_', '-')}, T={result.horizons[-1]:g} s\n{meaning}")
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, Patch(facecolor="#d5eddf"))
    labels.insert(0, "Computed safe set" if result.kind == "avoid" else "Computed winning BRT")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)


def animate(result, p, trajectories, output, *, frames=81):
    fig, ax = plt.subplots(figsize=(8, 6))
    reachable_background(ax, result, p)
    scene(ax, p, result.kind, labels=False)
    ax.text(.5, 6.7, "STEER", transform=ax.get_yaxis_transform(), ha="center", fontsize=10)
    ax.text(.5, 3.7, "DROP", transform=ax.get_yaxis_transform(), ha="center", fontsize=10)
    ax.text(.5, .5, "REST", transform=ax.get_yaxis_transform(), ha="center", fontsize=10)
    reach = result.kind == "reach"
    artists = []
    for i, (name, trajectory) in enumerate(trajectories.items()):
        color = plt.get_cmap("tab10")(i)
        line, = ax.plot([], [], color=color, linewidth=1.8, label=display_label(name, result.kind))
        marker, = ax.plot([], [], "s", color=color, markersize=7, markeredgewidth=2)
        text_y = .61-i*.20 if reach else .87-i*.17
        text = ax.text(1.04, text_y, "", transform=ax.transAxes, fontsize=9, va="top")
        artists.append((name, trajectory, line, marker, text))
    handles, labels = ax.get_legend_handles_labels()
    if reach:
        handles.insert(0, Patch(facecolor="#d5eddf"))
        labels.insert(0, "Reach BRT")
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    title = ax.set_title("")
    fig.subplots_adjust(left=.10, right=.67, top=.89, bottom=.11)
    horizon = float(result.horizons[-1])

    def draw(time):
        if reach:
            title.set_text(f"Reach | t={time:.2f} s")
        else:
            title.set_text(f"{result.kind.replace('_', '-')} | t={time:.2f} s | remaining={horizon-time:.2f} s\n"
                           "Background: initial horizon; orange outline: filter intervention")
        for name, tr, line, marker, text in artists:
            times, states = tr["time"], tr["state"]
            sample_time = min(time, float(times[-1]))
            position = np.array([np.interp(sample_time, times, states[:, axis]) for axis in range(2)])
            count = np.searchsorted(times, sample_time, side="right")
            path = np.vstack((states[:count], position))
            line.set_data(path[:, 0], path[:, 1])
            marker.set_data([position[0]], [position[1]])
            index = min(max(0, count-1), len(tr["mode"])-1)
            stopped = time >= float(times[-1])-1e-10
            control_index = min(index, len(tr["control"])-1)
            control = 0. if stopped or control_index < 0 else float(tr["control"][control_index, 0])
            intervention = (not stopped and "filtered" in name and control_index >= 0
                            and bool(tr["continuous_intervention"][control_index]))
            marker.set_markeredgecolor("#f19a22" if intervention else line.get_color())
            status = tr["status"] if stopped else MODE_NAMES[int(tr["mode"][index])]
            text_name = name[:1] if reach else name
            text.set_text(f"{text_name}\n{status}\nu={control:+.1f}")
        return ()

    animation = FuncAnimation(fig, draw, frames=np.linspace(0., horizon, frames), blit=False)
    animation.save(output, writer=PillowWriter(fps=20), dpi=90)
    plt.close(fig)
