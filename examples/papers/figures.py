"""Public physical-coordinate trajectory, control and value-slice figures."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from hybrid_reach import ValuePolicy


def geometry(ax, scenario):
    p = scenario.adapter.parameters
    if scenario.adapter.system == 'satellite':
        for c in p.obstacle_centers:
            ax.add_patch(Circle(c, p.obstacle_radius, color='firebrick', alpha=.3))
        if scenario.problem.kind == 'reach_avoid':
            ax.add_patch(Rectangle(np.array(p.goal_center)-p.goal_half_width,
                * (2*np.array(p.goal_half_width)), color='green', alpha=.2))
    else:
        for left, right, top in p.pillars:
            ax.add_patch(Rectangle((left, 0), right-left, top, color='firebrick', alpha=.3))
        if scenario.problem.kind == 'reach_avoid':
            ax.add_patch(Circle(p.target_center, p.target_radius, color='green', alpha=.2))
    ax.axhline(0, color='black', linewidth=.8)


def plot_run(scenario, result, trajectories, folder):
    fig, ax = plt.subplots(figsize=(8, 5))
    geometry(ax, scenario)
    for name, tr in trajectories.items():
        s = tr['state']
        ax.plot(s[:, 0], s[:, 1], label=f"{name}: {tr['status']}", linestyle='--' if name.endswith('nominal') else '-')
        ax.plot(s[0, 0], s[0, 1], 'o', markersize=3)
    ax.set(xlabel='Along-track / horizontal position [m]', ylabel='Radial / vertical position [m]')
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'trajectories.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for name, tr in trajectories.items():
        if name.endswith('nominal'): continue
        t = tr['time'][:-1]
        axes[0].plot(t, tr['value'], label=name)
        axes[1].step(t, tr['mode'][:-1], where='post')
        if len(tr['control']):
            axes[2].plot(t, tr['control'][:, 0], label=name+' u0')
            axes[2].plot(t, tr['control'][:, 1], linestyle='--', label=name+' u1')
    axes[0].axhline(0., color='black', linewidth=.8)
    for ax, label in zip(axes, ('Value', 'Mode', 'Control')):
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); axes[2].legend(fontsize=7)
    axes[-1].set_xlabel('Time [s]')
    fig.tight_layout(); fig.savefig(folder/'diagnostics.png', dpi=150); plt.close(fig)
    # Sample a physical plane. Only local interpolation queries are needed, even
    # for a large memory-mapped history on a transformed satellite grid.
    policy = ValuePolicy(scenario.problem, result)
    coordinates = scenario.adapter.coordinates
    lo = coordinates.decode(np.array([v[0] for v in result.coordinates]))
    hi = coordinates.decode(np.array([v[-1] for v in result.coordinates]))
    xs = np.linspace(max(lo[0], -6000.) if scenario.adapter.system == 'satellite' else lo[0],
                     min(hi[0], 8000.) if scenario.adapter.system == 'satellite' else hi[0], 65)
    zs = np.linspace(max(lo[1], -400.) if scenario.adapter.system == 'satellite' else lo[1],
                     min(hi[1], 10000.) if scenario.adapter.system == 'satellite' else hi[1], 65)
    fig, axes = plt.subplots(1, len(result.mode_names), figsize=(5*len(result.mode_names), 4), squeeze=False)
    for q, ax in enumerate(axes[0]):
        field = np.empty((len(zs), len(xs)))
        state = np.array(next(iter(scenario.initial_states.values())), dtype=float)
        for j, z in enumerate(zs):
            for i, x in enumerate(xs):
                state[:2] = (x, z)
                query = coordinates.encode(state)
                query = np.clip(query, [a[0] for a in result.coordinates], [a[-1] for a in result.coordinates])
                field[j, i] = policy.value(query, q, result.horizons[-1])
        ax.pcolormesh(xs, zs, field, cmap='coolwarm', shading='auto')
        if field.min() < 0 < field.max(): ax.contour(xs, zs, field, levels=[0.], colors='black')
        geometry(ax, scenario)
        ax.set(title=f'{result.mode_names[q]}, initial velocity / phase', xlabel='Position [m]', ylabel='Height [m]')
    fig.suptitle('Finite-grid value slices; black contour V=0')
    fig.tight_layout(); fig.savefig(folder/'value_slices.png', dpi=150); plt.close(fig)
