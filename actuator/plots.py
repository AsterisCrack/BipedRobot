"""
actuator/plots.py -- the figures for "does this servo model describe our hardware?"

Two forms, because there are two questions:

  overlay_grid  WHERE does it disagree? Small multiples, one panel per (trajectory, kp),
                recorded against each model on a shared time axis. Reading curves is the
                only way to tell a phase lag from an amplitude error from a divergence.
  mae_summary   WHICH CASES carry the error? A mean over a session hides everything: a
                run can average 131 mrad with a median of 8 because eight free-swing
                recordings diverge. The sorted strip shows that shape at a glance, and
                the torque on/off split separates "the model is wrong" from "nothing was
                driving the joint".

Both take `curves`: {series label -> {log.name -> np.ndarray}}, so the same code draws
recorded-vs-BAM (validate.py) and recorded-vs-MuJoCo-vs-Isaac (replay.py). Series are
capped at three overlaid; see plotstyle for why.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from actuator import plotstyle as ps
from actuator.logset import Log, axes_of


def _mae(a: np.ndarray, b: np.ndarray, mask=None) -> float:
    d = np.abs(a - b)
    if mask is not None:
        d = d[mask]
    return float(d.mean()) if d.size else float("nan")


def overlay_grid(logs: list[Log], curves: dict[str, dict[str, np.ndarray]],
                 colors: dict[str, str], title: str, subtitle: str, out: Path) -> Path:
    """Small multiples: rows = trajectory, cols = kp, for ONE hole."""
    import matplotlib.pyplot as plt

    trajs, kps, _ = axes_of(logs)
    by = {(l.trajectory, l.kp): l for l in logs}
    nr, nc = len(trajs), len(kps)
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.25 * nr),
                             squeeze=False, sharex=True)

    for r, tr in enumerate(trajs):
        # Shared y across a row: the same trajectory at four gains is the comparison.
        row = [by[(tr, k)] for k in kps if (tr, k) in by]
        lo = min(min(l.q.min(), l.goal.min()) for l in row) if row else -1
        hi = max(max(l.q.max(), l.goal.max()) for l in row) if row else 1
        pad = 0.08 * (hi - lo or 1)
        for c, kp in enumerate(kps):
            ax = axes[r][c]
            lg = by.get((tr, kp))
            if lg is None:
                ax.set_axis_off()
                continue
            ps.shade_torque_off(ax, lg.t, lg.torque_enable)
            ax.plot(lg.t, lg.goal, color=ps.MUTED, lw=ps.LW_REF, label="goal", zorder=1)
            ax.plot(lg.t, lg.q, color=ps.TRUTH, lw=ps.LW, label="recorded", zorder=3)
            notes = []
            for name, series in curves.items():
                y = series.get(lg.name)
                if y is None:
                    continue
                ax.plot(lg.t[:len(y)], y, color=colors[name], lw=ps.LW, label=name, zorder=2)
                on = lg.torque_enable[:len(y)]
                m_all = _mae(y, lg.q[:len(y)])
                notes.append(f"{name} {m_all * 1000:.0f}"
                             + (f" ({_mae(y, lg.q[:len(y)], on) * 1000:.0f} on)"
                                if (~on).any() else ""))
            ax.set_ylim(lo - pad, hi + pad)
            if notes:
                ps.panel_note(ax, "\n".join(notes) + "  mrad")
            if r == 0:
                ax.set_title(f"kp {kp}", color=ps.INK)
            if c == 0:
                ax.set_ylabel(f"{tr}\nposition [rad]", color=ps.INK2)
            if r == nr - 1:
                ax.set_xlabel("time [s]")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.012))
    fig.tight_layout(rect=(0, 0.022, 1, 0.945))
    fig.suptitle(title, color=ps.INK, fontsize=12, y=0.992)
    fig.text(0.5, 0.962, subtitle, ha="center", va="top", color=ps.INK2, fontsize=8.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def mae_summary(logs: list[Log], curves: dict[str, dict[str, np.ndarray]],
                colors: dict[str, str], title: str, out: Path) -> Path:
    """Where the error lives: sorted per-log strip, then MAE by trajectory / kp / hole."""
    import matplotlib.pyplot as plt

    series = list(curves)
    # Per-log MAE, overall and torque-on-only, for every series.
    rec = {}
    for name in series:
        rows = []
        for lg in logs:
            y = curves[name].get(lg.name)
            if y is None:
                continue
            on = lg.torque_enable[:len(y)]
            rows.append({
                "log": lg, "all": _mae(y, lg.q[:len(y)]),
                "on": _mae(y, lg.q[:len(y)], on),
                "off": _mae(y, lg.q[:len(y)], ~on) if (~on).any() else np.nan,
            })
        rec[name] = rows

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.4))

    # A: sorted per-log MAE, log scale -- the shape of the distribution.
    ax = axes[0][0]
    for name in series:
        v = np.sort([r["all"] for r in rec[name]]) * 1000
        ax.plot(np.arange(1, len(v) + 1), v, marker="o", ms=4.5, lw=ps.LW,
                color=colors[name], label=name)
        med = float(np.median(v))
        ax.axhline(med, color=colors[name], lw=1.0, alpha=0.5)
        ax.text(0.99, med, f" median {med:.1f}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=7.5, color=ps.INK2)
    ax.set_yscale("log")
    ax.set_xlabel("recording, sorted by error")
    ax.set_ylabel("MAE [mrad]")
    ax.set_title("Per-recording error (log scale)", color=ps.INK)
    if len(series) > 1:
        ax.legend()

    # B/C/D: grouped bars by trajectory / kp / hole.
    def bars(ax, key, label):
        groups = sorted({getattr(r["log"], key) for r in rec[series[0]]})
        x = np.arange(len(groups))
        n = len(series)
        w = 0.8 / n
        for i, name in enumerate(series):
            vals = [np.nanmean([r["on"] for r in rec[name]
                                if getattr(r["log"], key) == g]) * 1000 for g in groups]
            ax.bar(x + (i - (n - 1) / 2) * w, vals, w * 0.88, color=colors[name],
                   label=name, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups], fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("MAE, torque on [mrad]")
        ax.set_title(f"By {label}", color=ps.INK)

    bars(axes[0][1], "trajectory", "trajectory")
    bars(axes[1][0], "kp", "kp")
    bars(axes[1][1], "hole", "hole")
    for ax in (axes[0][1], axes[1][0], axes[1][1]):
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle(title, color=ps.INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
