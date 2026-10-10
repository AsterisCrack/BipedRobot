#!/usr/bin/env python3
"""
actuator/validate.py -- do the published 12 V STS3215 parameters describe OUR servos?

This is the Phase-2 gate in TODO_SIM2REAL.md. It takes one SysID session recorded on the
Jetson (copied to actuator/data/<session>/) and:

  1. Screens out PWM-saturated recordings before anything else. Saturated frames make
     the fit "explain" a motor that won't move by inflating R and deflating kt. The
     reference project's 29% torque underestimate came from exactly this. A frame counts
     as stalled when torque is on, |speed| < 0.05 rad/s and |goal - position| > 0.2 rad.
     A file with >= 10% stalled frames is set aside, not processed.
  2. Resamples to BAM's fixed dt with `bam.process`.
  3. Scores every model in the params directory with `bam.mae`, then prints the result
     next to the published hold-out numbers.

Two adjustments are made to temporary copies of the params, never to the files in
actuator/params/:
  * "actuator": "sts3215_12v" -> "sts3215". The 12 V registry key exists only in the
    unmerged BAM PR #20. The 12 V supply still applies, because every log carries its own
    measured `vin` and BAM's load_log uses it.
  * q_offset -> 0. q_offset is the publishing rig's calibration error, pinned at its
    -0.1 bound. Our rig defines position 0 directly from the captured hanging zero.

Reading the result. Published hold-out MAE for these params is ~10 mrad (M6) and ~20 mrad
(M1). Landing in that neighbourhood means the placeholder already describes our servos
and identification (Phase 3) can be skipped. Several times worse means fit our own:
    python -m bam.fit --actuator sts3215 --model m6 --logdir <processed> \\
                      --validation_kp 8 --output actuator/params/ours/m6.json
then point env_config.actuator.params_file at the result.

Usage:
    python -m actuator.validate --session actuator/data/bench_servo1
    python -m actuator.validate --session ... --params actuator/params/sts3215_12v --keep-saturated
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from actuator import plots, plotstyle
from actuator.logset import NAME_RE, axes_of, load_session

_REPO = Path(__file__).resolve().parent.parent
PUBLISHED_MRAD = {"m1": 20.8, "m6": 10.0}       # BAM PR #20, kp=8 held out
STALL_SPEED, STALL_ERR, STALL_FRAC = 0.05, 0.2, 0.10


def stalled_fraction(log: dict) -> float:
    e = log["entries"]
    if not e:
        return 1.0
    stall = [x["torque_enable"] and abs(x["speed"]) < STALL_SPEED
             and abs(x["goal_position"] - x["position"]) > STALL_ERR for x in e]
    return float(np.mean(stall))


def rollout(model_name: str, params_path: Path, log_path: Path) -> np.ndarray:
    """Re-simulate one recorded log with a BAM model, the way bam.fit scores it.

    simulate_control=True means the firmware control law is re-run from the simulated
    state, so this is an open-loop rollout over the whole 6 s: the model gets the goal
    sequence and nothing else. That is the honest test -- feeding it the recorded
    position each step would hide any error.
    """
    import copy

    from bam.feetech.actuator import STS3215Actuator
    from bam.model import models
    from bam.simulate import Simulator
    from bam.testbench import Pendulum

    log = json.loads(log_path.read_text())
    m = models[model_name]()
    m.set_actuator(STS3215Actuator(Pendulum))
    m.load_parameters(str(params_path))
    m.q_offset.value = 0.0
    q, _, _ = Simulator(m).rollout_log(copy.deepcopy(log), simulate_control=True)
    return np.asarray(q, dtype=float)


def make_plots(session: Path, model: str, params_path: Path) -> dict:
    """Roll `model` out over every log, write the figures, return the error split.

    The split matters: a servo whose gearbox will not back-drive holds its position when
    torque is cut, and a model with too little stiction lets it fall. Those samples then
    dominate a session mean while saying nothing about how the model behaves under the
    only condition the policy ever runs in, which is torque on.
    """
    plotstyle.apply()
    logs = load_session(session)
    curves = {model: {}}
    stats = {"all": [], "on": [], "off": []}
    for lg in logs:
        y = rollout(model, params_path, lg.path)
        curves[model][lg.name] = y
        on = lg.torque_enable[:len(y)]
        err = np.abs(y - lg.q[:len(y)])
        stats["all"].append(err.mean())
        stats["on"].append(err[on].mean() if on.any() else np.nan)
        if (~on).any():
            stats["off"].append(err[~on].mean())
    colors = {model: plotstyle.SIM1}
    out_dir = session / "plots"
    written = []
    _, _, holes = axes_of(logs)
    for h in holes:
        sub = [l for l in logs if l.hole == h]
        mm = sub[0]
        written.append(plots.overlay_grid(
            sub, {model: curves[model]}, colors,
            title=f"{session.name} - recorded vs BAM {model}, hole {h}",
            subtitle=f"point-pendulum equivalent {mm.mass * 1000:.0f} g at "
                     f"{mm.length * 1000:.1f} mm, vin {mm.vin:.2f} V, dt {mm.dt * 1000:.0f} ms"
                     f"   |   grey band = torque off",
            out=out_dir / f"fit_{model}_h{h}.png"))
    written.append(plots.mae_summary(
        logs, curves, colors,
        title=f"{session.name} - where the error is, BAM {model}",
        out=out_dir / f"summary_{model}.png"))
    return {"written": written,
            "mean": float(np.mean(stats["all"])), "median": float(np.median(stats["all"])),
            "on_mean": float(np.nanmean(stats["on"])), "on_median": float(np.nanmedian(stats["on"])),
            "off_mean": float(np.mean(stats["off"])) if stats["off"] else float("nan"),
            "n_off": len(stats["off"])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True, help="dir of raw BAM logs copied off the Jetson")
    ap.add_argument("--params", default=str(_REPO / "actuator" / "params" / "sts3215_12v"))
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--keep-saturated", action="store_true", help="skip the saturation screen")
    ap.add_argument("--plot-model", default="best",
                    help="model to plot: a name (m6), 'best' (lowest MAE), or 'none'")
    a = ap.parse_args()

    session = Path(a.session)
    # NAME_RE, not *.json: a re-run would otherwise feed mae.json back in as a recording.
    raw_files = [f for f in sorted(session.glob("*.json")) if NAME_RE.search(f.name)]
    if not raw_files:
        print(f"no raw logs (*.json) in {session}", file=sys.stderr)
        return 1

    # 1. saturation screen
    ok_dir = session / "_raw_ok"
    proc_dir = session / "processed"
    for d in (ok_dir, proc_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
    rejected = []
    for f in raw_files:
        frac = stalled_fraction(json.loads(f.read_text()))
        if frac >= STALL_FRAC and not a.keep_saturated:
            rejected.append((f.name, frac))
        else:
            shutil.copy(f, ok_dir / f.name)
    kept = len(raw_files) - len(rejected)
    print(f"[validate] {len(raw_files)} raw logs, {len(rejected)} rejected as saturated "
          f"(>= {STALL_FRAC:.0%} stalled frames), {kept} kept")
    for name, frac in rejected:
        print(f"           rejected {name}: {frac:.0%} stalled")
    if kept == 0:
        return 1

    # 2. resample
    subprocess.run([sys.executable, "-m", "bam.process", "--raw", str(ok_dir),
                    "--logdir", str(proc_dir), "--dt", str(a.dt)], check=True)

    # 3. score, on adjusted temporary copies of the params
    with tempfile.TemporaryDirectory() as tmp:
        for p in sorted(Path(a.params).glob("*.json")):
            d = json.loads(p.read_text())
            if d.get("actuator") == "sts3215_12v":
                d["actuator"] = "sts3215"
            d["q_offset"] = 0.0
            (Path(tmp) / p.name).write_text(json.dumps(d))
        out = session / "mae.json"
        subprocess.run([sys.executable, "-m", "bam.mae", "--params", tmp, "--logdir", str(proc_dir),
                        "--json", str(out)], check=True)
        res = json.loads(out.read_text())

        print(f"\n[validate] MAE on {kept} logs from {session.name} (published hold-out in brackets):")
        for model in sorted(res):
            mean = res[model]["mean"] * 1000
            pub = PUBLISHED_MRAD.get(model)
            ref = f"  [{pub:.0f}]" if pub else ""
            ratio = f"  x{mean / pub:.1f}" if pub else ""
            print(f"    {model}: {mean:6.2f} +- {res[model]['std'] * 1000:5.2f} mrad{ref}{ratio}")

        best = min(res, key=lambda m: res[m]["mean"])
        chosen = best if a.plot_model == "best" else a.plot_model
        split = None
        if chosen != "none":
            if chosen not in res:
                print(f"\n  --plot-model {chosen!r}: not in {a.params}", file=sys.stderr)
                return 1
            print(f"\n[validate] rolling {chosen} out over {kept} logs and plotting...")
            split = make_plots(session, chosen, Path(tmp) / f"{chosen}.json")

    # The session mean covers two physically different regimes. Separate them before
    # drawing any conclusion: with torque off nothing drives the joint, so those samples
    # measure the model's stiction, not its tracking.
    if split is not None:
        print(f"\n[validate] {chosen} error split (mean | median over logs):")
        print(f"    torque ON  : {split['on_mean'] * 1000:7.2f} | {split['on_median'] * 1000:7.2f} mrad"
              f"   <- the only regime the policy runs in")
        if split["n_off"]:
            print(f"    torque OFF : {split['off_mean'] * 1000:7.2f} |       - mrad"
                  f"   ({split['n_off']} logs have unpowered spans)")
        print(f"    all samples: {split['mean'] * 1000:7.2f} | {split['median'] * 1000:7.2f} mrad")
        for f in split["written"]:
            print(f"    wrote {f}")

    pub = PUBLISHED_MRAD.get(chosen if split is not None else best)
    score = (split["on_mean"] if split is not None else res[best]["mean"]) * 1000
    label = "torque-on MAE" if split is not None else "MAE"
    name = chosen if split is not None else best
    if pub and score < 2.0 * pub:
        print(f"\n  -> {name}: {label} {score:.1f} mrad, within 2x of the published {pub:.0f}. The "
              f"placeholder describes these servos under power; identification can wait.")
        if split is not None and split["n_off"] and split["off_mean"] * 1000 > 10 * pub:
            print("     It does diverge with torque off: the real gearbox holds position against")
            print("     gravity and the model lets it fall, so stiction is underestimated. That")
            print("     affects backdriven behaviour (impacts, falls), not tracking. See the")
            print("     shaded spans in the lift_and_drop panels.")
    elif pub:
        print(f"\n  -> {name}: {label} {score:.1f} mrad vs published {pub:.0f}. Noticeably worse; "
              f"fit our own (see this file's docstring).")
    else:
        print(f"\n  -> {name}: {label} {score:.1f} mrad (no published reference for this model).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
