"""sweep_all.py — run the policy_gate sweep over several checkpoint dirs (not part of the repo).

usage (from BipedRobot/):
    python S/sweep_all.py [--seeds 2] [--cmds 0.05 0.1 ...] [--legacy-imu] RUN[:step_N.pt] ...
RUN is checkpoints/<run> (or a bare run name); without :step the largest step_*.pt is used.
Load failures are reported per run, the loop continues.
"""
import argparse
import glob
import os
import re
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "BipedRobot", "src", "mujoco"))
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, "torch_rl_algorithms"))
import policy_gate as G     # noqa: E402  (the modified copy; _ROOT falls back to cwd)


def pick(spec):
    run, _, step = spec.partition(":")
    if not os.path.isdir(run):
        run = os.path.join("checkpoints", run)
    if step:
        return run, os.path.join(run, step)
    cks = [(int(re.search(r"step_(\d+)\.pt$", f).group(1)), f)
           for f in glob.glob(os.path.join(run, "step_*.pt")) if re.search(r"step_(\d+)\.pt$", f)]
    return run, (max(cks)[1] if cks else None)


def run_date(run):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(run))
    return m.group(1) if m else ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--cmds", type=float, nargs="+", default=[0.05, 0.1, 0.15, 0.2, 0.3])
    p.add_argument("--legacy-imu", action="store_true")
    a = p.parse_args()
    ok, bad = [], []
    for spec in a.runs:
        run, ck = pick(spec)
        print(f"\n=== {os.path.basename(run)} :: {os.path.basename(ck) if ck else None} ===", flush=True)
        if ck is None or not os.path.exists(ck):
            msg = "no step_*.pt checkpoint found"
            print("  SKIP:", msg); bad.append((run, msg)); continue
        try:
            policy, config = G.load_checkpoint_policy(ck)
            if a.legacy_imu:
                pass
            G.sweep(policy, config, a.cmds, [0], a.seeds, a.seconds, 0.5, legacy_imu=a.legacy_imu)
            ok.append(run)
        except Exception as e:                      # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
            hint = "" if run_date(run) >= "2026-10-11" else "  (pre-2026-10-11: try --legacy-imu)"
            print(f"  LOAD/RUN FAILED: {msg}{hint}")
            if os.environ.get("SWEEP_TB"):
                traceback.print_exc()
            bad.append((run, msg + hint))
    print("\nloadable:", [os.path.basename(r) for r in ok])
    for r, m in bad:
        print("failed:", os.path.basename(r), "-", m)


if __name__ == "__main__":
    main()
