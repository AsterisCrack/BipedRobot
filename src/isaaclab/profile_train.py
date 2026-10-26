"""Training-speed profiler: where does one SAC iteration (env step + learner update) spend its time?

Builds the env, wrapper and SAC exactly like train.py, then runs the Trainer.run iteration body
(agent.step -> env.step -> agent.update -> per-iteration TB logging) in four passes:

  1. baseline   un-instrumented iterations/s (what train.py would see at this num_envs)
  2. phases     wall time per phase, with torch.cuda.synchronize() at every phase edge (nested)
  3. syncs      host-device sync sites (torch.cuda.set_sync_debug_mode), grouped by project call site
  4. profiler   torch.profiler key averages (self CPU time) + cudaStreamSynchronize / memcpy counts

    python -m src.isaaclab.profile_train --task BipedRobotV2 --config_path config/config_gait_reference.yaml \
        --num_envs 1024 --out <dir>

Small on purpose so it can run next to a live training run: flat terrain by default (rough terrain makes
the first reset take minutes), a 100k replay, no checkpoints. Launch-bound costs do not scale with
num_envs and the learner does not either (fixed batch), so run two --num_envs values and fit a + b*N
to extrapolate to 4096.
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", type=str, default="BipedRobotV2")
parser.add_argument("--config_path", type=str, default="config/config_gait_reference.yaml")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--warmup", type=int, default=60, help="iterations before any measurement")
parser.add_argument("--baseline_iters", type=int, default=100)
parser.add_argument("--phase_iters", type=int, default=100)
parser.add_argument("--sync_iters", type=int, default=3)
parser.add_argument("--profiler_iters", type=int, default=10)
parser.add_argument("--buffer_size", type=int, default=100_000)
parser.add_argument("--rough", action="store_true", help="keep the config's rough terrain (slow first reset)")
parser.add_argument("--out", type=str, default=None, help="output dir (TB scratch, trace, json)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, _root)
# Appended, not prepended: torch_rl_algorithms has its own top-level `config` module, which
# would shadow this repo's `config` package.
sys.path.append(os.path.join(_root, "torch_rl_algorithms"))

import collections  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tensorboardX import SummaryWriter  # noqa: E402
from torch_rl_algorithms.algorithms.sac.model import SAC  # noqa: E402
from torch_rl_algorithms.algorithms.utils import _ScalarLogBuffer  # noqa: E402
from utils import Config  # noqa: E402
from utils.residual_home import freeze_residual_map  # noqa: E402
from src.isaaclab.common import BaseIsaacLabWrapper  # noqa: E402
from src.isaaclab.env_config import build_env_cfg  # noqa: E402
from envs.isaaclab.biped_env import BipedEnv  # noqa: E402
from envs.isaaclab.biped_env_cfg import BipedEnvCfg, BipedRobotV2EnvCfg  # noqa: E402


class Timers:
    """Sync-bracketed wall timers, keyed by phase name. Disabled = zero-overhead passthrough."""

    def __init__(self):
        self.enabled = False
        self.ms = collections.defaultdict(list)
        self.iter_ms = collections.defaultdict(float)   # sum within the current iteration

    def wrap(self, obj, attr, name):
        orig = getattr(obj, attr)
        timers = self

        def timed(*a, **k):
            if not timers.enabled:
                return orig(*a, **k)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = orig(*a, **k)
            torch.cuda.synchronize()
            timers.iter_ms[name] += 1000.0 * (time.perf_counter() - t0)
            return r

        setattr(obj, attr, timed)

    def end_iteration(self):
        for k, v in self.iter_ms.items():
            self.ms[k].append(v)
        self.iter_ms.clear()


class TimedCallable:
    """Callable-object stand-in (updaters, class-based event terms) with attribute passthrough."""

    def __init__(self, inner, timers, name):
        self.__dict__.update(_inner=inner, _timers=timers, _name=name)

    def __call__(self, *a, **k):
        t = self._timers
        if not t.enabled:
            return self._inner(*a, **k)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = self._inner(*a, **k)
        torch.cuda.synchronize()
        t.iter_ms[self._name] += 1000.0 * (time.perf_counter() - t0)
        return r

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __setattr__(self, name, value):
        setattr(self._inner, name, value)


def instrument(env, wenv, agent, timers):
    for attr in ("_pre_physics_step", "_apply_action", "_get_dones", "_get_rewards", "_reset_idx",
                 "_get_observations"):
        timers.wrap(env, attr, f"env.{attr.lstrip('_')}")
    timers.wrap(env.scene, "write_data_to_sim", "env.physics.write_data_to_sim (actuator)")
    timers.wrap(env.sim, "step", "env.physics.sim_step")
    timers.wrap(env.scene, "update", "env.physics.scene_update (sensors)")
    for i, act in enumerate(getattr(env, "bam_actuators", []) or []):
        timers.wrap(act, "compute", f"env.physics.write_data_to_sim.bam_compute[{i}]")
        timers.wrap(act, "_effective_inertia", f"env.physics.write_data_to_sim.bam_effective_inertia[{i}]")
        timers.wrap(act.servo, "step", f"env.physics.write_data_to_sim.bam_servo_step[{i}]")
    # PhysX tensor-API getters/setters the reset-time DR goes through (CPU-side buffers in GPU sim)
    view = env.robot.root_physx_view
    for meth in ("get_material_properties", "set_material_properties", "get_masses", "set_masses",
                 "get_inertias", "set_inertias", "get_coms", "set_coms"):
        try:
            timers.wrap(view, meth, f"physx_view.{meth}")
        except (AttributeError, TypeError):
            print(f"[profile] cannot instrument physx view method {meth}")
    em = env.event_manager
    for mode, cfgs in em._mode_term_cfgs.items():
        for name, term_cfg in zip(em._mode_term_names[mode], cfgs):
            term_cfg.func = TimedCallable(term_cfg.func, timers, f"events.{mode}.{name}")
    timers.wrap(wenv, "step", "wrapper.step (env.step + obs norm)")
    timers.wrap(wenv, "_process_obs", "wrapper.process_obs (norm + mirror stats)")
    timers.wrap(agent, "step", "agent.step (actor)")
    timers.wrap(agent, "update", "agent.update")
    timers.wrap(agent.replay, "store", "agent.update.replay_store")
    timers.wrap(agent, "_update", "agent.update._update (10 grad steps)")
    agent.critic_updater = TimedCallable(agent.critic_updater, timers, "agent.update._update.critic")
    agent.actor_updater = TimedCallable(agent.actor_updater, timers, "agent.update._update.actor")
    timers.wrap(agent.model, "update_targets", "agent.update._update.update_targets")


class Loop:
    """Trainer.run's per-iteration body, logging included (same buffered pattern, same log_interval)."""

    def __init__(self, wenv, agent, writer, timers, log_interval):
        self.wenv, self.agent, self.writer, self.timers = wenv, agent, writer, timers
        self.obs = wenv.start()
        self.num_workers = len(next(iter(self.obs.values()))) if isinstance(self.obs, dict) else len(self.obs)
        self.steps = 0
        self.iterations = 0
        self.log_interval = max(1, int(log_interval))
        self.scores = torch.zeros(self.num_workers, device=wenv.device)
        self.lengths = torch.zeros(self.num_workers, device=wenv.device, dtype=torch.int32)
        self.window = torch.zeros(3, device=wenv.device)
        self.buffer = _ScalarLogBuffer()
        self.last_env_log = None

    def _log(self, agent_infos, infos):
        for key, value in agent_infos.items():
            self.buffer.add(f"train/{key}", value)
        env_log = infos.get("log")
        if env_log and env_log is not self.last_env_log:
            self.last_env_log = env_log
            for key, value in env_log.items():
                self.buffer.add(key, value)
        self.scores += infos["rewards"]
        self.lengths += 1
        done = infos["resets"].to(torch.float32)
        self.window += torch.stack([(self.scores * done).sum(), (self.lengths.float() * done).sum(), done.sum()])
        self.scores *= 1.0 - done
        self.lengths.masked_fill_(infos["resets"].bool(), 0)
        if self.iterations % self.log_interval == 0:
            self.buffer.add_episodes(self.window)
            self.buffer.flush(self.writer, self.steps)
            self.window.zero_()

    def iterate(self, n):
        for _ in range(n):
            self.iterations += 1
            actions = self.agent.step(self.obs, self.steps)
            self.obs, infos = self.wenv.step(actions)
            agent_infos = self.agent.update(**infos, steps=self.steps)
            if self.timers.enabled:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                self._log(agent_infos, infos)
                torch.cuda.synchronize()
                self.timers.iter_ms["trainer.logging"] += 1000.0 * (time.perf_counter() - t0)
            else:
                self._log(agent_infos, infos)
            self.steps += self.num_workers
            self.timers.end_iteration()


def sync_sites(loop, n):
    """Count host-device syncs PyTorch can see, keyed by the innermost project frame (and the op)."""
    counts = collections.Counter()
    proj = os.path.normcase(_root)

    def show(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" not in str(message):
            return
        frames = [f for f in traceback.extract_stack()[:-1] if os.path.normcase(f.filename).startswith(proj)
                  and "profile_train" not in f.filename]
        site = "<non-project>"
        if frames:
            f = frames[-1]
            site = f"{os.path.relpath(f.filename, _root)}:{f.lineno}  {f.line.strip()[:90] if f.line else ''}"
        counts[site] += 1

    old = warnings.showwarning
    warnings.showwarning = show
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    try:
        loop.iterate(n)
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = old
    return counts


def main():
    out = args.out or os.path.join(tempfile.gettempdir(), "biped_profile")
    os.makedirs(out, exist_ok=True)
    tag = f"n{args.num_envs}{'_rough' if args.rough else ''}"

    config = Config(args.config_path)
    freeze_residual_map(config)                       # as train.py
    torch.backends.cuda.matmul.allow_tf32 = bool(config.train.tf32)   # as train.py
    torch.backends.cudnn.allow_tf32 = bool(config.train.tf32)
    if not args.rough:
        config.train.use_rough_terrain = False
    config.buffer.size = args.buffer_size
    env_cfg = build_env_cfg(config, BipedRobotV2EnvCfg if args.task == "BipedRobotV2" else BipedEnvCfg,
                            training=True, num_envs=args.num_envs)
    env_cfg.seed = config.train.seed
    torch.manual_seed(config.train.seed)
    np.random.seed(config.train.seed)

    env = BipedEnv(cfg=env_cfg)
    wenv = BaseIsaacLabWrapper(env, config,
                               symmetric_obs_stats=bool(getattr(config.train, "symmetry_augmentation", False)))
    model = SAC(env=wenv, device=env.device, config=config,
                use_history=config.train.use_history, history_size=config.train.history_size)
    agent = model.agent
    timers = Timers()
    instrument(env, wenv, agent, timers)
    writer = SummaryWriter(os.path.join(out, f"tb_{tag}"))
    loop = Loop(wenv, agent, writer, timers, config.train.log_interval)

    print(f"[profile] warmup {args.warmup} it ({args.num_envs} envs) ...", flush=True)
    loop.iterate(args.warmup)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    loop.iterate(args.baseline_iters)
    torch.cuda.synchronize()
    base_ms = 1000.0 * (time.perf_counter() - t0) / args.baseline_iters
    print(f"[profile] baseline: {base_ms:.1f} ms/it = {1000.0 / base_ms:.2f} it/s", flush=True)

    timers.enabled = True
    timers.ms.clear()
    loop.iterate(args.phase_iters)
    timers.enabled = False

    counts = sync_sites(loop, args.sync_iters) if args.sync_iters > 0 else collections.Counter()

    prof_table, prof_counts = "", {}
    if args.profiler_iters > 0:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            loop.iterate(args.profiler_iters)
        ka = prof.key_averages()
        prof_table = ka.table(sort_by="self_cpu_time_total", row_limit=40)
        for e in ka:
            if any(s in e.key for s in ("Synchronize", "cudaMemcpy", "cudaLaunchKernel", "cudaStreamWait")):
                prof_counts[e.key] = {"count_per_it": e.count / args.profiler_iters,
                                      "self_cpu_ms_per_it": e.self_cpu_time_total / 1000.0 / args.profiler_iters}
        prof.export_chrome_trace(os.path.join(out, f"trace_{tag}.json"))

    # ---- report ---------------------------------------------------------------------------------
    lines = [f"PROFILE {tag}: baseline {base_ms:.1f} ms/it ({1000.0 / base_ms:.2f} it/s); "
             f"phase times are sync-bracketed means over {args.phase_iters} it (nested, ms/it)"]
    rows = sorted(((k, float(np.mean(v)), float(np.percentile(v, 90))) for k, v in timers.ms.items()),
                  key=lambda r: r[0])
    for k, mean, p90 in rows:
        depth = k.count(".") - 1
        lines.append(f"  {'  ' * max(depth, 0)}{k:<70s} {mean:8.2f}  (p90 {p90:7.2f})")
    n_sync = sum(counts.values())
    lines.append(f"SYNCS seen by torch per iteration: {n_sync / max(args.sync_iters, 1):.0f}; top sites (per it):")
    for site, c in counts.most_common(45):
        lines.append(f"  {c / args.sync_iters:7.1f}  {site}")
    if prof_counts:
        lines.append("CUDA API calls per iteration (torch.profiler):")
        for k, v in sorted(prof_counts.items(), key=lambda kv: -kv[1]["count_per_it"]):
            lines.append(f"  {k:<40s} {v['count_per_it']:9.1f} calls  {v['self_cpu_ms_per_it']:8.2f} ms CPU")
    report = "\n".join(lines)
    print(report)
    print(prof_table)
    with open(os.path.join(out, f"report_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n\n" + prof_table)
    with open(os.path.join(out, f"phases_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"baseline_ms": base_ms, "phases": {k: m for k, m, _ in rows},
                   "syncs": dict(counts), "cuda_api": prof_counts}, f, indent=1)
    print(f"[profile] wrote {out}", flush=True)
    sys.stdout.flush()
    os._exit(0)   # Isaac's shutdown can hang; results are already written


if __name__ == "__main__":
    main()
