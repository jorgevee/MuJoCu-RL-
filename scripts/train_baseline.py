"""Short baseline PPO run on a stock MuJoCo Playground Go1 task.

Toolchain smoke test (plan §12): train with the upstream Brax PPO config,
scaled down to fit an 8 GB GPU, then log throughput, save params, and
render a short video.

Usage (from WSL, inside the `loco` env):
    python train_baseline.py                      # 2M steps, 1024 envs
    python train_baseline.py --num-envs 512 --timesteps 1000000
    python train_baseline.py --impl warp          # force a physics backend
"""

import argparse
import functools
import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering

import jax
import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Go1JoystickFlatTerrain")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=1024,
                   help="must be a multiple of 256")
    p.add_argument("--num-evals", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--impl", choices=["jax", "warp"], default=None,
                   help="physics backend; default = Playground's choice")
    p.add_argument("--naconmax", type=int, default=None,
                   help="warp contact pool, shared by ALL envs (upstream: 4*8192)")
    p.add_argument("--out-dir", type=Path,
                   default=Path.home() / "projects/mujoco_rl1/runs")
    p.add_argument("--no-render", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.num_envs % 256 == 0, "--num-envs must be a multiple of 256"

    run_dir = args.out_dir / f"{args.env}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_dir}")
    print(f"jax {jax.__version__}, devices: {jax.devices()}")

    env_cfg = registry.get_default_config(args.env)
    if args.impl is not None:
        env_cfg.impl = args.impl
    if args.naconmax is not None:
        env_cfg.naconmax = args.naconmax
    print(f"physics impl: {env_cfg.get('impl', 'n/a')}, "
          f"naconmax: {env_cfg.get('naconmax', 'n/a')} "
          f"(~{env_cfg.get('naconmax', 0) / args.num_envs:.1f} contacts/env)")

    env = registry.load(args.env, config=env_cfg)
    eval_env = registry.load(args.env, config=env_cfg)

    # Upstream tuned config, scaled down. Keep the minibatch size (256 envs x
    # unroll) and gradient-steps-per-env-step ratio the same as upstream.
    ppo_params = locomotion_params.brax_ppo_config(args.env)
    ppo_params.num_timesteps = args.timesteps
    ppo_params.num_envs = args.num_envs
    ppo_params.batch_size = 256
    ppo_params.num_minibatches = args.num_envs // 256
    ppo_params.num_evals = args.num_evals

    train_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in train_params:
        del train_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks, **ppo_params.network_factory)

    log = []
    t_start = time.monotonic()
    t_prev = [t_start, 0]

    def progress(step, metrics):
        now = time.monotonic()
        dt, dsteps = now - t_prev[0], step - t_prev[1]
        sps = dsteps / dt if dsteps > 0 else float("nan")
        t_prev[0], t_prev[1] = now, step
        entry = {
            "step": int(step),
            "wall_s": round(now - t_start, 1),
            "sps_since_last": round(sps, 1),
            "eval_reward": float(metrics.get("eval/episode_reward", float("nan"))),
            "eval_reward_std": float(metrics.get("eval/episode_reward_std", float("nan"))),
        }
        log.append(entry)
        # File log survives Ctrl+C and is readable while the run is going.
        with open(run_dir / "progress.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[{entry['wall_s']:7.1f}s] step {step:>10,}  "
              f"reward {entry['eval_reward']:8.2f} ± {entry['eval_reward_std']:6.2f}  "
              f"sps {sps:,.0f}", flush=True)

    def save_params(step, make_policy, params):
        # Called by Brax at every eval; keep the latest checkpoint on disk.
        with open(run_dir / "params.pkl", "wb") as f:
            pickle.dump(jax.device_get(params), f)

    # Written up front so an interrupted run can still be evaluated.
    with open(run_dir / "config.json", "w") as f:
        json.dump({"env_cfg": env_cfg.to_dict(), "ppo": ppo_params.to_dict()},
                  f, indent=2, default=str)
    with open(run_dir / "summary.json", "w") as f:
        json.dump({"env": args.env, "status": "running"}, f, indent=2)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=registry.get_domain_randomizer(args.env),
        network_factory=network_factory,
        progress_fn=progress,
        policy_params_fn=save_params,
        seed=args.seed,
        **train_params,
    )
    total_s = time.monotonic() - t_start

    # The first progress call happens after JIT compile + the step-0 eval, and
    # the first training interval also includes compiling the training step,
    # so steady-state throughput is measured from the second eval onward.
    steady = log[1:]
    summary = {
        "env": args.env,
        "impl": env_cfg.get("impl", None),
        "num_envs": args.num_envs,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "time_to_first_eval_s": log[0]["wall_s"] if log else None,
        "total_wall_s": round(total_s, 1),
        "steady_sps": round(
            (steady[-1]["step"] - steady[0]["step"]) / (steady[-1]["wall_s"] - steady[0]["wall_s"]), 1)
            if len(steady) > 1 else None,
        "final_eval_reward": log[-1]["eval_reward"] if log else None,
        "peak_gpu_mem_gb": None,
        "log": log,
    }
    try:
        stats = jax.devices()[0].memory_stats() or {}
        summary["peak_gpu_mem_gb"] = round(stats.get("peak_bytes_in_use", 0) / 1e9, 2)
    except Exception:
        pass

    with open(run_dir / "params.pkl", "wb") as f:
        pickle.dump(jax.device_get(params), f)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary ===")
    for k, v in summary.items():
        if k != "log":
            print(f"{k:>24}: {v}")

    if not args.no_render:
        try:
            render(eval_env, env_cfg, make_inference_fn, params, run_dir)
        except Exception as e:  # don't lose the training run over rendering
            print(f"\nrendering failed ({type(e).__name__}: {e})")
            print("params are saved; see notes/lessons_learned.md for render fixes")


def render(env, env_cfg, make_inference_fn, params, run_dir, n_steps=500):
    import mediapy as media

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    policy = jax.jit(make_inference_fn(params, deterministic=True))

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    state.info["command"] = jp.array([0.5, 0.0, 0.0])  # walk forward 0.5 m/s
    rollout = [state]
    for _ in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        action, _ = policy(state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)

    every = 2
    frames = env.render(rollout[::every], camera="track", height=480, width=640)
    path = run_dir / "rollout.mp4"
    media.write_video(path, frames, fps=1.0 / env.dt / every)
    print(f"\nvideo: {path}")


if __name__ == "__main__":
    main()
