"""Evaluate a saved baseline policy with a fixed sequence of held commands.

The joystick env resamples a random command every ~5 s, so a plain rollout
doesn't show whether the policy tracks a given command. Here the command is
pinned for each segment and the measured velocity is compared against it.

Usage (from WSL, inside the `loco` env):
    python eval_policy.py ~/projects/mujoco_rl1/runs/<run_dir>
    python eval_policy.py <run_dir> --no-render
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict
from mujoco_playground import registry

# (name, [vx m/s, vy m/s, yaw rad/s], seconds)
SEGMENTS = [
    ("stand",      [0.0, 0.0, 0.0], 3.0),
    ("fwd 0.5",    [0.5, 0.0, 0.0], 4.0),
    ("fwd 1.0",    [1.0, 0.0, 0.0], 4.0),
    ("back 0.5",   [-0.5, 0.0, 0.0], 4.0),
    ("left 0.4",   [0.0, 0.4, 0.0], 4.0),
    ("turn 0.8",   [0.0, 0.0, 0.8], 4.0),
    ("stand",      [0.0, 0.0, 0.0], 3.0),
]
SETTLE_S = 1.0  # ignore the first second of each segment when averaging


def render_frame(state):
    """Host copy of just the fields render needs. Full warp states carry large
    preallocated contact buffers, so keeping ~1000 of them on GPU OOMs."""
    d = state.data
    return SimpleNamespace(data=SimpleNamespace(
        qpos=np.asarray(d.qpos), qvel=np.asarray(d.qvel),
        mocap_pos=np.asarray(d.mocap_pos), mocap_quat=np.asarray(d.mocap_quat),
        xfrc_applied=np.asarray(d.xfrc_applied)))


def load_policy(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    env_name = summary["env"]
    env_cfg = config_dict.ConfigDict(cfg["env_cfg"])
    env = registry.load(env_name, config=env_cfg)

    ppo = cfg["ppo"]
    preprocess = (running_statistics.normalize if ppo.get("normalize_observations")
                  else lambda x, y: x)
    net_kwargs = dict(ppo.get("network_factory", {}))
    for k in ("policy_hidden_layer_sizes", "value_hidden_layer_sizes"):
        if k in net_kwargs:
            net_kwargs[k] = tuple(net_kwargs[k])
    network = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=preprocess, **net_kwargs)
    make_inference_fn = ppo_networks.make_inference_fn(network)

    with open(run_dir / "params.pkl", "rb") as f:
        params = pickle.load(f)
    policy = jax.jit(make_inference_fn(params, deterministic=True))
    return env_name, env, policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-render", action="store_true")
    args = p.parse_args()
    run_dir = args.run_dir.expanduser()

    env_name, env, policy = load_policy(run_dir)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    linvel = jax.jit(env.get_local_linvel)
    gyro = jax.jit(env.get_gyro)

    rng = jax.random.PRNGKey(args.seed)
    state = jit_reset(rng)
    keep_frames = not args.no_render
    rollout = [render_frame(state)] if keep_frames else []
    results = []
    fell_at = None

    for name, cmd, seconds in SEGMENTS:
        n = int(round(seconds / env.dt))
        settle = int(round(SETTLE_S / env.dt))
        measured = []
        for i in range(n):
            state.info["command"] = jp.array(cmd)  # pin it; env would resample
            act_rng, rng = jax.random.split(rng)
            action, _ = policy(state.obs, act_rng)
            state = jit_step(state, action)
            if keep_frames:
                rollout.append(render_frame(state))
            if i >= settle:
                v, w = linvel(state.data), gyro(state.data)
                measured.append([float(v[0]), float(v[1]), float(w[2])])
            if float(state.done) > 0:
                fell_at = name
                break
        m = np.array(measured) if measured else np.full((1, 3), np.nan)
        results.append({
            "segment": name,
            "cmd": cmd,
            "measured_mean": m.mean(axis=0).round(3).tolist(),
            "abs_err_mean": np.abs(m - np.array(cmd)).mean(axis=0).round(3).tolist(),
        })
        if fell_at:
            break

    print(f"\n{env_name}  run: {run_dir.name}")
    print(f"{'segment':<10} {'cmd (vx,vy,wz)':<22} {'measured (vx,vy,wz)':<26} abs err")
    for r in results:
        cmd = ", ".join(f"{x:+.2f}" for x in r["cmd"])
        meas = ", ".join(f"{x:+.2f}" for x in r["measured_mean"])
        err = ", ".join(f"{x:.2f}" for x in r["abs_err_mean"])
        print(f"{r['segment']:<10} {cmd:<22} {meas:<26} {err}")
    print(f"\nfell: {'during ' + fell_at if fell_at else 'no'}")

    (run_dir / "eval_commands.json").write_text(
        json.dumps({"results": results, "fell_during": fell_at}, indent=2))

    if not args.no_render:
        import mediapy as media
        every = 2
        frames = env.render(rollout[::every], camera="track", height=480, width=640)
        path = run_dir / "eval_commands.mp4"
        media.write_video(path, frames, fps=1.0 / env.dt / every)
        print(f"video: {path}")


if __name__ == "__main__":
    main()
