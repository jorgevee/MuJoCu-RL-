"""Push-recovery evaluation (plan §7): batched push scenarios on a saved policy.

Each scenario: walk at a pinned command for SETTLE_S, get a rectangular force
pulse on the torso (direction relative to the robot's heading), then keep
walking. Measures falls and recovery.

  fell       tilt > MAX_TILT_DEG or torso height < MIN_HEIGHT_FRAC x pre-push
             height, at any time after the push
  recovered  not fell, and |v_xy - cmd_xy| < VEL_BAND for HOLD_S continuously,
             starting within RECOVER_WINDOW_S after the push ends

The env's own training perturbation (pert_config) is a half-sine force pulse
of 0.05-0.2 s on the torso; this uses rectangular pulses so the shape differs.

Usage (from WSL, inside the `loco` env):
    python push_recovery.py ~/projects/mujoco_rl1/runs/<run_dir>
    python push_recovery.py <run_dir> --forces 0 100 200 300 --duration 0.1
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_policy import load_policy  # noqa: E402

SETTLE_S = 2.0           # walk before the push
PUSH_JITTER_S = 0.5      # push start uniform in [SETTLE_S, SETTLE_S + jitter]
AFTER_S = 4.0            # observe after the push window
MAX_TILT_DEG = 60.0
MIN_HEIGHT_FRAC = 0.5
VEL_BAND = 0.25          # m/s
HOLD_S = 0.5
RECOVER_WINDOW_S = 3.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--forces", type=float, nargs="+",
                   default=[0, 50, 100, 150, 200, 250, 300, 400])
    p.add_argument("--duration", type=float, default=0.1, help="push length, s")
    p.add_argument("--directions", type=int, default=8,
                   help="evenly spaced around the body; 0 deg = push from behind")
    p.add_argument("--repeats", type=int, default=4, help="per force x direction")
    p.add_argument("--cmd-vx", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def yaw_from_quat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return jp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    args = parse_args()
    run_dir = args.run_dir.expanduser()
    env_name, env, policy = load_policy(run_dir)
    torso = env._torso_body_id
    mass = float(env._mj_model.body_subtreemass[torso])
    dt = env.dt

    # Scenario grid: force x direction x repeat.
    angles = np.arange(args.directions) * 2 * np.pi / args.directions
    F, A, R = np.meshgrid(args.forces, angles, np.arange(args.repeats), indexing="ij")
    forces, dirs = F.ravel(), A.ravel()
    n = forces.size
    rng = np.random.default_rng(args.seed)
    push_start = SETTLE_S + rng.uniform(0, PUSH_JITTER_S, n)

    n_steps = int(round((SETTLE_S + PUSH_JITTER_S + args.duration + AFTER_S) / dt))
    cmd = jp.array([args.cmd_vx, 0.0, 0.0])
    start_step = jp.array(np.round(push_start / dt), dtype=jp.int32)
    dur_steps = max(1, int(round(args.duration / dt)))
    f_mag = jp.array(forces, dtype=jp.float32)
    f_ang = jp.array(dirs, dtype=jp.float32)

    def rollout(key, start, mag, ang):
        state = env.reset(key)

        def step(state, t):
            info = dict(state.info)
            info["command"] = cmd
            # Push direction is relative to current heading; 0 = from behind.
            yaw = yaw_from_quat(state.data.qpos[3:7])
            world = jp.array([jp.cos(yaw + ang), jp.sin(yaw + ang), 0.0])
            on = (t >= start) & (t < start + dur_steps)
            xfrc = jp.zeros((env.mjx_model.nbody, 6))
            xfrc = xfrc.at[torso, :3].set(jp.where(on, mag, 0.0) * world)
            state = state.replace(data=state.data.replace(xfrc_applied=xfrc), info=info)

            action, _ = policy(state.obs, key)
            state = env.step(state, action)
            v = env.get_local_linvel(state.data)
            out = (
                jp.linalg.norm(v[:2] - cmd[:2]),      # velocity error
                env.get_upvector(state.data)[-1],     # cos(tilt)
                state.data.qpos[2],                   # torso height
            )
            return state, out

        _, outs = jax.lax.scan(step, state, jp.arange(n_steps))
        return outs

    keys = jax.random.split(jax.random.PRNGKey(args.seed), n)
    print(f"{env_name}: {n} scenarios x {n_steps} steps, torso subtree mass {mass:.1f} kg")
    vel_err, cos_tilt, height = jax.jit(jax.vmap(rollout))(keys, start_step, f_mag, f_ang)
    vel_err, cos_tilt, height = map(np.asarray, (vel_err, cos_tilt, height))

    # Per-scenario metrics.
    hold = int(round(HOLD_S / dt))
    window = int(round(RECOVER_WINDOW_S / dt))
    min_cos = np.cos(np.radians(MAX_TILT_DEG))
    rows = []
    for i in range(n):
        s, e = int(start_step[i]), int(start_step[i]) + dur_steps
        h0 = height[i, max(0, s - 25):s].mean()
        after = slice(s, None)
        fell = bool((cos_tilt[i, after] < min_cos).any()
                    or (height[i, after] < MIN_HEIGHT_FRAC * h0).any())
        rec_time = None
        if not fell:
            ok = vel_err[i] < VEL_BAND
            for k in range(e, min(e + window, n_steps - hold) + 1):
                if ok[k:k + hold].all():
                    rec_time = (k - e) * dt
                    break
        rows.append({
            "force_N": float(forces[i]),
            "direction_deg": round(float(np.degrees(dirs[i])), 1),
            "push_start_s": round(float(push_start[i]), 3),
            "fell": fell,
            "recovered": rec_time is not None,
            "recovery_time_s": rec_time,
        })

    # Summary by force.
    print(f"\npush {args.duration * 1000:.0f} ms, cmd vx {args.cmd_vx} m/s, "
          f"{args.directions} directions x {args.repeats} repeats per force")
    print(f"{'force N':>8} {'Δv m/s':>7} {'fell':>6} {'recovered':>10} {'median t_rec':>13}")
    summary = []
    for f in args.forces:
        sub = [r for r in rows if r["force_N"] == f]
        times = [r["recovery_time_s"] for r in sub if r["recovered"]]
        entry = {
            "force_N": f,
            "delta_v": round(f * args.duration / mass, 3),
            "fall_rate": float(np.mean([r["fell"] for r in sub])),
            "recovery_rate": float(np.mean([r["recovered"] for r in sub])),
            "median_recovery_s": float(np.median(times)) if times else None,
        }
        summary.append(entry)
        med = f"{entry['median_recovery_s']:.2f} s" if times else "-"
        print(f"{f:>8.0f} {entry['delta_v']:>7.2f} {entry['fall_rate']:>6.0%} "
              f"{entry['recovery_rate']:>10.0%} {med:>13}")

    # Fall rate by direction, pooled over forces > 0.
    print("\nfall rate by push direction (forces > 0; 0° = from behind, 90° = from the right side)")
    by_dir = {}
    for d in sorted({r["direction_deg"] for r in rows}):
        sub = [r for r in rows if r["direction_deg"] == d and r["force_N"] > 0]
        by_dir[d] = float(np.mean([r["fell"] for r in sub])) if sub else None
        print(f"  {d:>6.1f}°  {by_dir[d]:.0%}")

    out = run_dir / f"push_recovery_{int(args.duration * 1000)}ms_vx{args.cmd_vx}.json"
    out.write_text(json.dumps({
        "settings": {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                     "mass_kg": mass, "settle_s": SETTLE_S, "push_jitter_s": PUSH_JITTER_S,
                     "max_tilt_deg": MAX_TILT_DEG, "min_height_frac": MIN_HEIGHT_FRAC,
                     "vel_band": VEL_BAND, "hold_s": HOLD_S,
                     "recover_window_s": RECOVER_WINDOW_S},
        "summary": summary, "by_direction": by_dir, "scenarios": rows,
    }, indent=2))
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
