# Lessons learned, important notes and lessons we learned so far

## Local setup (as of 2026-10-24)

### Where things live

| What | Location |
| --- | --- |
| Plan, notes, git repo (Windows side) | `C:\Users\jorge\Documents\tech_blogs\mujocu_rl1` |
| Training / code workspace (WSL side) | `~/projects/mujoco_rl1` (= `/home/jorge/projects/mujoco_rl1`) |
| Python env | micromamba env `loco` → `/home/jorge/micromamba/envs/loco` |
| MuJoCo Menagerie (robot models) | auto-downloaded by Playground on first `registry.load` |
| Warp kernel cache | `~/.cache/warp/1.18.0` |

Activate before working:

```bash
micromamba activate loco
cd ~/projects/mujoco_rl1
```

### Hardware / OS

- Windows 11 + WSL2 Ubuntu 24.04
- GPU: RTX 5060, 8 GB VRAM (Blackwell, sm_120)
- NVIDIA driver 610.x, CUDA 13.3 supported (driver installed on **Windows only**, never inside WSL)
- Windows desktop already uses ~1.3 GB VRAM → **~6.5 GB usable for training**

### Software versions (known working)

- Python 3.11 (in `loco`)
- JAX **0.9.2** (pinned; 0.10.x breaks Brax, see gotchas), flax 0.12.6, Brax 0.14.2 → `jax.devices()` shows `CudaDevice(id=0)`
- MuJoCo Playground via `pip install playground` (pulls in MuJoCo, MJX, Brax)
- Warp 1.18.0 (CUDA Toolkit 13.4), initialized automatically by Playground
- Exact versions pinned in `~/projects/mujoco_rl1/requirements.lock` (`pip freeze`, 2026-10-24)

### What we verified

1. `nvidia-smi` inside WSL sees the RTX 5060.
2. JAX sees the GPU (`CudaDevice`).
3. Playground loads Go1. Available Go1 envs: `Go1JoystickFlatTerrain`, `Go1JoystickRoughTerrain`, `Go1Getup`, `Go1Handstand`, `Go1Footstand`.
4. `Go1JoystickFlatTerrain`: `action_size = 12`, observations are a dict:
   - `state` (48): what the policy sees (proprioception + command + last action)
   - `privileged_state` (123): extra simulator info, for the critic only

### First baseline run (2026-10-24): toolchain works end to end

`scripts/train_baseline.py`, `Go1JoystickFlatTerrain`, physics impl **warp**, 1024 envs, seed 0.
Run dir: `~/projects/mujoco_rl1/runs/Go1JoystickFlatTerrain_20261024_154728/`

- Ran 2.46M steps (Brax rounds up to whole eval intervals) in 74 s total
- ~21 s to the first eval (JIT compile), plus ~30 s more compiling the training step in the first interval
- **Steady-state throughput ≈ 100k env steps/s on the RTX 5060**
- Eval reward 0.001 → 11.0, only starting to climb at the end
- Peak JAX GPU memory 0.45 GB (may not include Warp's allocations; check `nvidia-smi` during a run)

### 50M-step run (2026-10-24): reward went up, but the robot doesn't walk

Run dir: `~/projects/mujoco_rl1/runs/Go1JoystickFlatTerrain_20261024_155833/` (1024 envs, warp)

- ~148k steps/s steady, 50M steps in 6.6 min
- Eval reward 14 by 6M steps, then a slow creep to 19.8 by 50M
- `scripts/eval_policy.py` (commands held fixed) shows **zero linear velocity for every walk command**. The policy stands perfectly still and only turns in place (0.62 rad/s achieved of 0.8 commanded). It never falls.
- Standing still is a local optimum: it collects the pose reward, the full stand-still tracking reward, and partial tracking credit for small commands (exp(-v²/0.25)).
- **Lesson: reward ≈ 20 looked like progress but meant "stand and turn".** Always check command tracking directly, as plan §2 warns.
- Next step to test: the exact upstream config (8192 envs, 200M steps) to see whether our 1024-env scaling is the cause.

### 200M-step upstream config (2026-10-24): IT WALKS

Run dir: `~/projects/mujoco_rl1/runs/Go1JoystickFlatTerrain_20261024_161400/` (8192 envs, warp, default naconmax 32768 ≈ 4 contacts/env)

- Final eval reward 28.2. The breakthrough came between 46M (19.7) and 69M (25.0) steps
- `eval_policy.py`: tracks every command within ~5–12%, never falls:
  fwd 0.5 → 0.47, fwd 1.0 → 0.95, back 0.5 → 0.45, left 0.4 → 0.35, turn 0.8 → 0.76
- **This is our working baseline policy.**
- **Slow:** 5737 s (~96 min). Throughput was 24k steps/s early, rising to 47k late, vs 148k at 1024 envs. Speed improves as robots stop flailing, which fits the contact pool/solver being the bottleneck (benchmark pending).
- At 46M steps the 8192-env run had the same reward (19.7) as the 1024-env run at 50M (19.8). **The 50M run may simply have been too short, not too small.** Next test: 1024 envs × 200M steps (≈ 25 min at 148k sps).

### 1024 envs × 200M (2026-10-24): walks too, 4× faster

Run dir: `~/projects/mujoco_rl1/runs/Go1JoystickFlatTerrain_20261024_181831/`

- 22.6 min total at ~155k steps/s (vs 96 min for 8192 envs)
- Final reward 27.6. Breakthrough later than with 8192 envs: ~96M (20.7) → 117M (25.3)
- Tracking as good as the 8192 run: fwd 1.0 → 0.97, back 0.5 → 0.47, left 0.4 → 0.36, turn 0.8 → 0.75, never fell
- **Confirmed: the 50M run was simply too short.** 1024 envs is fine, it just needs >~120M steps.
- **Per-run budget for the curriculum experiments: 1024 envs × 200M steps ≈ 23 min.** The plan's 20M/run (§6) is far too small for this task; the direct baseline wouldn't even walk.

### Push-recovery calibration on the baseline (2026-10-25)

`scripts/push_recovery.py` on the 1024×200M baseline (`..._181831`), which was trained **without** pushes.
Rectangular 100 ms torso push, cmd vx 0.5 m/s, 8 directions × 4 repeats (32 scenarios per force). Torso subtree mass 12.7 kg.

| force | Δv | fell | recovered | median recovery time |
|---|---|---|---|---|
| 0–200 N | ≤1.57 m/s | 0% | 100% | ≤0.38 s |
| 250 N | 1.96 m/s | 6% | 94% | 0.54 s |
| 300 N | 2.35 m/s | 59% | 41% | 0.56 s |
| 400 N | 3.14 m/s | 94% | 6% | 0.90 s |

- **The cliff is at Δv ≈ 2–2.5 m/s.** Below that the baseline is already near-perfect. The held-out suite needs to sit around the cliff (≈250–400 N per 100 ms) or there's nothing to compare.
- Side and diagonal pushes are worse than pushes from behind (11% falls from behind vs 29–32% from the side/diagonal, pooled over forces).
- The env's training perturbation (`pert_config`) is a half-sine pulse with Δv = velocity_kick/π. With the default velocity_kick of [0, 3] that's at most **≈0.95 m/s**, far below the cliff. Strong held-out pushes are therefore naturally outside training conditions.
- Fall criterion is our own (tilt > 60° or height < 50% of pre-push). The env only terminates when flipped past 90°. Still to do: visually spot-check some "fell" cases.
- Also floods `linesearch iterations limit reached` (ls_iterations=5, upstream setting).

## Lessons / gotchas

- **Keep JAX in its own env.** `python3` was coming from an existing micromamba env `torch-dev`. Don't mix PyTorch and JAX in one env, because their CUDA libraries conflict.
- **Don't work from `/mnt/c/...`.** The Windows drive is slow from WSL. Keep code, venvs and caches in the Linux home directory (`~`). Edit from Windows via `\\wsl.localhost\Ubuntu\home\jorge\projects\mujoco_rl1` or VS Code's WSL extension.
- **JAX grabs 75% of VRAM at startup by default.** On 8 GB with the desktop and Warp also using memory, this caused `CUDA_ERROR_OUT_OF_MEMORY` warnings. Fix:
  ```bash
  echo 'export XLA_PYTHON_CLIENT_PREALLOCATE=false' >> ~/.bashrc
  ```
  (Applied 2026-10-24. Note: `source ~/.bashrc` deactivates the env, so re-run `micromamba activate loco` afterwards.)
- **The first `registry.load` is slow.** Cloning MuJoCo Menagerie took ~10 minutes. It only happens once.
- **Playground uses the MJX-Warp backend** (Warp banner on load). Keep this in mind for the JAX-vs-Warp benchmark (plan §3).
- **JAX 0.10 breaks Brax 0.14.2 (latest on PyPI as of 2026-10-24).** JAX 0.10.0 removed `jax.device_put_replicated`, and Brax PPO still calls it (`AttributeError` at `ppo.train`). Fix: `pip install "jax[cuda13]==0.9.2" "flax<0.12.8"` (flax 0.12.8 requires JAX ≥ 0.10, so this resolves to flax 0.12.6). Re-check when Brax gets a new release.
- **`Could not get kernel mode driver version ... X.Y.Z` is harmless.** WSL reports the driver as `610.88`, a format JAX doesn't parse.
- **The joystick env resamples the command every ~5 s** (exponential timer in `step`). Setting `state.info["command"]` once at reset doesn't hold. Pin it before every step when evaluating (see `scripts/eval_policy.py`).
- **Don't keep full Warp states on GPU for rendering.** Each one carries `naconmax`-sized contact buffers, and about 1300 of them OOM'd 8 GB. Copy `qpos/qvel/mocap_pos/mocap_quat/xfrc_applied` to host instead.
- **"solver iterations limit reached - please increase iterations beyond 1"** floods the log. The Go1 model runs the solver at 1 iteration for speed (upstream setting), so the warning is expected for now. Revisit if contacts look wrong.
- **8 GB VRAM means lowering `num_envs`.** Upstream PPO configs assume thousands of parallel envs. Start at 256–1024 locally.
