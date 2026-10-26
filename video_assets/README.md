# Video assets: Part 1, "Teaching a robot dog to walk"

Regenerate everything with `python scripts/make_video_assets.py` (from WSL, `loco` env).

| File | What it is |
| --- | --- |
| `side_by_side_horizontal.mp4` | 1280×480, 26 s. 50M (left) vs 200M (right), same held commands, command caption at the bottom |
| `side_by_side_vertical.mp4` | 640×960 stacked version for Shorts/Reels/TikTok |
| `stuck_50M_stands_and_spins.mp4` | 640×480, 50M-step policy alone |
| `walks_200M.mp4` | 640×480, 200M-step policy alone (8192 envs, 96 min) |
| `walks_200M_1024envs.mp4` | 640×480, 200M-step policy, same 1024 envs as the stuck run (23 min) |
| `reward_curves_16x9.png` | 1920×1080 reward chart, dark |
| `reward_curves_4x5.png` | 1080×1350 reward chart for vertical video |

Command sequence in all clips (seconds): 0–3 stand · 3–7 forward 0.5 m/s · 7–11 forward 1.0 m/s ·
11–15 backward 0.5 m/s · 15–19 sideways 0.4 m/s · 19–23 turn 0.8 rad/s · 23–26 stand.

The two robots look similar in still frames. The difference only shows in motion: the 50M robot stays in place for every walk command.

## Script outline (~2.5 min)

**0:00 Hook** (side-by-side clip)
"Both of these robot dogs were trained with the exact same reward. One learned to walk. The other learned to stand there and spin."

**0:10 What this is**
Teaching a simulated Unitree Go1 to walk from scratch with reinforcement learning (PPO). No animations, no demonstrations: just physics, trial and error, and a reward. MuJoCo Playground + JAX on an RTX 5060 (8 GB) under WSL2.

**0:30 Setup bumps** (quick cuts)
- JAX 0.10 removed a function that Brax still uses, so training crashed on the first call. Fix: pin JAX 0.9.2.
- JAX grabs 75% of GPU memory by default. On an 8 GB card that caused out-of-memory warnings; one environment variable fixed it.

**0:50 It's fast**
~100–150k simulated steps per second on a mid-range GPU. 50 million steps in under 7 minutes.

**1:05 The twist** (stuck clip, orange line on chart)
Reward climbed to ~20 and looked great. Then we measured what the robot actually does: zero forward speed for every walk command. It learned that standing still and turning earns most of the points. **Reward going up ≠ robot learning the task.**

**1:35 The fix** (chart, breakthrough arrow)
Ran the upstream recipe: 8192 robots in parallel, 200 million steps, ~96 minutes. Around 50M steps it breaks out of the trap and starts walking.
Then we checked: the same 1024 robots as the "failed" run, just trained to 200M steps, also walks (fwd 1.0 → 0.97 m/s), and it took 23 minutes instead of 96. Fewer robots was faster on this GPU; the failed run just stopped too early.

**1:55 Result** (walking clip + table)

| Command | Target | Achieved |
| --- | --- | --- |
| forward | 1.0 m/s | 0.95 |
| backward | 0.5 m/s | 0.45 |
| sideways | 0.4 m/s | 0.35 |
| turn | 0.8 rad/s | 0.76 |

Within ~10% on every command, never fell.

**2:15 Lessons**
1. Measure behavior, not just reward.
2. Pin your versions; ML libraries break each other.
3. The training budget matters: the exact same setup that "failed" at 50M steps walked at 200M, in 23 minutes on an 8 GB GPU.

**2:30 Next time**
The real question: if we teach it in stages (stand → balance → walk → survive pushes), is it better at recovering from shoves it has never seen?
