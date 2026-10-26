# Short script: "Teaching a robot dog to walk" (Part 1)

- **Format:** 1080×1920 (9:16), 30 fps, target ~2:40
- **Voiceover:** ~390 words, about 150 words/min
- **Look:** dark background (#1b1b1b, same as the chart), blue = walking policy (200M), orange = stuck policy (50M), white bold captions
- **Layout:** sim clips sit in the middle third. The top third holds the title or label, the bottom third holds captions. Keep text clear of the bottom ~250 px and the right ~120 px, where the platform UI goes.

Timestamps are estimates. Once the audio is in, we re-time every section to the real voiceover.

---

## 1. Hook (0:00–0:10)

**VO:**
> These two robot dogs were trained with the exact same reward. One learned to walk. The other learned to… stand there and spin. Here's what went wrong.

**Visuals:**
- `side_by_side_vertical.mp4`, full frame, starting at 3 s so both robots get the "forward" command
- Labels pop in over each half: orange "50M steps" (top), blue "200M steps" (bottom)
- On "stand there and spin": a 2D spinning-arrow icon bounces over the orange robot
- Caption: **SAME REWARD. DIFFERENT DOG.**

---

## 2. What this is (0:10–0:30)

**VO:**
> This is a simulated Unitree Go1, learning to walk from scratch with reinforcement learning. No animations, no demonstrations. Just physics, trial and error, and a score. Every step, the robot gets a command, like "walk forward," and decides how to move its twelve motors. Thousands of copies practice at once, on a single 8-gig gaming GPU.

**Visuals:**
- `walks_200M.mp4` in the middle third
- Animated 2D RL loop diagram drawn over the clip: **Command → Policy (brain icon) → 12 motors → Physics → Reward → back to Policy**. The arrows trace themselves in sequence.
- On "thousands of copies": the robot tile shrinks and multiplies into a grid of tiny dog icons (16 → 256 → 1024, with a counter ticking up)
- Tag chips slide in: `MuJoCo Playground` · `JAX` · `PPO` · `RTX 5060 · 8 GB`

---

## 3. Setup bumps (0:30–0:48)

**VO:**
> Getting it running had bumps. The newest JAX deleted a function the training library still uses, so it crashed instantly. Fix: pin the older version. Then JAX grabbed three-quarters of the GPU memory for itself. One environment variable fixed that.

**Visuals:** quick-cut "terminal card" graphics, styled as a code window
- Card 1: red `AttributeError: jax.device_put_replicated` shakes, gets stamped ❌, then morphs into green `jax==0.9.2 ✅`
- Card 2: GPU memory bar fills to 75% in red with "OUT OF MEMORY" flashing, then `XLA_PYTHON_CLIENT_PREALLOCATE=false` types in and the bar drops to a small green sliver

---

## 4. It's fast (0:48–1:00)

**VO:**
> Once it ran, it flew. About a hundred and fifty thousand simulation steps per second. Fifty million steps in under seven minutes.

**Visuals:**
- Big animated counter: `0 → 150,000 steps/sec` (speedometer-style gauge)
- Then a progress bar `50,000,000 steps` filling, with a stopwatch reading `6:36`

---

## 5. The twist (1:00–1:30)

**VO:**
> And the score climbed, up to about twenty. Looked great. Then we actually asked it to walk. Forward: zero. Backward: zero. Sideways: zero. It figured out that standing still and turning in place earns most of the points, without ever risking a fall. The reward went up. The robot didn't learn the task.

**Visuals:**
- `reward_curves_4x5.png` with only the orange line drawn in (animated draw-on), reward number ticking up to ~20, a green "📈 looks great!" sticker
- Hard cut to `stuck_50M_stands_and_spins.mp4`. Three scoreboard rows stamp in, one per beat: `FORWARD 0.0 m/s ❌` · `BACKWARD 0.0 m/s ❌` · `SIDEWAYS 0.0 m/s ❌`
- Caption lands in large type: **REWARD ↑ ≠ ROBOT LEARNED**

---

## 6. The fix (1:30–1:58)

**VO:**
> So we ran the official recipe: eight thousand robots, two hundred million steps, an hour and a half. Right around fifty million steps, it breaks out of the trap… and starts walking. Then the surprise: our original thousand-robot setup, just trained longer, walks too. In twenty-three minutes. It wasn't broken. We just stopped too early.

**Visuals:**
- Chart again, with the blue line drawing on past the orange one. The "breakthrough" arrow pulses at ~50M steps.
- On "starts walking": cut to `walks_200M.mp4`
- Comparison card, two columns:
  - `8192 robots · 96 min · walks ✅`
  - `1024 robots · 23 min · walks ✅` (highlighted, with a "4× faster" badge)
- Caption: **IT WASN'T BROKEN. IT NEEDED MORE PRACTICE.**

---

## 7. Result (1:58–2:15)

**VO:**
> Final result: it tracks every command within about ten percent. Forward, backward, sideways, turning. And it never fell once.

**Visuals:**
- `walks_200M.mp4` up top. Below it, animated bars fill to target, one per command:

| Command | Target | Achieved |
| --- | --- | --- |
| forward | 1.0 m/s | 0.95 |
| backward | 0.5 m/s | 0.45 |
| sideways | 0.4 m/s | 0.35 |
| turn | 0.8 rad/s | 0.76 |

- A `FALLS: 0` badge stamps in at the end

---

## 8. Lessons (2:15–2:33)

**VO:**
> Three lessons. One: measure behavior, not just reward. Two: pin your versions, because ML libraries break each other. Three: give it enough practice. The same setup that failed at fifty million steps walked at two hundred.

**Visuals:** numbered cards stack in one at a time over a dimmed `side_by_side_vertical.mp4`
1. 📏 Measure behavior, not reward
2. 📌 Pin your versions
3. ⏱️ Enough training: 50M ❌ → 200M ✅

---

## 9. Next time (2:33–2:45)

**VO:**
> Next: what if we teach it in stages: stand, balance, walk, then survive shoves? Is it better at recovering from pushes it's never seen? Follow to find out.

**Visuals:**
- Animated 4-step staircase: **Stand → Balance → Walk → Survive pushes**, with a dog icon hopping up each step
- On "shoves": a 2D arrow "pushes" the dog icon and it wobbles
- End card: **Part 2 coming soon**, plus a follow prompt

---

## Recording notes

- Read it at a natural pace. A single take is fine; I'll find the section breaks from the audio.
- Leave about half a second of silence between sections. It makes cutting easier.
- The pauses written as "…" (hook, fix) are deliberate beats. Hold them a little.
- Export WAV or high-bitrate MP3.
