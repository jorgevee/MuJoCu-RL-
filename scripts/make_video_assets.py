"""Build assets for the progress video: reward-curve chart + side-by-side clips.

Usage (from WSL, inside the `loco` env):
    python make_video_assets.py
Outputs go to <repo>/video_assets/.
"""

import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = Path.home() / "projects/mujoco_rl1/runs"
STUCK = RUNS / "Go1JoystickFlatTerrain_20261024_155833"   # 1024 envs, 50M
WALKS = RUNS / "Go1JoystickFlatTerrain_20261024_161400"   # 8192 envs, 200M
OUT = Path(__file__).resolve().parent.parent / "video_assets"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Must match SEGMENTS in eval_policy.py (start s, end s, caption).
CAPTIONS = [
    (0, 3, "STAND STILL"),
    (3, 7, "WALK FORWARD 0.5 m/s"),
    (7, 11, "WALK FORWARD 1.0 m/s"),
    (11, 15, "WALK BACKWARD 0.5 m/s"),
    (15, 19, "STEP SIDEWAYS 0.4 m/s"),
    (19, 23, "TURN 0.8 rad/s"),
    (23, 27, "STAND STILL"),
]

# Dark theme tokens (dataviz reference palette, validated for dark surface).
SURFACE, INK, INK2, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
BLUE, ORANGE = "#3987e5", "#d95926"


def load_log(run):
    return json.loads((run / "summary.json").read_text())["log"]


def plot_rewards(path, figsize, dpi=120):
    stuck, walks = load_log(STUCK), load_log(WALKS)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for log, color, label, dy in [(walks, BLUE, "200M steps · walks", 0),
                                  (stuck, ORANGE, "50M steps · stands & spins", -22)]:
        x = [e["step"] / 1e6 for e in log]
        y = [e["eval_reward"] for e in log]
        ax.plot(x, y, color=color, lw=2.5, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        ax.annotate(label, (x[-1], y[-1]), xytext=(10, dy), textcoords="offset points",
                    va="center", color=INK, fontsize=13, fontweight="bold")

    ax.annotate("breakthrough:\nlearns to walk", xy=(57, 22.3), xytext=(12, 28),
                color=INK2, fontsize=12, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.5))

    ax.set_title("Same reward function, more practice", color=INK, fontsize=20,
                 fontweight="bold", loc="left", pad=16)
    ax.set_xlabel("environment steps (millions)", color=MUTED, fontsize=13)
    ax.set_ylabel("eval reward", color=MUTED, fontsize=13)
    ax.set_xlim(0, 260)
    ax.set_ylim(0, 32)
    ax.grid(True, color=GRID, lw=1)
    ax.tick_params(colors=MUTED, labelsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#383835")
    ax.legend(loc="lower right", frameon=False, labelcolor=INK2, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def side_by_side(path, stack):
    """stack='h' (1280x480) or 'v' (640x960 for vertical shorts)."""
    def label(text):
        return (f"drawtext=fontfile={FONT}:text='{text}':fontcolor=white:fontsize=26:"
                f"box=1:boxcolor=black@0.55:boxborderw=10:x=(w-text_w)/2:y=16")

    left = f"[0:v]{label('50M steps')}[a]"
    right = f"[1:v]{label('200M steps')}[b]"
    combine = "[a][b]hstack=inputs=2[s]" if stack == "h" else "[a][b]vstack=inputs=2[s]"
    caps = ",".join(
        f"drawtext=fontfile={FONT}:text='COMMAND  {c}':fontcolor=white:fontsize=24:"
        f"box=1:boxcolor=0x2a78d6@0.85:boxborderw=12:x=(w-text_w)/2:y=h-text_h-24"
        f":enable='between(t,{a},{b})'"
        for a, b, c in CAPTIONS)
    graph = f"{left};{right};{combine};[s]{caps}[out]"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", str(STUCK / "eval_commands.mp4"), "-i", str(WALKS / "eval_commands.mp4"),
         "-filter_complex", graph, "-map", "[out]",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        check=True)


def main():
    OUT.mkdir(exist_ok=True)
    plot_rewards(OUT / "reward_curves_16x9.png", figsize=(16, 9))
    plot_rewards(OUT / "reward_curves_4x5.png", figsize=(9, 11.25))
    side_by_side(OUT / "side_by_side_horizontal.mp4", "h")
    side_by_side(OUT / "side_by_side_vertical.mp4", "v")
    shutil.copy(STUCK / "eval_commands.mp4", OUT / "stuck_50M_stands_and_spins.mp4")
    shutil.copy(WALKS / "eval_commands.mp4", OUT / "walks_200M.mp4")
    for p in sorted(OUT.iterdir()):
        print(p.name)


if __name__ == "__main__":
    main()
