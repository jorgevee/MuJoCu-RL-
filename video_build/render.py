"""Render the 9:16 short: 2D motion graphics + sim clips + captions, synced to the voiceover.

    video_build/.venv/Scripts/python video_build/render.py               # full video
    video_build/.venv/Scripts/python video_build/render.py --still 61.5  # one frame -> out/still_61.50.png
    video_build/.venv/Scripts/python video_build/render.py --from 52 --to 72

Timing comes from words.json (run transcribe.py then align.py first).
"""
import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import skia

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "video_assets"
BUILD = ROOT / "video_build"
OUT = BUILD / "out"
AUDIO = ASSETS / "mujocu_audio1.wav"

W, H, FPS = 1080, 1920, 30
TAIL = 1.6  # seconds of end card after the voiceover

# ---------------------------------------------------------------- palette (matches the chart)
def hexc(h, a=255):
    h = h.lstrip("#")
    return skia.Color(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)

BG, BG2 = "#141413", "#1f1f1d"
INK, INK2, MUTED, GRID = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a"
BLUE, ORANGE = "#3987e5", "#e0622c"
GREEN, RED, YELLOW = "#3ecf8e", "#ef4f4f", "#ffd84d"
CARD = "#232321"

# ---------------------------------------------------------------- fonts
FONTS = Path("C:/Windows/Fonts")
TF = {
    "black": skia.Typeface.MakeFromFile(str(FONTS / "seguibl.ttf")),
    "bold": skia.Typeface.MakeFromFile(str(FONTS / "segoeuib.ttf")),
    "semi": skia.Typeface.MakeFromFile(str(FONTS / "seguisb.ttf")),
    "reg": skia.Typeface.MakeFromFile(str(FONTS / "segoeui.ttf")),
    "mono": skia.Typeface.MakeFromFile(str(FONTS / "consola.ttf")),
    "monob": skia.Typeface.MakeFromFile(str(FONTS / "consolab.ttf")),
    "sym": skia.Typeface.MakeFromFile(str(FONTS / "seguisym.ttf")),
}
_font_cache = {}


def font(kind, size):
    key = (kind, round(size))
    if key not in _font_cache:
        f = skia.Font(TF[kind], round(size))
        f.setEdging(skia.Font.Edging.kAntiAlias)
        f.setSubpixel(True)
        _font_cache[key] = f
    return _font_cache[key]


# ---------------------------------------------------------------- easing / timing helpers
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def ease_out(x):
    return 1 - (1 - x) ** 3


def ease_in_out(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_back(x, s=1.9):
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def lerp(a, b, x):
    return a + (b - a) * x


# ---------------------------------------------------------------- word timing
WORDS = json.loads((BUILD / "words.json").read_text(encoding="utf-8"))
AUDIO_END = WORDS[-1]["e"]
DURATION = AUDIO_END + TAIL


def _n(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def cue(phrase, after=0.0, end=False):
    """Start time of the first occurrence of `phrase` in the voiceover at/after `after`."""
    ps = [_n(p) for p in phrase.split()]
    for i in range(len(WORDS) - len(ps) + 1):
        if WORDS[i]["s"] >= after - 1e-6 and all(_n(WORDS[i + k]["w"]) == ps[k] for k in range(len(ps))):
            return WORDS[i + len(ps) - 1]["e"] if end else WORDS[i]["s"]
    raise KeyError(phrase)


SECTIONS = []
for w in WORDS:
    if len(SECTIONS) <= w["sec"]:
        SECTIONS.append(w["s"])
# scene boundaries: each scene starts a hair before its first word
SCENE_START = [0.0] + [s - 0.25 for s in SECTIONS[1:]] + [DURATION]


# ---------------------------------------------------------------- drawing primitives
def paint(color, alpha=1.0, stroke=None, cap=None, **kw):
    if isinstance(color, str):
        color = hexc(color)
    p = skia.Paint(AntiAlias=True, Color=color, **kw)
    p.setAlphaf(p.getAlphaf() * clamp(alpha))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(stroke)
        p.setStrokeCap(cap or skia.Paint.kRound_Cap)
        p.setStrokeJoin(skia.Paint.kRound_Join)
    return p


def rrect(c, x, y, w, h, r, color, alpha=1.0, stroke=None):
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r), paint(color, alpha, stroke))


def text_w(s, f):
    return f.measureText(s)


def text(c, s, x, y, f, color=INK, alpha=1.0, align="left", shadow=False, outline=0):
    """Draw text with its baseline at y. align: left|center|right."""
    w = f.measureText(s)
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    if shadow:
        sp = paint("#000000", 0.55 * alpha)
        sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 6))
        c.drawString(s, x, y + 5, f, sp)
    if outline:
        c.drawString(s, x, y, f, paint("#000000", alpha, stroke=outline))
    c.drawString(s, x, y, f, paint(color, alpha))
    return w


def pill(c, s, cx, y, f, fg, bg, alpha=1.0, pad=(26, 14), scale=1.0, stroke=None):
    """Rounded label centered at cx with top at y. Returns the box width."""
    tw = f.measureText(s)
    m = f.getMetrics()
    th = -m.fAscent + m.fDescent
    bw, bh = tw + 2 * pad[0], th + 2 * pad[1]
    c.save()
    c.translate(cx, y + bh / 2)
    c.scale(scale, scale)
    rrect(c, -bw / 2, -bh / 2, bw, bh, bh / 2, bg, alpha)
    if stroke:
        rrect(c, -bw / 2, -bh / 2, bw, bh, bh / 2, stroke, alpha, stroke=3)
    c.drawString(s, -tw / 2, -bh / 2 + pad[1] - m.fAscent, f, paint(fg, alpha))
    c.restore()
    return bw


def partial_path(path, x):
    """The first fraction x of a path (for draw-on animations)."""
    if x >= 1:
        return path
    meas = skia.PathMeasure(path, False)
    out = skia.Path()
    total, acc = 0.0, []
    while True:
        acc.append(meas.getLength())
        total += acc[-1]
        if not meas.nextContour():
            break
    meas = skia.PathMeasure(path, False)
    remaining = total * clamp(x)
    for L in acc:
        if remaining <= 0:
            break
        meas.getSegment(0, min(L, remaining), out, True)
        remaining -= L
        meas.nextContour()
    return out


def check(c, cx, cy, s, color=GREEN, alpha=1.0, x=1.0):
    p = skia.Path()
    p.moveTo(cx - 0.5 * s, cy)
    p.lineTo(cx - 0.15 * s, cy + 0.35 * s)
    p.lineTo(cx + 0.55 * s, cy - 0.4 * s)
    c.drawPath(partial_path(p, x), paint(color, alpha, stroke=s * 0.22))


def cross(c, cx, cy, s, color=RED, alpha=1.0, x=1.0):
    p = skia.Path()
    p.moveTo(cx - 0.4 * s, cy - 0.4 * s)
    p.lineTo(cx + 0.4 * s, cy + 0.4 * s)
    p.moveTo(cx + 0.4 * s, cy - 0.4 * s)
    p.lineTo(cx - 0.4 * s, cy + 0.4 * s)
    c.drawPath(partial_path(p, x), paint(color, alpha, stroke=s * 0.22))


def badge(c, kind, cx, cy, r, alpha=1.0, scale=1.0):
    """Filled circle with a check or cross."""
    c.save()
    c.translate(cx, cy)
    c.scale(scale, scale)
    col = GREEN if kind == "ok" else RED
    c.drawCircle(0, 0, r, paint(col, alpha))
    (check if kind == "ok" else cross)(c, 0, 0, r * 1.0, "#111111", alpha)
    c.restore()


def arrow(c, x1, y1, x2, y2, color, alpha=1.0, width=6, x=1.0, head=22):
    """Straight arrow drawn on to fraction x."""
    ex, ey = lerp(x1, x2, x), lerp(y1, y2, x)
    c.drawLine(x1, y1, ex, ey, paint(color, alpha, stroke=width))
    if x > 0.05:
        a = math.atan2(y2 - y1, x2 - x1)
        p = skia.Path()
        p.moveTo(ex + math.cos(a) * 4, ey + math.sin(a) * 4)
        p.lineTo(ex - head * math.cos(a - 0.45), ey - head * math.sin(a - 0.45))
        p.lineTo(ex - head * math.cos(a + 0.45), ey - head * math.sin(a + 0.45))
        p.close()
        c.drawPath(p, paint(color, alpha))


def dog(c, cx, cy, s, color=INK, alpha=1.0, phase=0.0, tilt=0.0):
    """Little side-view robot dog icon, centered on its body. `phase` animates the legs."""
    c.save()
    c.translate(cx, cy)
    c.rotate(tilt)
    c.scale(s / 100, s / 100)
    p = paint(color, alpha)
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-50, -16, 100, 30), 12, 12), p)
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(42, -26, 26, 22), 7, 7), p)
    lp = paint(color, alpha, stroke=9)
    for i, hx in enumerate((-38, -24, 26, 40)):
        sw = math.sin(phase + (i % 2) * math.pi) * 10
        kx, ky = hx + 8 + sw * 0.4, 18
        c.drawLine(hx, 8, kx, ky, lp)
        c.drawLine(kx, ky, hx + sw, 40, lp)
    c.restore()


# ---------------------------------------------------------------- video clips
class Clip:
    """Sequential ffmpeg reader that resamples a clip to the output fps."""

    def __init__(self, name):
        self.path = ASSETS / name
        self.w, self.h = map(int, subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(self.path)], capture_output=True, text=True).stdout.strip().split(","))
        self.proc, self.idx, self.img = None, -1, None

    def _open(self, start_frame):
        if self.proc:
            self.proc.kill()
        self.proc = subprocess.Popen(
            ["ffmpeg", "-v", "quiet", "-ss", f"{start_frame / FPS:.4f}", "-i", str(self.path),
             "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
            stdout=subprocess.PIPE, bufsize=10 ** 8)
        self.idx = start_frame - 1

    def frame(self, t):
        """skia.Image at clip time t (seconds); clamps at the last frame."""
        want = max(0, int(round(t * FPS)))
        if self.proc is None or want < self.idx or want > self.idx + 60:
            self._open(want)
        while self.idx < want:
            buf = self.proc.stdout.read(self.w * self.h * 4)
            if len(buf) < self.w * self.h * 4:  # past the end: hold the last frame
                break
            self.idx += 1
            arr = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 4)
            self.img = skia.Image.fromarray(arr, colorType=skia.kRGBA_8888_ColorType)
        return self.img


_clips = {}


def clip(key, name=None):
    """One reader per usage key, so two scenes using the same file don't fight over seeking."""
    if key not in _clips:
        _clips[key] = Clip(name or key)
    return _clips[key]


SAMPLING = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)


def draw_clip(c, img, x, y, w, h, radius=28, alpha=1.0, border=None, border_w=6, dim=0.0):
    if img is None:
        return
    rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), radius, radius)
    c.save()
    c.clipRRect(rr, True)
    p = skia.Paint(AntiAlias=True)
    p.setAlphaf(clamp(alpha))
    c.drawImageRect(img, skia.Rect.MakeXYWH(x, y, w, h), SAMPLING, p)
    if dim:
        c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), paint("#000000", dim * alpha))
    c.restore()
    if border:
        c.drawRRect(rr, paint(border, alpha, stroke=border_w))


# Command schedule baked into every eval clip (see video_assets/README.md)
CLIP_COMMANDS = [(0, 3, "STAND STILL"), (3, 7, "WALK FORWARD 0.5 m/s"), (7, 11, "WALK FORWARD 1.0 m/s"),
                 (11, 15, "WALK BACKWARD 0.5 m/s"), (15, 19, "STEP SIDEWAYS 0.4 m/s"),
                 (19, 23, "TURN 0.8 rad/s"), (23, 27, "STAND STILL")]


def command_at(ct):
    return next((s for a, b, s in CLIP_COMMANDS if a <= ct < b), "STAND STILL")


def command_tag(c, ct, cx, y, alpha=1.0, color=BLUE):
    f = font("bold", 34)
    pill(c, "COMMAND  " + command_at(ct), cx, y, f, INK, color, alpha, pad=(24, 10))


# ---------------------------------------------------------------- shared chrome
def background(c, t):
    c.clear(hexc(BG))
    # slow drifting glow so static moments still feel alive
    gx = W / 2 + math.sin(t * 0.25) * 220
    gy = 760 + math.cos(t * 0.18) * 260
    sh = skia.GradientShader.MakeRadial(skia.Point(gx, gy), 900, [hexc(BG2), hexc(BG)])
    c.drawRect(skia.Rect.MakeWH(W, H), skia.Paint(Shader=sh))
    # faint dot grid
    p = paint(GRID, 0.6)
    for yy in range(60, H, 60):
        for xx in range(60, W, 60):
            c.drawCircle(xx, yy, 1.6, p)


def header(c, t, start, label, num):
    """Section label pill near the top: '02 · THE PROJECT'."""
    a = ease_out(prog(t, start, 0.35))
    f = font("bold", 30)
    s = f"{num:02d}  ·  {label}"
    pill(c, s, W / 2, 168 - (1 - a) * 30, f, INK2, CARD, a, pad=(28, 12), stroke=GRID)


def progress_bar(c, t):
    x = clamp(t / AUDIO_END)
    c.drawRect(skia.Rect.MakeXYWH(0, 0, W, 8), paint(GRID))
    c.drawRect(skia.Rect.MakeXYWH(0, 0, W * x, 8), paint(BLUE))


def big_title(c, lines, cx, y, t, start, size=92, gap=1.12, stagger=0.12, colors=None):
    """Stacked headline lines that pop in one after another."""
    f = font("black", size)
    for i, ln in enumerate(lines):
        a = prog(t, start + i * stagger, 0.35)
        if a <= 0:
            continue
        sc = ease_back(a)
        c.save()
        c.translate(cx, y + i * size * gap)
        c.scale(sc, sc)
        text(c, ln, 0, 0, f, (colors or [INK] * len(lines))[i], min(1, a * 2), "center", shadow=True)
        c.restore()


# ---------------------------------------------------------------- captions
def build_caption_chunks():
    chunks, cur = [], []
    for i, w in enumerate(WORDS):
        cur.append(w)
        nxt = WORDS[i + 1] if i + 1 < len(WORDS) else None
        chars = sum(len(x["w"]) + 1 for x in cur)
        brk = (nxt is None or nxt["sec"] != w["sec"] or w["w"][-1] in ".,:;?!…\""
               or nxt["s"] - w["e"] > 0.35 or len(cur) >= 4 or chars + len(nxt["w"]) > 22)
        if brk:
            chunks.append(cur)
            cur = []
    for i, ch in enumerate(chunks):
        start = ch[0]["s"]
        end = ch[-1]["e"] + 0.35
        if i + 1 < len(chunks):
            end = min(end, chunks[i + 1][0]["s"])
        chunks[i] = (start, end, ch)
    return chunks


CHUNKS = build_caption_chunks()
CAPTION_Y = 1640


def captions(c, t):
    for start, end, ch in CHUNKS:
        if start - 0.05 <= t < end:
            break
    else:
        return
    f = font("black", 70)
    disp = [w["w"].upper().strip('"') for w in ch]
    widths = [f.measureText(d) for d in disp]
    space = f.measureText(" ")
    total = sum(widths) + space * (len(disp) - 1)
    a = prog(t, start - 0.05, 0.12)
    sc = lerp(0.88, 1.0, ease_back(a))
    c.save()
    c.translate(W / 2, CAPTION_Y)
    c.scale(sc, sc)
    x = -total / 2
    for w, d, wd in zip(ch, disp, widths):
        active = w["s"] <= t < (w["e"] + 0.08)
        col = YELLOW if active else INK
        text(c, d, x, 0, f, col, min(1, a * 1.5), outline=12)
        x += wd + space
    c.restore()


# ---------------------------------------------------------------- reward chart
LOGS = json.loads((BUILD / "reward_logs.json").read_text())


class Chart:
    def __init__(self, x, y, w, h, xmax=220, ymax=30):
        self.x, self.y, self.w, self.h, self.xmax, self.ymax = x, y, w, h, xmax, ymax

    def pt(self, sx, sy):
        return self.x + sx / self.xmax * self.w, self.y + self.h - sy / self.ymax * self.h

    def axes(self, c, alpha=1.0):
        f = font("semi", 28)
        for v in (0, 10, 20, 30):
            _, yy = self.pt(0, v)
            c.drawLine(self.x, yy, self.x + self.w, yy, paint(GRID, alpha, stroke=2))
            text(c, str(v), self.x - 18, yy + 10, f, MUTED, alpha, "right")
        for v in range(0, self.xmax + 1, 50):
            xx, _ = self.pt(v, 0)
            text(c, f"{v}M", xx, self.y + self.h + 46, f, MUTED, alpha, "center")
        text(c, "reward", self.x, self.y - 26, font("bold", 30), MUTED, alpha)
        text(c, "training steps", self.x + self.w, self.y + self.h + 92, font("bold", 30), MUTED, alpha, "right")

    def line(self, c, key, color, x=1.0, alpha=1.0, upto=None):
        """Draw a reward curve on to fraction x of its x-range. Returns the pen tip."""
        data = LOGS[key]
        xmax = data[-1][0] if upto is None else upto
        cut = xmax * clamp(x)
        p = skia.Path()
        tip = self.pt(*data[0])
        p.moveTo(*tip)
        for (x0, y0), (x1, y1) in zip(data, data[1:]):
            if x0 >= cut:
                break
            f = clamp((cut - x0) / (x1 - x0))
            tip = self.pt(lerp(x0, x1, f), lerp(y0, y1, f))
            p.lineTo(*tip)
        glow = paint(color, 0.25 * alpha, stroke=18)
        c.drawPath(p, glow)
        c.drawPath(p, paint(color, alpha, stroke=7))
        for sx, sy in data:
            if sx <= cut + 1e-6:
                c.drawCircle(*self.pt(sx, sy), 7, paint(color, alpha))
        c.drawCircle(*tip, 12, paint(color, alpha))
        c.drawCircle(*tip, 5, paint(INK, alpha))
        return tip


# ================================================================ scenes
def scene_hook(c, t):
    s0 = 0.0
    big_title(c, ["SAME REWARD.", "DIFFERENT DOG."], W / 2, 170, t, s0 + 0.1, size=84,
              colors=[INK, YELLOW])
    x, y, w, h = 160, 320, 760, 1140
    a = ease_out(prog(t, s0, 0.4))
    img = clip("hook", "side_by_side_vertical.mp4").frame(3.0 + t - s0)
    draw_clip(c, img, x, y + (1 - a) * 80, w, h, alpha=a)
    # colour-code the halves once the voiceover tells us which is which
    t_walk, t_other = cue("One learned to walk"), cue("The other learned")
    half = h / 2
    ab = ease_out(prog(t, t_walk, 0.3))
    if ab > 0:
        rrect(c, x, y + half, w, half, 28, BLUE, ab, stroke=8)
        pill(c, "WALKS", x + w - 120, y + half + 30, font("black", 40), "#0d1b2e", BLUE, ab,
             scale=ease_back(ab))
    ao = ease_out(prog(t, t_other, 0.3))
    if ao > 0:
        rrect(c, x, y, w, half, 28, ORANGE, ao, stroke=8)
        lab = "STANDS" if t < cue("and spin") else "SPINS"
        pill(c, lab, x + w - 120, y + 30, font("black", 40), "#2a1206", ORANGE, ao, scale=ease_back(ao))
        # spinning arrow ring over the stuck dog
        sp = prog(t, cue("and spin") - 0.2, 0.3)
        if sp > 0:
            cx, cy = x + w / 2, y + half * 0.45
            c.save()
            c.translate(cx, cy)
            c.rotate(t * 240)
            ring = skia.Path()
            ring.addArc(skia.Rect.MakeXYWH(-150, -60, 300, 120), 20, 280)
            c.drawPath(ring, paint(ORANGE, sp, stroke=10))
            c.restore()
    wrong = prog(t, cue("Here's what went wrong"), 0.25)
    if wrong > 0:
        pill(c, "WHAT WENT WRONG?", W / 2, y + half - 40, font("black", 52), "#111111", YELLOW, wrong,
             scale=ease_back(wrong))


def scene_project(c, t):
    s0 = SCENE_START[1]
    header(c, t, s0, "THE PROJECT", 1)
    big_title(c, ["TEACH A ROBOT DOG", "TO WALK FROM SCRATCH"], W / 2, 300, t, s0 + 0.1, size=64,
              colors=[INK, BLUE])
    x, y, w, h = 70, 430, 940, 705
    a = ease_out(prog(t, s0, 0.4))
    t_copies = cue("Thousands of copies")
    img = clip("project", "walks_200M.mp4").frame(3.0 + t - s0)
    grow = ease_in_out(prog(t, t_copies, 2.4))
    if grow <= 0:
        draw_clip(c, img, x, y + (1 - a) * 60, w, h, alpha=a)
    else:
        # the one robot multiplies: tile the same clip 1 -> 32 across (1024 copies)
        n = 2 ** round(lerp(0, 5, grow))
        rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), 28, 28)
        c.save()
        c.clipRRect(rr, True)
        m = skia.Matrix()
        m.setScaleTranslate(w / n / img.width(), h / n / img.height(), x, y)
        sh = img.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, SAMPLING, m)
        c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), skia.Paint(Shader=sh))
        c.restore()
        num = f"{n * n:,}"
        pill(c, f"{num} ROBOTS", W / 2, y + h - 50, font("black", 58), "#111111", YELLOW,
             scale=ease_back(prog(t, t_copies, 0.3)), pad=(30, 8))

    # Region under the clip: "no animations" -> RL loop -> hardware chips
    ry = 1185
    t_no, t_every = cue("No animations"), cue("Every step")
    if t_no <= t < t_every:
        f = font("black", 46)
        items = [("ANIMATIONS", cue("animations"), False), ("DEMOS", cue("demonstrations"), False),
                 ("PHYSICS", cue("physics"), True), ("TRIAL & ERROR", cue("trial"), True),
                 ("A SCORE", cue("score", after=t_no), True)]
        pos = [(300, ry + 20), (780, ry + 20), (220, ry + 170), (560, ry + 170), (900, ry + 170)]
        for (lab, ts, ok), (cx, cy) in zip(items, pos):
            k = prog(t, ts - 0.1, 0.3)
            if k <= 0:
                continue
            col = GREEN if ok else RED
            bw = pill(c, lab, cx, cy, f, INK, CARD, k, scale=ease_back(k), stroke=col, pad=(22, 10))
            if not ok:
                st = ease_out(prog(t, ts + 0.15, 0.25))
                c.drawLine(cx - bw / 2 + 14, cy + 40, cx - bw / 2 + 14 + (bw - 28) * st, cy + 40,
                           paint(RED, k, stroke=7))
    elif t_every <= t < t_copies + 0.4:
        out = 1 - prog(t, t_copies, 0.4)
        f = font("black", 38)
        nodes = {"COMMAND": (190, ry + 40, cue("command,")), "POLICY": (540, ry + 40, cue("decides")),
                 "12 MOTORS": (890, ry + 40, cue("twelve")),
                 "PHYSICS": (890, ry + 230, cue("motors.") + 0.15),
                 "SCORE": (540, ry + 230, cue("motors.") + 0.45)}
        cols = {"COMMAND": BLUE, "POLICY": YELLOW, "12 MOTORS": INK2, "PHYSICS": INK2, "SCORE": GREEN}
        edges = [("COMMAND", "POLICY"), ("POLICY", "12 MOTORS"), ("12 MOTORS", "PHYSICS"),
                 ("PHYSICS", "SCORE"), ("SCORE", "POLICY")]
        for a_, b_ in edges:
            (x1, y1, _), (x2, y2, tb) = nodes[a_], nodes[b_]
            if a_ == "SCORE":
                tb = nodes["SCORE"][2] + 0.25
            k = ease_out(prog(t, tb - 0.25, 0.3))
            if k <= 0:
                continue
            dx, dy = x2 - x1, y2 - y1
            L = math.hypot(dx, dy)
            off = 120 if dy == 0 else 52
            arrow(c, x1 + dx / L * off, y1 + dy / L * off, x2 - dx / L * off, y2 - dy / L * off,
                  MUTED, out, width=6, x=k)
        for lab, (cx, cy, ts) in nodes.items():
            k = prog(t, ts - 0.1, 0.3)
            if k > 0:
                pill(c, lab, cx, cy - 34, f, INK, CARD, k * out, scale=ease_back(k), stroke=cols[lab])
        # a dot travelling around the loop once it's closed
        lk = prog(t, nodes["SCORE"][2] + 0.5, 1.2)
        if 0 < lk < 1:
            pts = [(540, ry + 40), (890, ry + 40), (890, ry + 230), (540, ry + 230), (540, ry + 40)]
            seg = lk * 4
            i = min(3, int(seg))
            px, py = lerp(pts[i][0], pts[i + 1][0], seg - i), lerp(pts[i][1], pts[i + 1][1], seg - i)
            c.drawCircle(px, py, 12, paint(YELLOW, out))
    t_gpu = cue("single 8-gig")
    if t >= t_copies + 0.3:
        f = font("bold", 38)
        chips = [("MuJoCo Playground", BLUE, t_copies + 0.4), ("JAX + PPO", BLUE, t_copies + 0.8),
                 ("RTX 5060 · 8 GB", YELLOW, t_gpu)]
        for i, (lab, col, ts) in enumerate(chips):
            k = prog(t, ts, 0.3)
            if k > 0:
                pill(c, lab, W / 2, ry + 10 + i * 100, f, INK, CARD, k, scale=ease_back(k), stroke=col)


def terminal(c, x, y, w, h, title, lines, alpha=1.0, shake=0.0):
    c.save()
    c.translate(shake, 0)
    rrect(c, x, y, w, h, 24, "#0e0e0d", alpha)
    rrect(c, x, y, w, h, 24, GRID, alpha, stroke=3)
    for i, col in enumerate((RED, YELLOW, GREEN)):
        c.drawCircle(x + 40 + i * 34, y + 38, 10, paint(col, alpha))
    text(c, title, x + w / 2, y + 48, font("semi", 28), MUTED, alpha, "center")
    f = font("mono", 33)
    for i, (s, col) in enumerate(lines):
        text(c, s, x + 36, y + 120 + i * 50, f, col, alpha)
    c.restore()


def typed(s, t, start, cps=38):
    n = int(clamp((t - start) * cps, 0, len(s)))
    return s[:n]


def scene_bumps(c, t):
    s0 = SCENE_START[2]
    header(c, t, s0, "SETUP BUMPS", 2)
    t_crash, t_fix = cue("crashed instantly"), cue("Fix:")
    t_mem, t_var = cue("Then JAX grabbed"), cue("One environment variable")
    # card 1: the JAX 0.10 crash
    a = ease_out(prog(t, s0, 0.35))
    dim1 = 0.45 if t >= t_mem else 1.0
    shake = math.sin(t * 90) * 14 * (1 - prog(t, t_crash, 0.45)) if t >= t_crash else 0
    lines = [(typed("$ python train.py", t, s0 + 0.3), INK)]
    if t >= cue("newest JAX"):
        lines.append((typed("  jax 0.10.0 (newest)", t, cue("newest JAX")), MUTED))
    if t >= t_crash:
        lines += [("AttributeError: module 'jax'", RED), ("  has no 'device_put_replicated'", RED)]
    if t >= t_fix:
        lines.append((typed('$ pip install "jax==0.9.2"', t, t_fix, 30), GREEN))
    if t >= cue("older version.") + 0.2:
        lines.append(("  training started", GREEN))
    terminal(c, 60, 260 + (1 - a) * 60, 960, 470, "brax + jax", lines, a * dim1, shake)
    k = prog(t, t_crash, 0.3)
    if t_crash <= t < t_fix:
        badge(c, "bad", 920, 330, 54, k, ease_back(k))
    k2 = prog(t, cue("older version.") + 0.2, 0.3)
    if k2 > 0:
        badge(c, "ok", 920, 330, 54, k2 * dim1, ease_back(k2))

    # card 2: JAX grabbing 75% of VRAM
    b = ease_out(prog(t, t_mem - 0.2, 0.35))
    if b <= 0:
        return
    x, y, w = 60, 790 + (1 - b) * 60, 960
    rrect(c, x, y, w, 560, 24, "#0e0e0d", b)
    rrect(c, x, y, w, 560, 24, GRID, b, stroke=3)
    text(c, "GPU MEMORY", x + 40, y + 70, font("black", 40), INK, b)
    text(c, "RTX 5060 · 8 GB", x + w - 40, y + 70, font("semi", 32), MUTED, b, "right")
    bx, by, bw, bh = x + 40, y + 120, w - 80, 90
    rrect(c, bx, by, bw, bh, 18, CARD, b)
    desk = 1.3 / 8
    c.save()
    c.clipRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(bx, by, bw, bh), 18, 18), True)
    c.drawRect(skia.Rect.MakeXYWH(bx, by, bw * desk, bh), paint(MUTED, b))
    grab = 0.75 * ease_out(prog(t, cue("grabbed"), 0.8))
    fixed = ease_in_out(prog(t, cue("fixed that"), 0.6))
    jax_frac = lerp(grab, 0.45 / 8, fixed)
    col = RED if fixed < 0.5 else GREEN
    c.drawRect(skia.Rect.MakeXYWH(bx + bw * desk, by, bw * jax_frac, bh), paint(col, b))
    c.restore()
    f = font("bold", 30)
    text(c, "desktop", bx, by + bh + 44, f, MUTED, b)
    if jax_frac > 0.02:
        lab = f"JAX: {jax_frac * 8:.1f} GB" + (" (75%)" if fixed == 0 and grab > 0.74 else "")
        text(c, lab, bx + bw * (desk + jax_frac), by + bh + 44, f, col, b, "right")
    if grab > 0.7 and fixed < 0.3 and t < t_var:
        if int(t * 4) % 2 == 0:
            pill(c, "CUDA_ERROR_OUT_OF_MEMORY", W / 2, by + 180, font("monob", 36), INK, RED, b)
    if t >= t_var:
        s = typed("export XLA_PYTHON_CLIENT_PREALLOCATE=false", t, t_var, 44)
        f2 = font("mono", 30)
        text(c, "$ " + s, x + 40, by + 230, f2, GREEN, b)
    if fixed >= 1:
        badge(c, "ok", x + w - 70, y + 70 - 14, 40, 1, ease_back(prog(t, cue("fixed that") + 0.6, 0.3)))


def scene_fast(c, t):
    s0 = SCENE_START[3]
    header(c, t, s0, "IT'S FAST", 3)
    a = ease_out(prog(t, s0, 0.4))
    cx, cy, r = W / 2, 780, 360
    # speedometer arc
    rect = skia.Rect.MakeXYWH(cx - r, cy - r, 2 * r, 2 * r)
    bg = skia.Path()
    bg.addArc(rect, 180, 180)
    c.drawPath(bg, paint(CARD, a, stroke=48, cap=skia.Paint.kButt_Cap))
    v = ease_out(prog(t, cue("About a hundred"), cue("second.") - cue("About a hundred")))
    arc = skia.Path()
    arc.addArc(rect, 180, 180 * v)
    sh = skia.GradientShader.MakeSweep(cx, cy, [hexc(BLUE), hexc(BLUE), hexc(GREEN), hexc(YELLOW)],
                                       [0, 0.5, 0.75, 1.0])
    p = paint(BLUE, a, stroke=48, cap=skia.Paint.kButt_Cap)
    p.setShader(sh)
    c.drawPath(arc, p)
    for i in range(11):
        ang = math.radians(180 + 18 * i)
        c.drawLine(cx + math.cos(ang) * (r - 50), cy + math.sin(ang) * (r - 50),
                   cx + math.cos(ang) * (r - 76), cy + math.sin(ang) * (r - 76), paint(MUTED, a, stroke=4))
    ang = math.radians(180 + 180 * v)
    c.drawLine(cx, cy, cx + math.cos(ang) * (r - 90), cy + math.sin(ang) * (r - 90), paint(INK, a, stroke=10))
    c.drawCircle(cx, cy, 22, paint(INK, a))
    text(c, f"{int(150000 * v):,}", cx, cy + 140, font("black", 130), INK, a, "center", shadow=True)
    text(c, "simulation steps / second", cx, cy + 205, font("bold", 40), MUTED, a, "center")
    # 50M steps progress bar + stopwatch
    t50 = cue("Fifty million steps in")
    b = ease_out(prog(t, t50 - 0.3, 0.35))
    if b > 0:
        y = 1120
        rrect(c, 90, y, 900, 330, 28, CARD, b)
        text(c, "50,000,000 STEPS", 140, y + 80, font("black", 52), INK, b)
        fill = ease_in_out(prog(t, t50, cue("minutes.") + 0.4 - t50))
        rrect(c, 140, y + 120, 800, 56, 28, "#0e0e0d", b)
        if fill > 0.02:
            rrect(c, 140, y + 120, 800 * fill, 56, 28, GREEN, b)
        secs = int(396 * fill)
        text(c, f"{secs // 60}:{secs % 60:02d}", 140, y + 270, font("black", 84), YELLOW if fill >= 1 else INK, b)
        text(c, "min training time", 400, y + 262, font("bold", 36), MUTED, b)


def scene_twist(c, t):
    s0 = SCENE_START[4]
    header(c, t, s0, "THE TWIST", 4)
    t_ask, t_up = cue("Then we actually asked"), cue("The reward went up.")
    if t < t_ask:
        a = ease_out(prog(t, s0, 0.35))
        ch = Chart(150, 420, 840, 900, xmax=60)
        ch.axes(c, a)
        x = ease_in_out(prog(t, s0 + 0.3, cue("twenty.") - s0))
        tip = ch.line(c, "stuck", ORANGE, x, a)
        r = 19.78 * x if x < 1 else 19.78
        text(c, f"{r:.1f}", tip[0] - 20, tip[1] - 40, font("black", 72), ORANGE, a, "right")
        k = prog(t, cue("Looked great."), 0.3)
        if k > 0:
            pill(c, "LOOKS GREAT!", 470, 520, font("black", 54), "#06180f", GREEN, k, scale=ease_back(k))
        return
    # the stuck robot, under walk commands
    a = ease_out(prog(t, t_ask, 0.3))
    ct = 3.0 + (t - t_ask)
    dim = 0.6 * prog(t, t_up, 0.3)
    rows_out = 1 - prog(t, t_up - 0.1, 0.25)
    draw_clip(c, clip("stuck", "stuck_50M_stands_and_spins.mp4").frame(ct), 70, 300, 940, 705,
              alpha=a, border=ORANGE, dim=dim)
    command_tag(c, ct, W / 2, 940, a * (1 - dim), ORANGE)
    rows = [("FORWARD", cue("Forward:"), "0.0 m/s", "bad"), ("BACKWARD", cue("Backward:"), "0.0 m/s", "bad"),
            ("SIDEWAYS", cue("Sideways:"), "0.0 m/s", "bad"),
            ("TURN IN PLACE", cue("turning in place"), "0.6 rad/s", "ok")]
    for i, (lab, ts, val, kind) in enumerate(rows):
        k = prog(t, ts - 0.05, 0.25)
        if k <= 0:
            continue
        y = 1060 + i * 115
        al = k * rows_out
        c.save()
        c.translate(W / 2, y + 45)
        c.scale(lerp(1.3, 1, ease_out(k)), lerp(1.3, 1, ease_out(k)))
        rrect(c, -440, -45, 880, 96, 20, CARD, al)
        text(c, lab, -400, 18, font("black", 46), INK, al)
        text(c, val, 290, 18, font("black", 46), RED if kind == "bad" else GREEN, al, "right")
        badge(c, kind, 380, 2, 32, al)
        c.restore()
    if t >= t_up:
        big_title(c, ["REWARD WENT UP"], W / 2, 1180, t, t_up, size=96, colors=[GREEN])
        big_title(c, ["ROBOT DIDN'T", "LEARN THE TASK"], W / 2, 1350, t, cue("The robot didn't"), size=96,
                  colors=[RED, RED])


def scene_fix(c, t):
    s0 = SCENE_START[5]
    header(c, t, s0, "THE FIX", 5)
    t_right, t_walk = cue("Right around"), cue("and starts walking")
    t_sur, t_broken = cue("Then the surprise"), cue("It wasn't broken.")
    if t < t_right:
        big_title(c, ["THE OFFICIAL", "RECIPE"], W / 2, 400, t, s0 + 0.1, size=100, colors=[INK, BLUE])
        stats = [("8,192", "robots in parallel", cue("eight thousand")),
                 ("200M", "training steps", cue("two hundred million")),
                 ("96 min", "of training", cue("an hour and"))]
        for i, (big, small, ts) in enumerate(stats):
            k = prog(t, ts - 0.05, 0.3)
            if k <= 0:
                continue
            y = 700 + i * 250
            c.save()
            c.translate(0, (1 - ease_out(k)) * 40)
            rrect(c, 110, y, 860, 210, 28, CARD, k)
            text(c, big, 160, y + 135, font("black", 110), BLUE, k)
            text(c, small, 930, y + 125, font("bold", 42), INK2, k, "right")
            c.restore()
        return
    if t < t_walk:
        ch = Chart(150, 420, 840, 900)
        ch.axes(c)
        ch.line(c, "stuck", ORANGE)
        x = ease_in_out(prog(t, t_right - 0.1, t_walk - t_right + 0.2))
        ch.line(c, "walk8192", BLUE, x)
        k = prog(t, cue("breaks out"), 0.4)
        if k > 0:
            bx, by = ch.pt(57, 22.3)
            c.drawCircle(bx, by, 40 + 30 * math.sin(t * 8) ** 2, paint(YELLOW, k, stroke=6))
            pill(c, "BREAKTHROUGH", bx + 40, by - 190, font("black", 44), "#111111", YELLOW, k,
                 scale=ease_back(k))
        return
    if t < t_sur:
        a = ease_out(prog(t, t_walk, 0.25))
        ct = 7.0 + (t - t_walk)
        draw_clip(c, clip("fix_walk", "walks_200M.mp4").frame(ct), 70, 380, 940, 705, alpha=a, border=BLUE)
        command_tag(c, ct, W / 2, 1020, a)
        pill(c, "IT WALKS", W / 2, 1180, font("black", 90), "#0d1b2e", BLUE, a,
             scale=ease_back(prog(t, t_walk, 0.35)))
        return
    if t < t_broken:
        big_title(c, ["SAME 1,024-ROBOT SETUP", "JUST TRAINED LONGER"], W / 2, 300, t, t_sur, size=60,
                  colors=[INK, YELLOW])
        a = ease_out(prog(t, cue("our original"), 0.3))
        ct = 7.0 + (t - t_sur)
        draw_clip(c, clip("fix_1024", "walks_200M_1024envs.mp4").frame(ct), 120, 430, 840, 630,
                  alpha=a, border=GREEN)
        k = prog(t, cue("walks too."), 0.3)
        if k > 0:
            pill(c, "WALKS TOO", W / 2, 990, font("black", 52), "#06180f", GREEN, k, scale=ease_back(k))
        # training-time bars
        y = 1150
        text(c, "TRAINING TIME", 110, y, font("black", 40), MUTED, a)
        k1 = ease_out(prog(t, cue("our original"), 0.8))
        rrect(c, 110, y + 30, 860 * k1, 90, 20, BLUE, a)
        text(c, "8,192 robots · 96 min", 140, y + 90, font("black", 40), "#0d1b2e" if k1 > 0.6 else INK, a)
        k2 = ease_out(prog(t, cue("twenty-three"), 0.5))
        if k2 > 0:
            rrect(c, 110, y + 150, 860 * 23 / 96 * k2, 90, 20, GREEN, k2)
            text(c, "1,024 robots · 23 min", 360, y + 210, font("black", 40), GREEN, k2)
            k3 = prog(t, cue("minutes.", after=cue("twenty-three")), 0.3)
            if k3 > 0:
                pill(c, "4× FASTER", 830, y + 270, font("black", 50), "#111111", YELLOW, k3, scale=ease_back(k3))
        return
    # it wasn't broken: we stopped too early
    ch = Chart(150, 640, 840, 760)
    ch.axes(c)
    ch.line(c, "walk1024", GREEN)
    tip = ch.line(c, "stuck", ORANGE)
    k = prog(t, t_broken + 0.2, 0.3)
    c.drawLine(tip[0], tip[1] - 60, tip[0], ch.y + ch.h, paint(YELLOW, k, stroke=5))
    pill(c, "WE STOPPED HERE", tip[0] + 150, tip[1] - 140, font("black", 40), "#111111", YELLOW, k,
         scale=ease_back(k))
    big_title(c, ["IT WASN'T BROKEN."], W / 2, 330, t, t_broken, size=80)
    big_title(c, ["IT NEEDED MORE PRACTICE."], W / 2, 450, t, cue("We just stopped"), size=62,
              colors=[YELLOW])


def scene_result(c, t):
    s0 = SCENE_START[6]
    header(c, t, s0, "THE RESULT", 6)
    a = ease_out(prog(t, s0, 0.35))
    ct = 3.0 + (t - s0) * 1.0
    draw_clip(c, clip("result", "walks_200M.mp4").frame(ct), 120, 250, 840, 630, alpha=a, border=BLUE)
    command_tag(c, ct, W / 2, 815, a)
    k = prog(t, cue("ten percent."), 0.3)
    if k > 0:
        pill(c, "WITHIN ~10% OF EVERY COMMAND", W / 2, 920, font("black", 44), "#111111", YELLOW, k,
             scale=ease_back(k))
    rows = [("FORWARD", 1.0, 0.95, "m/s", cue("Forward,", after=s0)),
            ("BACKWARD", 0.5, 0.45, "m/s", cue("backward,", after=s0)),
            ("SIDEWAYS", 0.4, 0.35, "m/s", cue("sideways,", after=s0)),
            ("TURN", 0.8, 0.76, "rad/s", cue("turning.", after=s0))]
    for i, (lab, tgt, got, unit, ts) in enumerate(rows):
        y = 1050 + i * 120
        r = prog(t, ts - 0.1, 0.25)
        if r <= 0:
            continue
        text(c, lab, 110, y + 52, font("black", 40), INK, r)
        bx, bw = 400, 420
        rrect(c, bx, y + 14, bw, 50, 25, CARD, r)
        f = ease_out(prog(t, ts, 0.7))
        rrect(c, bx, y + 14, max(50, bw * got / tgt * f), 50, 25, GREEN, r)
        c.drawLine(bx + bw, y + 2, bx + bw, y + 76, paint(INK, r, stroke=5))
        text(c, f"{got * f:.2f}", 975, y + 54, font("black", 44), GREEN, r, "right")
        text(c, f"target {tgt} {unit}", bx + bw, y + 104, font("semi", 26), MUTED, r, "right")
    k = prog(t, cue("never fell once"), 0.3)
    if k > 0:
        c.save()
        c.translate(W / 2, 560)
        c.rotate(-6)
        pill(c, "FALLS: 0", 0, -60, font("black", 80), "#06180f", GREEN, k, scale=ease_back(k) * 1.0)
        c.restore()


def scene_lessons(c, t):
    s0 = SCENE_START[7]
    header(c, t, s0, "3 LESSONS", 7)
    big_title(c, ["WHAT WE LEARNED"], W / 2, 330, t, s0 + 0.1, size=82)
    cards = [("1", "Measure behavior,", "not just reward", cue("One:", after=s0), BLUE),
             ("2", "Pin your versions", "ML libraries break each other", cue("Two:", after=s0), YELLOW),
             ("3", "Give it enough practice", "", cue("Three:", after=s0 + 1), GREEN)]
    for i, (n, l1, l2, ts, col) in enumerate(cards):
        k = prog(t, ts - 0.1, 0.35)
        if k <= 0:
            continue
        y = 450 + i * 330
        c.save()
        c.translate((1 - ease_out(k)) * 300, 0)
        rrect(c, 70, y, 940, 290, 32, CARD, k)
        rrect(c, 70, y, 940, 290, 32, col, k * 0.8, stroke=4)
        c.drawCircle(170, y + 145, 62, paint(col, k))
        text(c, n, 170, y + 177, font("black", 90), "#111111", k, "center")
        text(c, l1, 270, y + 130, font("black", 54), INK, k)
        if l2:
            text(c, l2, 270, y + 200, font("bold", 40), INK2, k)
        c.restore()
    # lesson 3 detail: 50M fails, 200M walks
    k = prog(t, cue("failed at"), 0.3)
    if k > 0:
        y = 450 + 2 * 330
        w50 = text(c, "50M", 270, y + 210, font("black", 54), ORANGE, k)
        badge(c, "bad", 270 + w50 + 40, y + 192, 28, k, ease_back(k))
        k2 = prog(t, cue("walked at"), 0.3)
        ax = 270 + w50 + 100
        arrow(c, ax, y + 192, ax + 110, y + 192, MUTED, k2, width=6, x=ease_out(k2), head=18)
        if k2 > 0:
            w200 = text(c, "200M", ax + 140, y + 210, font("black", 54), GREEN, k2)
            badge(c, "ok", ax + 140 + w200 + 40, y + 192, 28, k2, ease_back(k2))


def scene_next(c, t):
    s0 = SCENE_START[8]
    header(c, t, s0, "NEXT TIME", 8)
    t_end = cue("Follow to find out")
    big_title(c, ["WHAT IF WE TEACH IT", "IN STAGES?"], W / 2, 330, t, cue("what if"), size=72,
              colors=[INK, YELLOW])
    steps = [("STAND", cue("stand,")), ("BALANCE", cue("balance,")), ("WALK", cue("walk,", after=s0)),
             ("SURVIVE PUSHES", cue("survive shoves?"))]
    sw, sh = 230, 150
    x0, base = 80, 1330
    fade = 1 - 0.6 * prog(t, t_end, 0.3)
    reached = -1
    for i, (lab, ts) in enumerate(steps):
        k = prog(t, ts - 0.15, 0.3)
        if k <= 0:
            continue
        reached = i
        x, y = x0 + i * sw, base - (i + 1) * sh
        hgt = (i + 1) * sh * ease_out(k)
        col = [BLUE, BLUE, GREEN, YELLOW][i]
        rrect(c, x + 6, base - hgt, sw - 12, hgt, 18, CARD, fade)
        rrect(c, x + 6, base - hgt, sw - 12, 16, 8, col, fade)
        f = font("black", 34 if i < 3 else 30)
        if i == 3:
            text(c, "SURVIVE", x + sw / 2, base - hgt + 70, f, INK, k * fade, "center")
            text(c, "PUSHES", x + sw / 2, base - hgt + 110, f, INK, k * fade, "center")
        else:
            text(c, lab, x + sw / 2, base - hgt + 70, f, INK, k * fade, "center")
    if reached >= 0:
        # dog hops from step to step
        ts = steps[reached][1] - 0.15
        hop = ease_in_out(prog(t, ts, 0.4))
        prev = max(reached - 1, 0)
        px, py = x0 + prev * sw + sw / 2, base - (prev + 1) * sh
        nx, ny = x0 + reached * sw + sw / 2, base - (reached + 1) * sh
        dx, dy = lerp(px, nx, hop), lerp(py, ny, hop) - math.sin(hop * math.pi) * 90
        tilt, push = 0.0, prog(t, cue("shoves?") - 0.1, 0.25)
        if push > 0:
            k = prog(t, cue("shoves?"), 0.9)
            tilt = math.sin(k * math.pi * 3) * 22 * (1 - k)
            arrow(c, dx - 230, dy - 60, dx - 80 - 30 * push, dy - 60, RED, push * fade, width=12,
                  x=ease_out(push), head=34)
        dog(c, dx, dy - 50, 120, INK, fade, phase=t * 9 if 0 < hop < 1 else 0, tilt=tilt)
    k = prog(t, cue("Is it better"), 0.3)
    if k > 0 and t < t_end:
        big_title(c, ["BETTER AT RECOVERING FROM", "PUSHES IT'S NEVER SEEN?"], W / 2, 1450, t,
                  cue("Is it better"), size=50, colors=[INK, INK])
    if t >= t_end:
        k = prog(t, t_end, 0.35)
        c.drawRect(skia.Rect.MakeWH(W, H), paint("#000000", 0.88 * k))
        big_title(c, ["PART 2"], W / 2, 860, t, t_end, size=150, colors=[YELLOW])
        big_title(c, ["COMING SOON"], W / 2, 980, t, t_end + 0.15, size=80)
        k2 = prog(t, t_end + 0.6, 0.35)
        if k2 > 0:
            pulse = 1 + 0.05 * math.sin((t - t_end) * 7)
            pill(c, "FOLLOW FOR PART 2", W / 2, 1090, font("black", 54), "#111111", YELLOW, k2,
                 scale=ease_back(k2) * pulse)


SCENES = [scene_hook, scene_project, scene_bumps, scene_fast, scene_twist, scene_fix,
          scene_result, scene_lessons, scene_next]


# ---------------------------------------------------------------- frame assembly
def render_frame(c, t):
    background(c, t)
    i = max(k for k in range(len(SCENES)) if SCENE_START[k] <= t)
    # quick fade-through at scene cuts
    fade_in = prog(t, SCENE_START[i], 0.18) if i else 1.0
    c.saveLayerAlpha(None, int(255 * fade_in))
    SCENES[i](c, t)
    c.restore()
    progress_bar(c, t)
    captions(c, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, nargs="*")
    ap.add_argument("--from", dest="t0", type=float, default=0.0)
    ap.add_argument("--to", dest="t1", type=float, default=DURATION)
    ap.add_argument("--out", default=str(OUT / "mujoco_short.mp4"))
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    surface = skia.Surface(W, H)
    c = surface.getCanvas()

    if args.still is not None:
        for t in args.still:
            render_frame(c, t)
            path = OUT / f"still_{t:06.2f}.png"
            surface.makeImageSnapshot().save(str(path), skia.kPNG)
            print(path)
        return

    t0, t1 = args.t0, min(args.t1, DURATION)
    partial = (t0, t1) != (0.0, DURATION)
    out = args.out if not partial else str(OUT / f"preview_{t0:.0f}_{t1:.0f}.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-ss", f"{t0:.3f}", "-i", str(AUDIO),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-af", "apad", "-t", f"{t1 - t0:.3f}",
           "-movflags", "+faststart", out]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n0, n1 = int(round(t0 * FPS)), int(round(t1 * FPS))
    for n in range(n0, n1):
        render_frame(c, n / FPS)
        enc.stdin.write(surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType).tobytes())
        if n % 150 == 0:
            print(f"  {n / FPS:6.1f}s / {t1:.1f}s", flush=True)
    enc.stdin.close()
    enc.wait()
    print(out)


if __name__ == "__main__":
    main()
