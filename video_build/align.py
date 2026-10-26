"""Align the script's voiceover text to Whisper word timings -> video_build/words.json

Whisper mishears some words ("unitry", "15 million"), so caption text comes from
video_assets/script.md and only the timing comes from the transcript.
"""
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "video_assets" / "script.md"
TRANSCRIPT = ROOT / "video_build" / "transcript.json"
OUT = ROOT / "video_build" / "words.json"


def script_words():
    """[(section_index, display_word)] from the '> ...' lines under each **VO:**."""
    words, section = [], -1
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section += 1
        elif line.startswith("> "):
            words += [(section, w) for w in line[2:].split()]
    return words


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def main():
    sw = script_words()
    tw = json.loads(TRANSCRIPT.read_text())["words"]
    a, b = [norm(w) for _, w in sw], [norm(w["w"]) for w in tw]
    times = [None] * len(sw)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = (tw[j1 + k]["s"], tw[j1 + k]["e"])
        elif op == "replace":
            # spread the script words over the misheard span, weighted by length
            s, e = tw[j1]["s"], tw[j2 - 1]["e"]
            lens = [max(len(a[i]), 1) for i in range(i1, i2)]
            acc = 0
            for k, n in enumerate(lens):
                t0 = s + (e - s) * acc / sum(lens)
                acc += n
                times[i1 + k] = (t0, s + (e - s) * acc / sum(lens))
    # script-only words ("delete" ops): squeeze them into the gap after the previous word
    for i, t in enumerate(times):
        if t is None:
            prev = times[i - 1][1] if i else 0.0
            nxt = next((x[0] for x in times[i + 1:] if x), prev + 0.3)
            times[i] = (prev, max(prev + 0.05, min(nxt, prev + 0.3)))

    out = [{"w": w, "sec": sec, "s": round(t[0], 3), "e": round(t[1], 3)}
           for (sec, w), t in zip(sw, times)]
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for sec in sorted({o["sec"] for o in out}):
        ws = [o for o in out if o["sec"] == sec]
        print(f"section {sec + 1}: {ws[0]['s']:6.2f} - {ws[-1]['e']:6.2f}  ({len(ws)} words)")


if __name__ == "__main__":
    main()
