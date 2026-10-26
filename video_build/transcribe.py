"""Transcribe the voiceover with word-level timestamps -> video_build/transcript.json"""
import json, subprocess, sys
import numpy as np
from faster_whisper import WhisperModel

audio = sys.argv[1] if len(sys.argv) > 1 else "video_assets/mujocu_audio1.wav"
model = WhisperModel("small.en", device="cpu", compute_type="int8")
# decode via ffmpeg (faster-whisper's PyAV path breaks with newer av)
pcm = subprocess.run(["ffmpeg", "-v", "quiet", "-i", audio, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                     capture_output=True, check=True).stdout
samples = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
segments, info = model.transcribe(samples, word_timestamps=True, beam_size=5)
words = []
for seg in segments:
    print(f"[{seg.start:6.2f}-{seg.end:6.2f}] {seg.text}")
    for w in seg.words:
        words.append({"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)})
json.dump({"duration": info.duration, "words": words}, open("video_build/transcript.json", "w"), indent=1)
