"""
CHANGE from the original script: adds refine_speaker_boundaries().

WHY: smooth_speakers() (Directive 3) already fixes short, spurious flips --
a run of <= MAX_FLIP_WORDS words, <= MAX_FLIP_DURATION seconds, flanked by
the SAME other speaker on both sides (A -> [noise] -> A). That's real, but
it's a different failure mode from what a transcript diff against an
official caption source turned up: pyannote can also mistime a GENUINE
speaker transition by several words/seconds -- not a spurious blip, an
actual A -> B handoff that's just cut too early or too late relative to
where the sentence really ends. smooth_speakers() correctly leaves these
alone (the run is too long and isn't flanked by the same speaker on both
sides -- it's not noise, it's a real transition, just mistimed), which
means nothing in the original pipeline corrects it.

Confirmed example (raw diarized output vs. the video's official
transcript): pyannote cut the speaker boundary ~13 words into what was
actually still one continuous question from a single speaker, gluing the
tail of that question onto the next speaker's turn.

FIX: assign_speakers() only ever asks pyannote "which acoustic turn does
this word overlap most" -- it never checks whether that boundary lands
somewhere grammatically sane. refine_speaker_boundaries() adds that check:
at every speaker-change point, look within a short time window (not a
fixed word count, since speaking rate varies) for the nearest sentence-
final punctuation mark, and snap the boundary there instead of trusting
the raw acoustic overlap. This runs AFTER smooth_speakers() (so short
noise-blips are already cleaned up first) and BEFORE build_utterances()
(so the utterances you actually write out are already correct, instead of
relying on a downstream chunking heuristic to guess-repair them from word
choice alone after the fact).

Validated standalone against a reconstruction of the exact real failure
case (see test_boundary_snap_v2.py) before being wired in here.
"""
import argparse
import json
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from pyannote.audio import Pipeline

TARGET_SR = 16000

MAX_PAUSE_WITHIN_UTTERANCE = 1.5
MAX_FLIP_WORDS = 2
MAX_FLIP_DURATION = 1.0
MAX_LOOP_GAP = 0.3

# NEW: how far (in seconds, not words -- speaking rate varies) to search for
# a sentence-final punctuation mark near a raw speaker-change point before
# giving up and leaving the acoustic boundary as-is. Wide enough to cover
# the ~3-4s drift observed in the real failure case; not so wide that it
# risks reaching into an entirely separate, later exchange.
BOUNDARY_SNAP_MAX_SECONDS = 5.0

SENTENCE_END_RE = re.compile(r'[.?!]$')


@dataclass
class Word:
    start: float
    end: float
    text: str


# --------------------------------------------------------------------------
# Audio (unchanged)
# --------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio = decode_audio(str(path), sampling_rate=TARGET_SR)
    return audio.astype(np.float32), TARGET_SR


# --------------------------------------------------------------------------
# Transcription (unchanged)
# --------------------------------------------------------------------------

def transcribe(whisper_model: WhisperModel, audio: np.ndarray) -> list[Word]:
    raw_segments, _ = whisper_model.transcribe(
        audio, beam_size=5, vad_filter=True, word_timestamps=True
    )
    words: list[Word] = []
    for seg in raw_segments:
        if not seg.words:
            continue
        for w in seg.words:
            text = w.word.strip()
            if text:
                words.append(Word(start=w.start, end=w.end, text=text))
    return words


def dedupe_loops(words: list[Word], max_gap: float = MAX_LOOP_GAP) -> list[Word]:
    cleaned: list[Word] = []
    for w in words:
        if cleaned:
            prev = cleaned[-1]
            same_word = prev.text.strip(".,!?").lower() == w.text.strip(".,!?").lower()
            close_in_time = (w.start - prev.end) <= max_gap
            if same_word and close_in_time:
                cleaned[-1] = Word(start=prev.start, end=w.end, text=w.text)
                continue
        cleaned.append(w)
    return cleaned


# --------------------------------------------------------------------------
# Diarization (unchanged)
# --------------------------------------------------------------------------

def diarize(pipeline: Pipeline, audio: np.ndarray, sr: int, num_speakers: int = 2):
    result = pipeline(
        {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr},
        num_speakers=num_speakers,
    )
    return [(t.start, t.end, spk) for t, spk in result.speaker_diarization]


def dominant_speaker(seg_start: float, seg_end: float, turns) -> str:
    overlap_by_speaker: dict[str, float] = {}
    for t_start, t_end, spk in turns:
        ov = min(seg_end, t_end) - max(seg_start, t_start)
        if ov > 0:
            overlap_by_speaker[spk] = overlap_by_speaker.get(spk, 0.0) + ov
    if not overlap_by_speaker:
        return "SPEAKER_00"
    return max(overlap_by_speaker, key=overlap_by_speaker.get)


def assign_speakers(words: list[Word], turns) -> list[str]:
    return [dominant_speaker(w.start, w.end, turns) for w in words]


def smooth_speakers(speakers: list[str], words: list[Word]) -> list[str]:
    """Unchanged -- fixes short, spurious flips (Directive 3). Left as the
    first pass because it targets a genuinely different failure mode
    (acoustic noise) than refine_speaker_boundaries() below (mistimed
    genuine transitions); cleaning up noise first avoids it confusing the
    grammar-based search that follows."""
    n = len(speakers)
    smoothed = speakers.copy()
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            j = i
            while j < n and smoothed[j] == smoothed[i]:
                j += 1
            run_len = j - i
            run_duration = words[j - 1].end - words[i].start
            if (
                i > 0
                and j < n
                and smoothed[i - 1] == smoothed[j]
                and smoothed[i - 1] != smoothed[i]
                and run_len <= MAX_FLIP_WORDS
                and run_duration <= MAX_FLIP_DURATION
            ):
                for k in range(i, j):
                    smoothed[k] = smoothed[i - 1]
                changed = True
            i = j
    return smoothed


# --------------------------------------------------------------------------
# NEW: grammar-aware boundary refinement
# --------------------------------------------------------------------------
def _is_sentence_end_token(text: str) -> bool:
    """A token ending in . ? or ! counts as a sentence boundary -- UNLESS
    it's a bare ellipsis ('...'/'..'), which marks a trailing-off pause
    (still mid-thought), not a completed sentence. Whisper tokenizes
    ellipses both as their own token and fused onto punctuation depending
    on context, so both shapes need handling."""
    if not text:
        return False
    if set(text) <= {"."}:
        return len(text) == 1
    return bool(SENTENCE_END_RE.search(text))


def _find_nearby_boundary(words: list[Word], change_idx: int,
                           max_seconds: float = BOUNDARY_SNAP_MAX_SECONDS) -> int:
    """Time-windowed (not word-count-windowed) search for the nearest
    sentence-final token around a raw speaker-change index. Time-based
    scales correctly across different speaking rates -- a fixed word count
    sized for fast banter would be too narrow for a slower, more
    deliberate speaker, and vice versa. Returns the index unchanged if
    nothing is found nearby, rather than guessing."""
    ref_time = words[change_idx].start

    k = change_idx
    while k > 0 and (ref_time - words[k - 1].start) <= max_seconds:
        if _is_sentence_end_token(words[k - 1].text):
            return k
        k -= 1

    k = change_idx
    while k < len(words) and (words[k].start - ref_time) <= max_seconds:
        if k > 0 and _is_sentence_end_token(words[k - 1].text):
            return k
        k += 1

    return change_idx


def refine_speaker_boundaries(speakers: list[str], words: list[Word],
                               max_seconds: float = BOUNDARY_SNAP_MAX_SECONDS) -> list[str]:
    """Corrects a mistimed GENUINE speaker transition by snapping it to the
    nearest sentence-final punctuation, rather than trusting raw per-word
    acoustic overlap with pyannote's diarization turns. Complements (does
    not replace) smooth_speakers(): that catches short spurious noise
    flips; this catches a real transition that landed on the wrong word
    because pyannote's boundary drifted from the true clause break.
    Deliberately a heuristic, not ground truth -- verify against an
    official transcript/caption source when one exists for a given video."""
    speakers = speakers.copy()
    i = 1
    while i < len(speakers):
        if speakers[i] != speakers[i - 1]:
            snapped = _find_nearby_boundary(words, i, max_seconds)
            if snapped != i:
                fill_speaker = speakers[i - 1] if snapped > i else speakers[i]
                lo, hi = sorted((i, snapped))
                for k in range(lo, hi):
                    speakers[k] = fill_speaker
                i = hi
                continue
        i += 1
    return speakers


# --------------------------------------------------------------------------
# Combine (unchanged)
# --------------------------------------------------------------------------

def build_utterances(words: list[Word], speakers: list[str], audio_file: str) -> list[dict]:
    utterances: list[dict] = []
    for w, spk in zip(words, speakers):
        can_merge = (
            utterances
            and utterances[-1]["speaker"] == spk
            and (w.start - utterances[-1]["end"]) <= MAX_PAUSE_WITHIN_UTTERANCE
        )
        if can_merge:
            utterances[-1]["text"] += " " + w.text
            utterances[-1]["end"] = round(w.end, 2)
        else:
            utterances.append(
                {
                    "speaker": spk,
                    "audio_file": audio_file,
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "text": w.text,
                }
            )
    return utterances


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def process_file(audio_path: Path, whisper_model: WhisperModel, diarize_pipeline: Pipeline) -> list[dict]:
    audio, sr = load_audio(audio_path)

    words = transcribe(whisper_model, audio)
    if not words:
        print(f"  WARNING: no speech detected in {audio_path.name}")
        return []

    words = dedupe_loops(words)

    turns = diarize(diarize_pipeline, audio, sr, num_speakers=2)
    speakers = assign_speakers(words, turns)
    speakers = smooth_speakers(speakers, words)
    speakers = refine_speaker_boundaries(speakers, words)   # NEW

    return build_utterances(words, speakers, audio_path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path, nargs="?", default=Path("data/audio/"), help="Folder of .mp3 files")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("data/transcript_diarized"))
    parser.add_argument("--model", default="turbo", help="Whisper size: tiny/base/small/medium/large-v3/turbo")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("Set the HF_TOKEN environment variable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading models...")
    whisper_model = WhisperModel(args.model, device=device, compute_type="float16" if device == "cuda" else "int8")
    diarize_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    if device == "cuda":
        diarize_pipeline.to(torch.device("cuda"))

    mp3_files = sorted(args.dir.glob("*.mp3"))
    print(f"Found {len(mp3_files)} file(s)")

    for audio_path in mp3_files:
        print(f"Processing {audio_path.name}...")
        try:
            utterances = process_file(audio_path, whisper_model, diarize_pipeline)
        except Exception:
            print(f"  FAILED: {audio_path.name}")
            traceback.print_exc()
            continue

        out_file = args.out_dir / f"{audio_path.stem}.json"
        out_file.write_text(json.dumps(utterances, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  -> {out_file.name} ({len(utterances)} utterances)")


if __name__ == "__main__":
    main()