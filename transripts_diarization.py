import argparse
import json
import os
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

# Consecutive same-speaker words separated by more than this many seconds
# of silence start a new utterance instead of being merged into one blob.
MAX_PAUSE_WITHIN_UTTERANCE = 1.5

# A same-speaker run this short (word count AND duration) that is flanked
# on both sides by the SAME other speaker is treated as a false flip and
# reassigned to that flanking speaker instead of starting a new line.
MAX_FLIP_WORDS = 2
MAX_FLIP_DURATION = 1.0

# Whisper sometimes repeats a word/short phrase back-to-back during
# overlapping/noisy audio. Two identical words closer together than this
# (in seconds) are treated as an ASR loop artifact, not a real repetition.
MAX_LOOP_GAP = 0.3


@dataclass
class Word:
    start: float
    end: float
    text: str


# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Decode any audio file to mono float32 @ TARGET_SR.

    Uses faster-whisper's own decode_audio (backed by PyAV, which bundles
    ffmpeg's decoding libraries as compiled wheels). This avoids depending
    on a system ffmpeg install or the removed stdlib `audioop` module --
    the two things that make audio libraries like pydub/soundfile fragile
    to install across OSes and Python versions.
    """
    audio = decode_audio(str(path), sampling_rate=TARGET_SR)
    return audio.astype(np.float32), TARGET_SR


# --------------------------------------------------------------------------
# Transcription
# --------------------------------------------------------------------------

def transcribe(whisper_model: WhisperModel, audio: np.ndarray) -> list[Word]:
    """Transcribe with word-level timestamps.

    Word-level granularity (rather than whole Whisper segments) is what
    lets us split at a speaker boundary that falls *inside* a segment --
    the case that previously caused a whole sentence to get glued onto
    the wrong speaker when the switch happened mid-segment.
    """
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
    """Collapse duplicate word loops Whisper can emit during overlapping speech.

    e.g. "and i and i and i think" -> "and i think". Only collapses an
    immediate repeat of the same word within max_gap seconds, so genuine
    repeated words spoken naturally ("very very good") that aren't rapid
    ASR stutter are left alone (they'll typically have a normal speech
    gap, not a near-zero one).
    """
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
# Diarization
# --------------------------------------------------------------------------

def diarize(pipeline: Pipeline, audio: np.ndarray, sr: int, num_speakers: int = 2):
    result = pipeline(
        {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": sr},
        num_speakers=num_speakers,
    )
    return [(t.start, t.end, spk) for t, spk in result.speaker_diarization]


def dominant_speaker(seg_start: float, seg_end: float, turns) -> str:
    """Speaker whose turn(s) overlap this interval the most."""
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
    """Fix instant, spurious speaker flips (Directive 3).

    Finds runs of consecutive words assigned to the same speaker. If a run
    is short (<= MAX_FLIP_WORDS words and <= MAX_FLIP_DURATION seconds)
    and both its neighboring runs belong to the SAME other speaker, it's
    almost certainly diarization noise at a turn boundary rather than a
    real quick interjection -- so it gets folded into the surrounding
    speaker instead of splitting the line.
    """
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
# Combine
# --------------------------------------------------------------------------

def build_utterances(words: list[Word], speakers: list[str], audio_file: str) -> list[dict]:
    """Group words into utterances, splitting exactly at speaker changes
    (Directive 1) and merging consecutive same-speaker words that don't
    have a long pause between them.
    """
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