"""
Speaker-Aware Transcript Chunking Engine
=========================================

Chunks a raw ASR/diarization transcript (list of per-utterance dicts with
`speaker`, `start`, `end`, `text`) into speaker-coherent chunks, applying:

  1. Speaker Grouping   - consecutive same-speaker utterances become one turn.
  2. Underflow Merging  - turns under MIN_WORDS are held and merged with the
                           *next* turn from the same speaker (even if other
                           speakers talk in between), so short interjections
                           ("yeah", "right", "oh really") don't become their
                           own noisy chunk.
  3. Overflow Splitting - turns over MAX_WORDS are split on sentence
                           boundaries into sub-chunks, each carrying a
                           15-20% trailing-word overlap from the previous
                           sub-chunk for context continuity. All sub-chunks
                           keep the parent turn's speaker metadata and get a
                           lettered chunk_id suffix (e.g. chunk_004a).

Input schema (per element):
    {"speaker": str, "start": float, "end": float, "text": str, ...extra ignored}

Output schema (per element):
    {
      "chunk_id": "chunk_001",
      "speaker": "Speaker_01",
      "start_time": 12.4,
      "end_time": 45.2,
      "text": "...",
      "is_overflow_split": false
    }
"""

import json
import re
import sys
from copy import deepcopy
from pathlib import Path

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MIN_WORDS = 30          # underflow threshold
MAX_WORDS = 400         # overflow threshold
OVERLAP_MIN_PCT = 0.15  # trailing overlap lower bound
OVERLAP_MAX_PCT = 0.20  # trailing overlap upper bound
OVERLAP_PCT = (OVERLAP_MIN_PCT + OVERLAP_MAX_PCT) / 2  # 17.5% default

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.?!])\s+')


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------
# Step 1: Speaker Grouping
# --------------------------------------------------------------------------
def group_consecutive_speakers(utterances):
    """Merge consecutive utterances from the same speaker into single turns."""
    turns = []
    for u in utterances:
        speaker = u["speaker"]
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] = turns[-1]["text"].rstrip() + " " + u["text"].strip()
            turns[-1]["end"] = u["end"]
        else:
            turns.append({
                "speaker": speaker,
                "start": u["start"],
                "end": u["end"],
                "text": u["text"].strip(),
            })
    return turns


# --------------------------------------------------------------------------
# Step 2: Underflow Merging
# --------------------------------------------------------------------------
def merge_underflow_turns(turns, min_words=MIN_WORDS):
    """
    Walk through turns in order. If a turn is under `min_words`, hold it in a
    per-speaker pending buffer instead of emitting it immediately, and fold
    it into that speaker's next turn (whenever it comes, even after other
    speakers talk). A turn is only finalized/emitted once it reaches the
    threshold, or at the very end of the transcript (best effort flush).
    """
    pending = {}     # speaker -> accumulating turn dict
    finalized = []   # turns ready for the overflow stage, in the order
                      # they were completed

    for turn in turns:
        speaker = turn["speaker"]
        if speaker in pending:
            buf = pending[speaker]
            buf["text"] = buf["text"].rstrip() + " " + turn["text"].strip()
            buf["end"] = turn["end"]
        else:
            buf = deepcopy(turn)
            pending[speaker] = buf

        if word_count(buf["text"]) >= min_words:
            finalized.append(buf)
            del pending[speaker]

    # Flush any short turns left over at the end of the transcript (there's
    # nothing left to merge them with, so emit as-is).
    for speaker, buf in pending.items():
        finalized.append(buf)

    return finalized


# --------------------------------------------------------------------------
# Step 3: Overflow Splitting + Overlap
# --------------------------------------------------------------------------
def split_sentences(text):
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s for s in sentences if s]


def pack_sentences_into_segments(sentences, max_words):
    """Greedily pack whole sentences into segments <= max_words words."""
    segments, current, current_len = [], [], 0
    for sent in sentences:
        n = word_count(sent)
        if current and current_len + n > max_words:
            segments.append(" ".join(current))
            current, current_len = [], 0
        current.append(sent)
        current_len += n
    if current:
        segments.append(" ".join(current))
    return segments


def pack_words_into_segments(words, max_words):
    """Fallback when no sentence punctuation exists: hard-split by word count."""
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def apply_trailing_overlap(segments, pct=OVERLAP_PCT):
    """
    For each segment after the first, prepend a trailing slice (pct of the
    *previous* segment's word count, clamped to 15-20%) of that previous
    segment's words, so consecutive sub-chunks share context.
    """
    if len(segments) <= 1:
        return segments

    overlapped = [segments[0]]
    for i in range(1, len(segments)):
        prev_words = segments[i - 1].split()
        overlap_n = max(1, round(len(prev_words) * pct))
        overlap_n = min(overlap_n, len(prev_words))
        overlap_text = " ".join(prev_words[-overlap_n:])
        overlapped.append(overlap_text + " " + segments[i])
    return overlapped


def split_overflowing_turn(turn, max_words=MAX_WORDS):
    """
    Split a single speaker turn into sentence-bounded sub-chunks (with
    trailing overlap) if it exceeds `max_words`. Timestamps for each
    sub-chunk are linearly interpolated across the turn's [start, end] span
    based on cumulative word position, since we only have turn-level
    timing, not per-word timing.
    """
    total_words = word_count(turn["text"])
    if total_words <= max_words:
        return [{**turn, "is_overflow_split": False}]

    sentences = split_sentences(turn["text"])
    if len(sentences) > 1:
        base_segments = pack_sentences_into_segments(sentences, max_words)
    else:
        # No sentence punctuation available (common in raw ASR output) ->
        # fall back to a straight word-count split so we still respect the
        # max-length constraint.
        base_segments = pack_words_into_segments(turn["text"].split(), max_words)

    segments = apply_trailing_overlap(base_segments)

    # Interpolate start/end times proportionally to each *original*
    # (pre-overlap) segment's share of total word count.
    duration = turn["end"] - turn["start"]
    cumulative_words = 0
    sub_chunks = []
    for base_seg, full_seg in zip(base_segments, segments):
        seg_words = word_count(base_seg)
        seg_start = turn["start"] + duration * (cumulative_words / total_words)
        cumulative_words += seg_words
        seg_end = turn["start"] + duration * (cumulative_words / total_words)
        sub_chunks.append({
            "speaker": turn["speaker"],
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "text": full_seg.strip(),
            "is_overflow_split": True,
        })
    return sub_chunks


# --------------------------------------------------------------------------
# Step 4: Assemble final schema
# --------------------------------------------------------------------------
def assign_chunk_ids(chunk_groups):
    """
    chunk_groups: list of lists -- each inner list is the 1+ sub-chunks
    produced from one finalized turn (1 item if no split, 2+ if overflow).
    """
    output = []
    for i, group in enumerate(chunk_groups, start=1):
        base_id = f"chunk_{i:03d}"
        if len(group) == 1:
            c = group[0]
            output.append({
                "chunk_id": base_id,
                "speaker": c["speaker"],
                "start_time": c["start"],
                "end_time": c["end"],
                "text": c["text"],
                "is_overflow_split": c["is_overflow_split"],
            })
        else:
            for j, c in enumerate(group):
                suffix = chr(ord("a") + j)
                output.append({
                    "chunk_id": f"{base_id}{suffix}",
                    "speaker": c["speaker"],
                    "start_time": c["start"],
                    "end_time": c["end"],
                    "text": c["text"],
                    "is_overflow_split": c["is_overflow_split"],
                })
    return output


def chunk_transcript(utterances, min_words=MIN_WORDS, max_words=MAX_WORDS):
    turns = group_consecutive_speakers(utterances)
    turns = merge_underflow_turns(turns, min_words=min_words)
    chunk_groups = [split_overflowing_turn(t, max_words=max_words) for t in turns]
    return assign_chunk_ids(chunk_groups)


# --------------------------------------------------------------------------
# Fixed-path file selection
# --------------------------------------------------------------------------
# All transcript JSON files are read from INPUT_DIR and each one's chunked
# result is written to OUTPUT_DIR under the same stem with a "_chunked"
# suffix, e.g. INPUT_DIR/interview.json -> OUTPUT_DIR/interview_chunked.json
INPUT_DIR = Path("data/transcript_diarized")
OUTPUT_DIR = Path("data/transcript_diarized/chunked")
INPUT_GLOB = "*.json"   # pattern used to select files from INPUT_DIR


def select_input_files(input_dir=INPUT_DIR, pattern=INPUT_GLOB):
    """Return every transcript file matching `pattern` in the fixed input dir."""
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    return sorted(input_dir.glob(pattern))


def process_file(in_file: Path, output_dir=OUTPUT_DIR):
    with open(in_file, "r", encoding="utf-8") as f:
        utterances = json.load(f)

    chunks = chunk_transcript(utterances)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{in_file.stem}_chunked.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    return out_file, chunks


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Optional overrides: `python chunking_engine.py [input_dir] [output_dir]`
    # With no args, the fixed INPUT_DIR / OUTPUT_DIR paths above are used and
    # every *.json file found in INPUT_DIR is processed.
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    files = select_input_files(input_dir)
    if not files:
        print(f"No files matching '{INPUT_GLOB}' found in {input_dir}")
        sys.exit(0)

    for in_file in files:
        out_file, chunks = process_file(in_file, output_dir)
        print(f"\n{in_file.name} -> {out_file} ({len(chunks)} chunks)")
        for c in chunks:
            print(f"  {c['chunk_id']:>10} | {c['speaker']:<12} | "
                  f"{c['start_time']:>7.2f}-{c['end_time']:<7.2f} | "
                  f"{word_count(c['text']):>3} words | split={c['is_overflow_split']}")