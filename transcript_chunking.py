"""
Speaker-Aware Transcript Chunking Engine v2
============================================

Builds on transcript_chunking.py's speaker-grouping / underflow-merge /
overflow-split pipeline, and fixes three chunking-level problems that were
causing bad retrieval (see chat analysis):

CHANGE 1 — absorb_inline_interjections()
-----------------------------------------
Problem: group_consecutive_speakers() only merges *consecutive* same-speaker
turns. A short reactive interjection from the OTHER speaker
("Oh, that's the best. Both" / "United" / "Really?") splits what is really
one continuous thought into three retrieval units: [A, tiny-B, A-continues].
Each of those becomes its own row in the `chunks` table with near-zero
topical signal, and they were observed polluting top-5 results across
*every* test query regardless of topic.
Fix: detect the A -> short-B -> A pattern and fold B's words in as a
bracketed aside, remerging the two A-turns into one coherent chunk. This
directly undoes the "split by who's talking" artifact instead of leaving it
for retrieval/reranking to route around.

CHANGE 2 — merge_qa_pairs()
-----------------------------------------
Problem: in interview transcripts, the topical/entity keywords a query is
likely to use ("what keeps you optimistic about the future?") often live in
the INTERVIEWER'S QUESTION, while the substantive answer content lives in a
separate chunk from the other speaker. Splitting them means a
keyword-anchored query can score well against a nearly-content-free question
chunk while missing the answer, or vice versa.
Fix: merge a question-ending turn with the immediately following
different-speaker turn into one chunk (bounded by QA_MAX_MERGED_WORDS so we
don't create unsplittable giants -- overflow splitting still applies to the
result).

CHANGE 3 — apply_baseline_overlap()
-----------------------------------------
Problem: the original apply_trailing_overlap() only runs inside
split_overflowing_turn(), i.e. only for turns over MAX_WORDS (400). Natural
interview turns almost never hit that length, so in practice nearly every
chunk got ZERO neighboring context baked in, which is why fragments like
"...You became, like, before you, who" show up with no continuation.
Fix: give every chunk a short trailing snippet of the previous chunk's text
as a baseline, independent of whether overflow splitting ever triggers.

Output schema is backward compatible with ingest.py's normalize_segment():
{chunk_id, speaker, start_time, end_time, text, is_overflow_split}, plus
new optional flags (absorbed_interjection, is_qa_pair, overlap_prefix_words)
that ingest.py will simply ignore.
"""

import json
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MIN_WORDS = 30
MAX_WORDS = 400
OVERLAP_MIN_PCT = 0.15
OVERLAP_MAX_PCT = 0.20
OVERLAP_PCT = (OVERLAP_MIN_PCT + OVERLAP_MAX_PCT) / 2

# NEW: threshold below which a turn is assumed to carry no independent
# topical content worth its own retrieval unit (pure backchannel / reaction).
BACKCHANNEL_WORD_THRESHOLD = 10

# NEW: cap on merged question+answer length -- above this we leave the pair
# unmerged rather than build an unwieldy chunk (overflow splitting takes over).
QA_MAX_MERGED_WORDS = 300

# NEW: baseline overlap applied to every chunk, not just overflow splits.
BASELINE_OVERLAP_PCT = 0.12
BASELINE_OVERLAP_MAX_WORDS = 25

# DISABLED BY DEFAULT (see chunk_transcript below): baking overlap into the
# stored `text` at ingestion time, AND expanding with neighbor text again at
# retrieval time (rag_pipeline_v2.expand_with_neighbors), stacks -- a chunk's
# neighbor may itself already carry baked-in overlap from ITS neighbor, so
# retrieval-time expansion ends up prepending increasingly large, increasingly
# off-topic blocks of text ahead of the actually-relevant chunk. Overlap now
# lives in exactly one place: expand_with_neighbors, applied as short word-
# level slices at query time, not baked into the indexed/embedded text.
APPLY_BASELINE_OVERLAP = False

SENTENCE_SPLIT_RE = re.compile(r'(?<=[.?!])\s+')


def word_count(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------
# Step 1: Speaker Grouping (unchanged from v1)
# --------------------------------------------------------------------------
def group_consecutive_speakers(utterances):
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
# CHANGE 1: absorb short cross-speaker interjections (A -> tiny B -> A)
# --------------------------------------------------------------------------
def absorb_inline_interjections(turns, word_threshold=BACKCHANNEL_WORD_THRESHOLD):
    """Fold a short reactive interjection from the OTHER speaker back into
    the surrounding same-speaker turns it artificially split, instead of
    leaving it as its own near-empty retrieval unit. Runs to a fixed point
    so cascading A-b-A-b-A patterns fully collapse."""
    turns = [dict(t) for t in turns]
    changed = True
    guard = 0
    while changed and guard < len(turns) + 5:
        changed = False
        guard += 1
        new_turns = []
        i, n = 0, len(turns)
        while i < n:
            if (
                i + 2 < n
                and turns[i]["speaker"] == turns[i + 2]["speaker"]
                and turns[i + 1]["speaker"] != turns[i]["speaker"]
                and word_count(turns[i + 1]["text"]) < word_threshold
            ):
                merged = dict(turns[i])
                aside_speaker = turns[i + 1]["speaker"]
                aside_text = turns[i + 1]["text"].strip()
                merged["text"] = (
                    f"{turns[i]['text'].rstrip()} "
                    f'[{aside_speaker}: "{aside_text}"] '
                    f"{turns[i + 2]['text'].strip()}"
                )
                merged["end"] = turns[i + 2]["end"]
                merged["absorbed_interjection"] = True
                new_turns.append(merged)
                i += 3
                changed = True
            else:
                new_turns.append(turns[i])
                i += 1
        turns = new_turns
    return turns


# --------------------------------------------------------------------------
# Step 2: Underflow Merging (unchanged from v1 -- same-speaker only,
# adjacent-only; see original docstring for why it never reaches across an
# intervening different-speaker turn)
# --------------------------------------------------------------------------
def merge_underflow_turns(turns, min_words=MIN_WORDS):
    merged = []
    i = 0
    while i < len(turns):
        current = dict(turns[i])
        while (
            word_count(current["text"]) < min_words
            and i + 1 < len(turns)
            and turns[i + 1]["speaker"] == current["speaker"]
        ):
            i += 1
            current["text"] = current["text"].rstrip() + " " + turns[i]["text"].strip()
            current["end"] = turns[i]["end"]
        merged.append(current)
        i += 1
    return merged


# --------------------------------------------------------------------------
# CHANGE 2: merge question-ending turn with the following answer turn
# --------------------------------------------------------------------------
def _ends_with_question(text: str) -> bool:
    return text.rstrip().endswith("?")


def merge_qa_pairs(turns, max_words=QA_MAX_MERGED_WORDS):
    """Keep a question and its answer in the same retrieval unit, so
    topical/entity keywords that live in the question aren't separated from
    the substantive content that lives in the answer."""
    result = []
    i, n = 0, len(turns)
    while i < n:
        current = turns[i]
        if (
            i + 1 < n
            and _ends_with_question(current["text"])
            and turns[i + 1]["speaker"] != current["speaker"]
            and word_count(current["text"]) + word_count(turns[i + 1]["text"]) <= max_words
        ):
            merged = dict(current)
            merged["text"] = f"{current['text'].rstrip()} {turns[i + 1]['text'].strip()}"
            merged["end"] = turns[i + 1]["end"]
            merged["speaker"] = f"{current['speaker']}+{turns[i + 1]['speaker']}"
            merged["is_qa_pair"] = True
            result.append(merged)
            i += 2
        else:
            result.append(dict(current))
            i += 1
    return result


# --------------------------------------------------------------------------
# CHANGE 3: baseline overlap for every chunk, not just overflow splits
# --------------------------------------------------------------------------
def apply_baseline_overlap(turns, pct=BASELINE_OVERLAP_PCT, max_words=BASELINE_OVERLAP_MAX_WORDS):
    if len(turns) <= 1:
        return turns
    out = [dict(turns[0])]
    out[0]["overlap_prefix_words"] = 0
    for i in range(1, len(turns)):
        prev_words = turns[i - 1]["text"].split()
        n_overlap = min(max_words, max(1, round(len(prev_words) * pct)))
        prefix = " ".join(prev_words[-n_overlap:])
        current = dict(turns[i])
        current["text"] = f"...{prefix} {current['text']}"
        current["overlap_prefix_words"] = n_overlap
        out.append(current)
    return out


# --------------------------------------------------------------------------
# Step 3: Overflow Splitting + Overlap (unchanged from v1)
# --------------------------------------------------------------------------
def split_sentences(text):
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [s for s in sentences if s]


def pack_sentences_into_segments(sentences, max_words):
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
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def apply_trailing_overlap(segments, pct=OVERLAP_PCT):
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
    total_words = word_count(turn["text"])
    if total_words <= max_words:
        return [{**turn, "is_overflow_split": False}]

    sentences = split_sentences(turn["text"])
    if len(sentences) > 1:
        base_segments = pack_sentences_into_segments(sentences, max_words)
    else:
        base_segments = pack_words_into_segments(turn["text"].split(), max_words)

    segments = apply_trailing_overlap(base_segments)

    duration = turn["end"] - turn["start"]
    cumulative_words = 0
    sub_chunks = []
    for base_seg, full_seg in zip(base_segments, segments):
        seg_words = word_count(base_seg)
        seg_start = turn["start"] + duration * (cumulative_words / total_words)
        cumulative_words += seg_words
        seg_end = turn["start"] + duration * (cumulative_words / total_words)
        sub_chunks.append({
            **{k: v for k, v in turn.items() if k not in ("start", "end", "text")},
            "speaker": turn["speaker"],
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "text": full_seg.strip(),
            "is_overflow_split": True,
        })
    return sub_chunks


# --------------------------------------------------------------------------
# Step 4: Assemble final schema (extended to carry the new optional flags)
# --------------------------------------------------------------------------
def assign_chunk_ids(chunk_groups):
    output = []
    for i, group in enumerate(chunk_groups, start=1):
        base_id = f"chunk_{i:03d}"
        for j, c in enumerate(group):
            chunk_id = base_id if len(group) == 1 else f"{base_id}{chr(ord('a') + j)}"
            row = {
                "chunk_id": chunk_id,
                "speaker": c["speaker"],
                "start_time": c["start"],
                "end_time": c["end"],
                "text": c["text"],
                "is_overflow_split": c["is_overflow_split"],
            }
            # Optional flags -- only included when true/present, so ingest.py's
            # normalize_segment() (which only reads known keys) is unaffected.
            if c.get("absorbed_interjection"):
                row["absorbed_interjection"] = True
            if c.get("is_qa_pair"):
                row["is_qa_pair"] = True
            if c.get("overlap_prefix_words"):
                row["overlap_prefix_words"] = c["overlap_prefix_words"]
            output.append(row)
    return output


def chunk_transcript(utterances, min_words=MIN_WORDS, max_words=MAX_WORDS,
                      apply_overlap=APPLY_BASELINE_OVERLAP):
    turns = group_consecutive_speakers(utterances)
    turns = absorb_inline_interjections(turns)          # CHANGE 1
    turns = merge_underflow_turns(turns, min_words=min_words)
    turns = merge_qa_pairs(turns)                        # CHANGE 2
    if apply_overlap:
        # Off by default -- see APPLY_BASELINE_OVERLAP comment above. Only
        # turn this on if you are NOT also doing neighbor expansion at
        # retrieval time, or you will get the double-stacking bug.
        turns = apply_baseline_overlap(turns)            # CHANGE 3 (optional)
    chunk_groups = [split_overflowing_turn(t, max_words=max_words) for t in turns]
    return assign_chunk_ids(chunk_groups)


# --------------------------------------------------------------------------
# Fixed-path file selection (unchanged from v1)
# --------------------------------------------------------------------------
INPUT_DIR = Path("data/transcript_diarized")
OUTPUT_DIR = Path("data/transcript_diarized/chunked")
INPUT_GLOB = "*.json"


def select_input_files(input_dir=INPUT_DIR, pattern=INPUT_GLOB):
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


if __name__ == "__main__":
    input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else INPUT_DIR
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTPUT_DIR

    files = select_input_files(input_dir)
    if not files:
        print(f"No files matching '{INPUT_GLOB}' found in {input_dir}")
        sys.exit(0)

    for in_file in files:
        out_file, chunks = process_file(in_file, output_dir)
        qa_count = sum(1 for c in chunks if c.get("is_qa_pair"))
        absorbed_count = sum(1 for c in chunks if c.get("absorbed_interjection"))
        print(f"\n{in_file.name} -> {out_file} ({len(chunks)} chunks, "
              f"{qa_count} QA-merged, {absorbed_count} interjection-absorbed)")
        for c in chunks:
            print(f"  {c['chunk_id']:>10} | {c['speaker']:<20} | "
                  f"{c['start_time']:>7.2f}-{c['end_time']:<7.2f} | "
                  f"{word_count(c['text']):>3} words | split={c['is_overflow_split']}"
                  f"{' | QA' if c.get('is_qa_pair') else ''}"
                  f"{' | absorbed' if c.get('absorbed_interjection') else ''}")