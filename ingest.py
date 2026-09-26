"""
Load chunked JSON into Postgres using the schema:
    chunks(source_file, chunk_index, speakers, start_time, end_time, text, embedding VECTOR(384))

Accepts two shapes of input JSON:
  - Already-chunked: {"text", "start_time", "end_time", "speakers"/"speaker"}
  - Raw diarization segments: {"text", "start", "end", "speaker"} -- typically many
    short per-utterance entries. Use --merge-max-chars to merge consecutive same-
    speaker segments into larger, more retrieval-friendly chunks before embedding.

Usage:
    python ingest.py data/chunks.json          # ingest a single file
    python ingest.py data/                     # ingest every *.json file in a directory
    python ingest.py data/ --pattern "*.chunks.json"   # custom glob pattern
    python ingest.py data/raw/ --merge-max-chars 800   # merge raw diarization segments first
"""
import argparse
import json
import os
import sys
from pathlib import Path

from psycopg2.extras import execute_values

from config import EMBED_MODEL, embed_text, get_connection

UPSERT_SQL = """
    INSERT INTO chunks (source_file, chunk_index, speakers, start_time, end_time, text, embedding)
    VALUES %s
    ON CONFLICT (source_file, chunk_index) DO UPDATE SET
        speakers   = EXCLUDED.speakers,
        start_time = EXCLUDED.start_time,
        end_time   = EXCLUDED.end_time,
        text       = EXCLUDED.text,
        embedding  = EXCLUDED.embedding;
"""


def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_segment(c: dict) -> dict:
    """Map raw diarization keys (start/end/speaker) onto the chunked schema
    (start_time/end_time/speakers) without touching already-correct keys."""
    return {
        "text": c.get("text"),
        "speakers": c.get("speakers") or c.get("speaker"),
        "start_time": c.get("start_time", c.get("start")),
        "end_time": c.get("end_time", c.get("end")),
    }


def merge_segments(segments: list[dict], max_chars: int) -> list[dict]:
    """Merge consecutive same-speaker segments into chunks up to ~max_chars.
    Intended for raw per-utterance diarization output (lots of tiny segments)
    -- merging gives the embedder enough context per chunk to be useful for
    retrieval, instead of embedding fragments like a single word."""
    merged: list[dict] = []
    current = None

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if (
            current is not None
            and current["speakers"] == seg["speakers"]
            and len(current["text"]) + 1 + len(text) <= max_chars
        ):
            current["text"] = f"{current['text']} {text}"
            current["end_time"] = seg["end_time"]
        else:
            if current is not None:
                merged.append(current)
            current = dict(seg)
            current["text"] = text

    if current is not None:
        merged.append(current)
    return merged


def resolve_files(path: str, pattern: str, recursive: bool) -> list[Path]:
    """Turn a file or directory argument into a sorted list of JSON files to ingest."""
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        globber = p.rglob if recursive else p.glob
        files = sorted(globber(pattern))
        if not files:
            print(f"No files matching '{pattern}' found under {p}"
                  f"{' (recursively)' if recursive else ''}.")
        return files
    print(f"Path not found: {p}")
    sys.exit(1)


def ingest_file(path: Path, conn, merge_max_chars: int | None = None) -> int:
    source_file = path.name
    raw = load_chunks(path)
    segments = [normalize_segment(c) for c in raw]

    if merge_max_chars:
        chunks = merge_segments(segments, merge_max_chars)
        print(f"Loaded {len(raw)} raw segments from {source_file}, "
              f"merged into {len(chunks)} chunks (max {merge_max_chars} chars)")
    else:
        chunks = segments
        print(f"Loaded {len(chunks)} chunks from {source_file}")

    rows = []
    for idx, c in enumerate(chunks):
        text = (c.get("text") or "").strip()
        if not text:
            continue
        vector = embed_text(text)
        speakers = c.get("speakers")
        rows.append((
            source_file,
            idx,
            speakers,
            c.get("start_time"),
            c.get("end_time"),
            text,
            vector,
        ))
        print(f"  [{idx + 1}/{len(chunks)}] embedded chunk_index={idx} (dim={len(vector)})")

    if not rows:
        print(f"  No non-empty chunks in {source_file}, skipping.")
        return 0

    with conn.cursor() as cur:
        execute_values(cur, UPSERT_SQL, rows)
    conn.commit()
    print(f"Upserted {len(rows)} chunks into Postgres for source_file='{source_file}'.")
    return len(rows)


DEFAULT_CHUNKED_DATA_PATH = Path("data/transcript_diarized/chunked")


def main():
    parser = argparse.ArgumentParser(description="Ingest chunked transcript JSON into Postgres.")
    parser.add_argument("path", nargs="?", default=DEFAULT_CHUNKED_DATA_PATH,
                         help="Path to a single chunks.json file, or a directory of them "
                              f"(default: {DEFAULT_CHUNKED_DATA_PATH})")
    parser.add_argument("--pattern", default="*.json",
                         help="Glob pattern used when path is a directory (default: *.json)")
    parser.add_argument("--recursive", action="store_true",
                         help="Recurse into subdirectories when path is a directory")
    parser.add_argument("--merge-max-chars", type=int, default=None, metavar="N",
                         help="If set, merge consecutive same-speaker segments up to N "
                              "characters before embedding. Use this for raw diarization "
                              "output (many short per-utterance entries) so each embedded "
                              "chunk has enough context to be useful for retrieval.")
    args = parser.parse_args()

    files = resolve_files(args.path, args.pattern, args.recursive)
    if not files:
        sys.exit(0)

    print(f"Found {len(files)} file(s) to ingest.")
    print(f"Embedding with Ollama model '{EMBED_MODEL}' (384-dim) ...\n")

    conn = get_connection()
    total = 0
    try:
        for path in files:
            total += ingest_file(path, conn, merge_max_chars=args.merge_max_chars)
            print()
    finally:
        conn.close()

    print(f"Done. Ingested {total} chunk(s) across {len(files)} file(s).")


if __name__ == "__main__":
    main()