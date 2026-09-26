"""
RAG pipeline v4 — wires file-relevance selection into the actual pipeline.

V4 CHANGE (see chat diagnosis):

rag_pipeline.py's select_relevant_files() was built specifically to stop a
stray high-scoring chunk from an unrelated video crowding out the correct
file's chunks -- but hybrid_search_v2 only ever imported bm25_search,
vector_search, and CANDIDATE_POOL from rag_pipeline.py. select_relevant_files
lived in a completely separate code path (rag_pipeline.py's own __main__/
hybrid_search()) that never touched the reranking pipeline actually used for
queries. Once the corpus grew past one file, that meant the one piece of
code written to solve exactly this problem wasn't running.

Fixed by calling select_relevant_files() on the fused pool, right after
fuse_normalized() and before reranking. select_relevant_files() expects a
"rrf_score" key (from rag_pipeline.py's own fuse_rrf); fuse_normalized()
here produces "fused_score" instead -- _alias_for_file_selection() bridges
that without needing to change select_relevant_files() itself, since it's
otherwise agnostic to which fusion formula produced the score, as long as
it's monotonic (higher = more relevant).

rerank_pool_size() (from V3) is also re-tuned for this: it used to scale by
"how many distinct files are in the pool" to avoid starving any one file of
reranker attention. Now that file selection runs FIRST and narrows the pool
to at most MAX_FILES files, that per-file scaling would shrink the pool
right when we most want deep same-file coverage (the Feynman-chunk case
from last diagnosis: correct answer 2 chunks away from the top pick, WITHIN
the same, correctly-selected file). So pool sizing now scales by
per-selected-file depth instead of file count.

---------------------------------------------------------------------------
V3 CHANGES (kept from the previous revision)
---------------------------------------------------------------------------
1. expand_with_neighbors() no longer gates on chunk length (see NEIGHBOR_
   SLICE_WORDS below) -- the old 40-word eligibility gate made it a no-op
   on nearly every chunk transcript_chunking.py's merge_qa_pairs()/
   merge_underflow_turns() actually produce.
2. Reranker pool sizing scales with corpus composition instead of a fixed
   top_k*4 constant tuned for a 21-chunk, single-file corpus.

---------------------------------------------------------------------------
V1 -> V2 CHANGELOG (original diagnosis, still accurate)
---------------------------------------------------------------------------
1. Score normalization before fusion (RRF alone flattened everything into
   a ~0.01-0.03 band at a 30-candidate pool).
2. Cross-encoder reranking (BGE-reranker-v2-m3) -- was missing entirely.
3. Metadata (title/guest) fused into the RERANKER INPUT ONLY, never into
   the embedded/BM25 text.
4. Absolute + relative score thresholds -- explicit "insufficient
   evidence" signal instead of forcing top_k results.
5. Sentence-window context expansion for fragment chunks.
6. Position ordering (best-first/second-best-last) to mitigate
   lost-in-the-middle.

Install: pip install sentence-transformers --break-system-packages
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from sentence_transformers import CrossEncoder

from rag_pipeline import bm25_search, vector_search, CANDIDATE_POOL, select_relevant_files
from config import get_connection

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

# Absolute floor: below this sigmoid-scaled relevance score, a chunk is
# considered noise regardless of how it ranked relative to its neighbors.
#
# NOTE ON THIS VALUE: calibrated (~15-20 labeled pairs) on a 21-chunk,
# single-file corpus. On-topic matches scored 0.63-0.75; a merely-adjacent
# topic scored 0.51. The corpus has since grown to 5 files / 110 chunks --
# re-run the calibration pass (via eval_recall.py, extended with per-file
# negative controls) before trusting this value at the new scale.
ABS_SCORE_FLOOR = 0.55
REL_SCORE_FLOOR = 0.50

RRF_K_NORMALIZED = 15   # small k so rank position differentiates documents
                        # instead of flattening everything into a narrow band.

# V5: fusion weights, now tunable instead of hardcoded 0.5/0.5.
#
# Diagnosed case: "Who inspired Bill Gates during Microsoft's early years?"
# -- idx=10 ("...my friend Paul Allen and I in the EARLY YEARS...") beat
# idx=8 (the chunk that actually says "who'd you look up to? ... Richard
# Feynman") because idx=10 shares a near-verbatim phrase with the query.
# BM25 rewards that kind of literal overlap; dense embeddings are supposed
# to be the arm that catches "inspired" ~ "look up to" as semantically
# close even with zero shared words. Shifting weight toward the dense arm
# is a reasonable hypothesis for fixing this class of miss -- but it's a
# hypothesis, not a proven fix: dense retrieval isn't guaranteed to rank
# idx=8 above idx=10 either without actually re-running it. Use
# eval_recall.py to sweep DENSE_WEIGHT (try 0.5, 0.65, 0.8) and keep
# whichever value actually improves recall/MRR on your labeled set --
# don't trust this default number over a measurement.
BM25_WEIGHT = 0.35
DENSE_WEIGHT = 0.65

_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME, max_length=512)
    return _reranker


# --------------------------------------------------------------------------
# Step 1: score-normalized hybrid fusion (replaces bare fuse_rrf)
# --------------------------------------------------------------------------
def _min_max_normalize(pairs: list[tuple[dict, float]], higher_is_better: bool) -> dict[int, float]:
    """Normalize a (row, score) list to [0,1] by id, independent of RRF rank."""
    if not pairs:
        return {}
    scores = [s for _, s in pairs]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1e-9
    out = {}
    for row, s in pairs:
        norm = (s - lo) / span
        out[row["id"]] = norm if higher_is_better else 1.0 - norm
    return out


def fuse_normalized(bm25_results, dense_results, rrf_k: int = RRF_K_NORMALIZED,
                     bm25_weight: float = BM25_WEIGHT, dense_weight: float = DENSE_WEIGHT):
    """
    Two-signal fusion: normalized raw scores (primary signal) blended with
    reciprocal-rank (tie-breaker / smoothing), rather than RRF alone.
    bm25_weight/dense_weight are exposed as arguments (not just module
    constants) specifically so eval_recall.py can sweep them without
    monkeypatching globals.
    """
    bm25_norm = _min_max_normalize(bm25_results, higher_is_better=True)
    dense_pairs = [(row, row["distance"]) for row in dense_results]
    dense_norm = _min_max_normalize(dense_pairs, higher_is_better=False)

    rows_by_id = {}
    combined: dict[int, float] = {}

    for rank, (row, _score) in enumerate(bm25_results, start=1):
        rid = row["id"]
        rows_by_id[rid] = row
        combined[rid] = combined.get(rid, 0.0) + bm25_weight * bm25_norm.get(rid, 0.0) + 1 / (rrf_k + rank)

    for rank, row in enumerate(dense_results, start=1):
        rid = row["id"]
        rows_by_id.setdefault(rid, row)
        combined[rid] = combined.get(rid, 0.0) + dense_weight * dense_norm.get(rid, 0.0) + 1 / (rrf_k + rank)

    fused = [{**rows_by_id[rid], "fused_score": score} for rid, score in combined.items()]
    fused.sort(key=lambda r: r["fused_score"], reverse=True)
    return fused


# --------------------------------------------------------------------------
# Step 1.5: file-relevance selection  [V4: NOW ACTUALLY WIRED IN]
# --------------------------------------------------------------------------
def _alias_for_file_selection(fused: list[dict]) -> list[dict]:
    """select_relevant_files() (rag_pipeline.py) expects a 'rrf_score' key,
    since it was originally written against fuse_rrf()'s output. It's
    otherwise agnostic to how the score was produced, as long as higher is
    better -- so we just alias fuse_normalized()'s 'fused_score' under the
    key it expects, rather than forking the function."""
    return [{**row, "rrf_score": row["fused_score"]} for row in fused]


def apply_file_selection(fused: list[dict], verbose: bool = False) -> list[dict]:
    """Filter the fused pool down to the file(s) that actually dominate it,
    before any reranking happens. This is the step that was implemented in
    rag_pipeline.py but never called from the actual query path -- without
    it, a stray high-scoring chunk from an unrelated video can occupy a
    reranker slot that should have gone to the right file's chunk."""
    selected_files, ranked_files = select_relevant_files(_alias_for_file_selection(fused))
    if verbose:
        print(f"  file scores: {ranked_files}")
        print(f"  selected file(s): {selected_files}")

    filtered = [r for r in fused if r["source_file"] in selected_files]
    # Fallback: selected_files is derived from fused itself, so this
    # shouldn't be able to empty the pool -- but don't silently return
    # zero results if the heuristic ever misbehaves.
    return filtered if filtered else fused


# --------------------------------------------------------------------------
# Step 2: source-file metadata lookup (for reranker-input prefixing only)
# --------------------------------------------------------------------------
def _display_title(source_file: str) -> str:
    """Turn 'Some_Video_Title_chunked.json' into a readable title.
    Swap this out for a real manifest (source_file -> {title, guest_name})
    if you have one -- filename parsing is a reasonable v1."""
    title = source_file.rsplit(".", 1)[0]
    title = title.replace("_chunked", "").replace("_", " ")
    return title


def _guest_name_hint(source_file: str) -> str:
    """Best-effort guest name for reranker context. Replace with a proper
    manifest lookup in production."""
    return _display_title(source_file).split(",")[0].split(" - ")[0]


# --------------------------------------------------------------------------
# Step 3: rerank with metadata-fused input text
# --------------------------------------------------------------------------
def rerank(query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
    """Cross-encoder rerank. Prefixes title/guest context into the text
    seen by the reranker ONLY -- candidates[i]['text'] (used later for the
    LLM prompt) is left untouched."""
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = []
    for row in candidates:
        title = _display_title(row["source_file"])
        guest = _guest_name_hint(row["source_file"])
        prefixed = f"[{title}] {guest}: {row['text']}"
        pairs.append((query, prefixed))

    raw_scores = reranker.predict(pairs)  # cross-encoder logits
    scored = [{**row, "rerank_score": float(s)} for row, s in zip(candidates, raw_scores)]
    scored.sort(key=lambda r: r["rerank_score"], reverse=True)

    import math
    for r in scored:
        r["rerank_score_norm"] = 1 / (1 + math.exp(-r["rerank_score"]))

    return scored[:top_n] if top_n else scored


# --------------------------------------------------------------------------
# Step 4: absolute + relative score filtering
# --------------------------------------------------------------------------
def filter_by_score(reranked: list[dict], abs_floor=ABS_SCORE_FLOOR, rel_floor=REL_SCORE_FLOOR):
    """Drop chunks that are noise regardless of rank position. Returns
    (kept, was_confident)."""
    if not reranked:
        return [], False

    top_score = reranked[0]["rerank_score_norm"]
    kept = [
        r for r in reranked
        if r["rerank_score_norm"] >= abs_floor and r["rerank_score_norm"] >= rel_floor * top_score
    ]
    was_confident = top_score >= abs_floor
    return kept, was_confident


# --------------------------------------------------------------------------
# Step 5: sentence-window context expansion  [V5: sentence-snapped + deduped]
# --------------------------------------------------------------------------
NEIGHBOR_SLICE_WORDS = 15    # base slice size -- safe regardless of chunk length
NEIGHBOR_SNAP_MAX_EXTRA = 15  # how much further to search for a sentence
                               # boundary before giving up and using the raw cut

_SENTENCE_END_RE = re.compile(r'[.?!]$')


def _ends_sentence(word: str) -> bool:
    """A trailing ellipsis ('...') marks a trailing-off pause, not a
    completed sentence -- same distinction transripts_diarization.py's
    boundary-snapping already makes, reused here for the same reason."""
    if set(word) <= {"."}:
        return False
    return bool(_SENTENCE_END_RE.search(word))


def _snap_tail_to_sentence_start(words: list[str], base_n: int, max_extra: int = NEIGHBOR_SNAP_MAX_EXTRA) -> str:
    """Take the last `base_n` words of a previous chunk, then walk further
    back (up to max_extra words) so the slice starts right after a
    sentence boundary instead of mid-sentence. Falls back to the raw
    base_n-word cut if no boundary is found within the search window --
    a slightly-awkward start beats silently growing the slice forever."""
    start = max(0, len(words) - base_n)
    limit = max(0, start - max_extra)
    i = start
    while i > limit:
        if _ends_sentence(words[i - 1]):
            break
        i -= 1
    else:
        i = start  # no boundary found in range -- use the original cut
    return " ".join(words[i:])


def _snap_head_to_sentence_end(words: list[str], base_n: int, max_extra: int = NEIGHBOR_SNAP_MAX_EXTRA) -> str:
    """Take the first `base_n` words of a next chunk, then walk further
    forward (up to max_extra words) so the slice ends on a complete
    sentence instead of mid-sentence. Falls back to the raw base_n-word
    cut if no boundary is found within the search window."""
    end = min(len(words), base_n)
    limit = min(len(words), end + max_extra)
    i = end
    while i < limit:
        if _ends_sentence(words[i - 1]):
            break
        i += 1
    else:
        i = end  # no boundary found in range -- use the original cut
    return " ".join(words[:i])


def expand_with_neighbors(kept: list[dict], conn=None) -> list[dict]:
    """Add a small, sentence-boundary-snapped slice of the previous/next
    chunk's tail/head around EVERY kept result. Two things this version
    fixes over a raw N-word slice:

    1. Sentence boundaries: a raw word-count cut regularly starts or ends
       mid-sentence (e.g. "[...the world. And my mom by kind of pushing
       me,"). _snap_tail_to_sentence_start / _snap_head_to_sentence_end
       extend the cut to the nearest sentence edge instead.
    2. De-duplication: if an adjacent chunk (chunk_index +/- 1) is ALSO
       one of the kept results, it will already appear as its own full
       passage -- so its content is skipped here rather than being
       duplicated inside this row's bracket.

    The row's own text always stays intact and clearly marked as the
    actual match -- neighbor context is wrapped in [...] so it can never
    bury or outweigh the real content."""
    owns_conn = conn is None
    conn = conn or get_connection()

    # (source_file, chunk_index) pairs that will already appear as their
    # own passage -- skip bracketing them into a neighbor's row too.
    kept_keys = {(row["source_file"], row["chunk_index"]) for row in kept}

    try:
        with conn.cursor() as cur:
            for row in kept:
                prev_key = (row["source_file"], row["chunk_index"] - 1)
                next_key = (row["source_file"], row["chunk_index"] + 1)
                need_prev = prev_key not in kept_keys
                need_next = next_key not in kept_keys

                neighbor_rows = {}
                if need_prev or need_next:
                    cur.execute(
                        """
                        SELECT chunk_index, text FROM chunks
                        WHERE source_file = %(sf)s
                          AND chunk_index IN (%(prev)s, %(next)s)
                        ORDER BY chunk_index
                        """,
                        {
                            "sf": row["source_file"],
                            "prev": row["chunk_index"] - 1,
                            "next": row["chunk_index"] + 1,
                        },
                    )
                    neighbor_rows = {idx: text for idx, text in cur.fetchall()}

                prefix = ""
                if need_prev and (row["chunk_index"] - 1) in neighbor_rows:
                    prev_words = neighbor_rows[row["chunk_index"] - 1].split()
                    tail = _snap_tail_to_sentence_start(prev_words, NEIGHBOR_SLICE_WORDS)
                    if tail:
                        prefix = f"[...{tail}] "

                suffix = ""
                if need_next and (row["chunk_index"] + 1) in neighbor_rows:
                    next_words = neighbor_rows[row["chunk_index"] + 1].split()
                    head = _snap_head_to_sentence_end(next_words, NEIGHBOR_SLICE_WORDS)
                    if head:
                        suffix = f" [{head}...]"

                row["expanded_text"] = f"{prefix}{row['text']}{suffix}".strip()
        return kept
    finally:
        if owns_conn:
            conn.close()


# --------------------------------------------------------------------------
# Step 6: position ordering to mitigate lost-in-the-middle
# --------------------------------------------------------------------------
def order_for_prompt(kept: list[dict]) -> list[dict]:
    """Best chunk first, second-best last, remainder tapered into the
    middle."""
    if len(kept) <= 2:
        return kept
    ordered = [None] * len(kept)
    lo, hi = 0, len(kept) - 1
    for i, row in enumerate(kept):
        if i % 2 == 0:
            ordered[lo] = row
            lo += 1
        else:
            ordered[hi] = row
            hi -= 1
    return ordered


# --------------------------------------------------------------------------
# Rerank pool sizing  [V4: re-tuned for post-file-selection depth]
# --------------------------------------------------------------------------
def rerank_pool_size(filtered_fused: list[dict], top_k: int, per_file: int = 20, floor: int = 30) -> int:
    """How many candidates (AFTER file selection) to send to the cross-
    encoder. V3 scaled this by file COUNT to stop cross-file crowding --
    now that apply_file_selection() handles cross-file crowding directly,
    that scaling would shrink the pool right when we want DEPTH within the
    1-2 selected files (the Feynman-chunk case: correct answer 2 chunks
    away from the top pick, within the correctly-selected file). So this
    scales by how many files survived selection * how deep to search
    within each, not by total file count in the raw pool."""
    n_selected_files = len({row["source_file"] for row in filtered_fused}) or 1
    return max(top_k * 4, floor, n_selected_files * per_file)


# --------------------------------------------------------------------------
# End-to-end: hybrid -> fuse -> select files -> rerank -> filter -> expand -> order
# --------------------------------------------------------------------------
def hybrid_search_v2(query: str, top_k: int = 5, candidate_pool: int = CANDIDATE_POOL, verbose: bool = False,
                      bm25_weight: float = BM25_WEIGHT, dense_weight: float = DENSE_WEIGHT):
    bm25_results = bm25_search(query, top_n=candidate_pool)
    dense_results = vector_search(query, top_n=candidate_pool)
    # bm25_weight/dense_weight pass through to fuse_normalized so
    # eval_recall.py can sweep them directly instead of monkeypatching
    # module constants.
    fused = fuse_normalized(bm25_results, dense_results, bm25_weight=bm25_weight, dense_weight=dense_weight)

    # V4: narrow to the dominant file(s) BEFORE reranking -- this is the
    # step that existed in rag_pipeline.py but was never actually called.
    filtered = apply_file_selection(fused, verbose=verbose)

    pool_size = rerank_pool_size(filtered, top_k)
    reranked = rerank(query, filtered[:pool_size])
    kept, was_confident = filter_by_score(reranked)

    if not kept:
        if verbose:
            print("  No chunk cleared the relevance floor.")
        return {"results": [], "confident": False}

    kept = kept[:top_k]
    kept = expand_with_neighbors(kept)
    kept = order_for_prompt(kept)

    if verbose:
        for r in kept:
            print(f"  rerank={r['rerank_score_norm']:.3f} fused={r['fused_score']:.4f} "
                  f"{r['source_file'][:40]:40s} idx={r['chunk_index']}")

    results = [{
        "speaker": row.get("speakers"),
        "source_file": row["source_file"],
        "chunk_index": row["chunk_index"],
        "start": row.get("start_time"),
        "end": row.get("end_time"),
        "rerank_score": row["rerank_score_norm"],
        "text": row.get("expanded_text", row["text"]),
    } for row in kept]

    return {"results": results, "confident": was_confident}


# --------------------------------------------------------------------------
# Step 7: prompt assembly with an explicit "insufficient evidence" branch
# --------------------------------------------------------------------------
def build_context_block(search_output: dict) -> str:
    if not search_output["results"]:
        return ("[No sufficiently relevant passages were found in the transcript "
                "corpus for this query. Say so explicitly rather than guessing.]")

    lines = []
    if not search_output["confident"]:
        lines.append("[NOTE: retrieval confidence was low -- treat the following as "
                      "weak evidence and hedge accordingly in the answer.]\n")

    for i, r in enumerate(search_output["results"], start=1):
        lines.append(
            f"--- Passage {i} (source: {_display_title(r['source_file'])}, "
            f"{r['start']:.1f}s-{r['end']:.1f}s, relevance={r['rerank_score']:.2f}) ---\n"
            f"{r['text']}"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    TEST_QUERIES = [
        "What did Bill Gates say about AI?",
        "What are the pros and cons of AI according to Bill Gates?",
        "What did Bill Gates say about healthcare and education?",
        "Who inspired Bill Gates during Microsoft's early years?",
        "How did MrBeast build his YouTube channel?",
        "What did MrBeast say about Feastables?",
    ]
    for q in TEST_QUERIES:
        print(f"\n=== {q} ===")
        out = hybrid_search_v2(q, verbose=True)
        # V5: print the full context block. The previous [:500] truncation
        # cut passages off mid-sentence and was mistaken for a retrieval
        # bug (the "pros and cons of AI" case) when the full answer was
        # actually present -- just not printed.
        print(build_context_block(out))