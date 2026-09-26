"""
RAG pipeline over the `chunks` table (config.py / ingest.py / hybrid_rag.py).

v2 CHANGES (see chat diagnosis -- made after rag.py's V4 wired
select_relevant_files() into the actual query path for the first time):

1. CANDIDATE_POOL: 30 -> 60. File selection (select_relevant_files) can
   only filter what's already in the fused pool -- it can't rescue a
   relevant chunk that never made it into the top-CANDIDATE_POOL BM25 or
   dense results to begin with. 30 was sized before file selection was
   actually wired into the query path; now that it is, this constant caps
   what file selection has to work with, not just what fusion sees.

2. Punctuation-aware tokenization. The old `.lower().split()` treated
   "AI," and "AI" as different tokens, silently losing lexical matches on
   real transcript text (which is full of commas, ellipses, filler
   punctuation from ASR output). _tokenize() now strips non-alphanumeric
   characters, applied identically at index time and query time.

3. Cheap staleness auto-detection for the in-memory BM25 index. Previously
   load_bm25() trusted its cache forever unless force_reload=True was
   passed explicitly -- an easy silent-staleness footgun in a long-lived
   process. Now it also checks the current row count in Postgres against
   what it indexed and auto-rebuilds on a mismatch. This catches the
   common case (chunks ingested since load) for free; it will NOT catch
   an in-place text update to an existing row (same id, same count) --
   force_reload=True is still the right call after any re-ingest that
   might update existing rows, not just add new ones.

4. select_relevant_files() takes an optional score_key, defaulting to
   "rrf_score" (unchanged default, so this file's own hybrid_search()
   below needs no changes). This lets other fusion methods (e.g. rag.py's
   fuse_normalized(), which produces "fused_score") call it directly
   without needing to alias/copy the dict first.

Architecture, mirroring the attached SQLAlchemy script's concept but built on
this project's existing Postgres schema (source_file, chunk_index, speakers,
start_time, end_time, text, embedding VECTOR(384)) and config.py's
`all-minilm` embeddings:

    1. BM25 (Python, rank_bm25)   -- true BM25 scoring, not Postgres ts_rank_cd.
    2. Dense vector search        -- pgvector cosine distance, HNSW-accelerated,
                                      run straight in Postgres.
    3. RRF fusion                 -- the two independently-ranked candidate
                                      lists are fused by reciprocal rank, same
                                      formula as hybrid_rag.py's SQL version.
    4. File relevance selection   -- when the corpus holds multiple unrelated
                                      source files (different videos/
                                      interviews), a query about one of them
                                      can still surface a stray high-scoring
                                      chunk from another. This step aggregates
                                      fused scores per source_file and keeps
                                      only the file(s) that dominate the
                                      candidate pool before truncating to
                                      top_k.

BM25 requires the whole corpus in memory (no inverted index, just like the
attached script) -- load_bm25() now self-checks staleness on every call, but
still call it with force_reload=True after any re-ingest that updates
existing rows in place, since row-count staleness detection can't see that.

Install: pip install rank_bm25 --break-system-packages
"""
import re
from collections import defaultdict

from rank_bm25 import BM25Okapi

from config import RRF_K, embed_text, get_connection

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
CANDIDATE_POOL = 60            # v2: was 30 -- see module docstring change #1
FILE_RELEVANCE_AGG = "sum"     # "sum" (reward consistently-relevant files) or
                                # "max" (reward files with one strong match)
FILE_RELEVANCE_RATIO = 0.5     # a file qualifies if its aggregate score >=
                                # ratio * the top file's aggregate score
MAX_FILES = 2                  # hard cap on qualifying files, even if close

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, alphanumeric-only tokenization. Applied identically at
    index time and query time so 'AI,' and 'AI' collide into the same
    token instead of being scored as unrelated words."""
    return _TOKEN_RE.findall(text.lower())


# --------------------------------------------------------------------------
# BM25 index (in-memory, rebuilt on demand)
# --------------------------------------------------------------------------
_bm25_index = None
_bm25_rows = None
_bm25_row_count = None


def _fetch_all_rows():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, source_file, chunk_index, speakers, start_time, end_time, text
                FROM chunks
                ORDER BY id
            """)
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _current_row_count() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunks")
            return cur.fetchone()[0]
    finally:
        conn.close()


def load_bm25(force_reload: bool = False) -> None:
    """Build (or rebuild) the in-memory BM25 index over every chunk currently
    in Postgres.

    Auto-detects the common staleness case (chunks ingested since the
    index was built) via a cheap row-count check, and rebuilds
    automatically -- no need to remember force_reload=True after a normal
    ingest.py run in the same process. It CANNOT detect an in-place update
    to existing rows (same id, same total count) -- pass force_reload=True
    explicitly after any re-ingest that might have changed existing chunk
    text rather than only adding new ones.
    """
    global _bm25_index, _bm25_rows, _bm25_row_count

    if _bm25_index is not None and not force_reload:
        current_count = _current_row_count()
        if current_count == _bm25_row_count:
            return
        print(f"BM25 index stale (indexed {_bm25_row_count} rows, "
              f"Postgres now has {current_count}) -- rebuilding.")

    rows = _fetch_all_rows()
    if not rows:
        raise RuntimeError("No chunks found in Postgres. Run ingest.py first.")

    tokenized = [_tokenize(r["text"]) for r in rows]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_rows = rows
    _bm25_row_count = len(rows)
    print(f"BM25 index built over {len(rows)} chunks "
          f"across {len({r['source_file'] for r in rows})} file(s).")


# --------------------------------------------------------------------------
# Retrieval arms
# --------------------------------------------------------------------------
def bm25_search(query: str, top_n: int = CANDIDATE_POOL):
    """Returns [(row, bm25_score), ...] sorted descending, zero-score docs
    dropped so they don't pollute the fused ranking with noise."""
    load_bm25()
    scores = _bm25_index.get_scores(_tokenize(query))
    ranked = sorted(zip(_bm25_rows, scores), key=lambda x: x[1], reverse=True)
    return [(row, score) for row, score in ranked[:top_n] if score > 0]


def vector_search(query: str, top_n: int = CANDIDATE_POOL):
    """Dense nearest neighbors straight from Postgres (HNSW-accelerated).
    Returns rows sorted by ascending cosine distance (closest first)."""
    query_embedding = embed_text(query)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, source_file, chunk_index, speakers, start_time, end_time, text,
                       embedding <=> %(qvec)s::vector AS distance
                FROM chunks
                ORDER BY embedding <=> %(qvec)s::vector
                LIMIT %(top_n)s
                """,
                {"qvec": query_embedding, "top_n": top_n},
            )
            cols = [d.name for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


# --------------------------------------------------------------------------
# RRF fusion
# --------------------------------------------------------------------------
def fuse_rrf(bm25_results, dense_results, k: int = RRF_K):
    """Fuse two independently-ranked candidate lists by reciprocal rank,
    keyed on chunk id. Same formula as hybrid_rag.py's SQL version."""
    scores = defaultdict(float)
    rows_by_id = {}

    for rank, (row, _score) in enumerate(bm25_results, start=1):
        scores[row["id"]] += 1 / (k + rank)
        rows_by_id[row["id"]] = row

    for rank, row in enumerate(dense_results, start=1):
        scores[row["id"]] += 1 / (k + rank)
        rows_by_id.setdefault(row["id"], row)

    fused = [{**rows_by_id[cid], "rrf_score": score} for cid, score in scores.items()]
    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused


# --------------------------------------------------------------------------
# File relevance selection
# --------------------------------------------------------------------------
def select_relevant_files(
    fused,
    agg: str = FILE_RELEVANCE_AGG,
    ratio: float = FILE_RELEVANCE_RATIO,
    max_files: int = MAX_FILES,
    score_key: str = "rrf_score",
):
    """Aggregate fused scores per source_file and keep only the file(s)
    that dominate the candidate pool. Returns (selected_files, ranked_files)
    where ranked_files is [(source_file, aggregate_score), ...] descending,
    for transparency/debugging.

    agg="sum" rewards files with several consistently-relevant chunks.
    agg="max" rewards files with one very strong match, even if it's the
    only chunk from that file in the pool. Tune both `agg` and `ratio`
    against your own corpus -- this is a heuristic, not a guarantee, and a
    genuinely cross-file query (e.g. "compare X's and Y's views on Z") will
    correctly keep multiple files if their scores are close.

    score_key: which field on each row holds the fused relevance score.
    Defaults to "rrf_score" (this module's own fuse_rrf() output) so
    existing callers are unaffected, but any monotonic (higher-is-better)
    fusion score works -- e.g. pass score_key="fused_score" to use this
    directly against rag.py's fuse_normalized() output with no aliasing.
    """
    file_scores = defaultdict(float)
    for row in fused:
        score = row[score_key]
        if agg == "max":
            file_scores[row["source_file"]] = max(file_scores[row["source_file"]], score)
        else:
            file_scores[row["source_file"]] += score

    ranked_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked_files:
        return set(), ranked_files

    top_score = ranked_files[0][1]
    selected = {f for f, score in ranked_files[:max_files] if score >= ratio * top_score}
    return selected, ranked_files


# --------------------------------------------------------------------------
# End-to-end hybrid search
# --------------------------------------------------------------------------
def hybrid_search(query: str, top_k: int = 5, candidate_pool: int = CANDIDATE_POOL, verbose: bool = False):
    """Full pipeline: BM25 -> dense -> RRF -> file-relevance filter -> top_k."""
    bm25_results = bm25_search(query, top_n=candidate_pool)
    dense_results = vector_search(query, top_n=candidate_pool)
    fused = fuse_rrf(bm25_results, dense_results)

    selected_files, ranked_files = select_relevant_files(fused)
    filtered = [r for r in fused if r["source_file"] in selected_files]
    if not filtered:
        filtered = fused

    if verbose:
        print(f"  file scores: {ranked_files}")
        print(f"  selected file(s): {selected_files}")

    results = []
    for row in filtered[:top_k]:
        results.append({
            "speaker": row["speakers"],
            "source_file": row["source_file"],
            "chunk_index": row["chunk_index"],
            "start": row["start_time"],
            "end": row["end_time"],
            "score": row["rrf_score"],
            "text": row["text"],
        })
    return results


# --------------------------------------------------------------------------
# Main: run test queries, write results, show file-relevance decisions
# --------------------------------------------------------------------------
if __name__ == "__main__":
    load_bm25()

    TEST_QUERIES = [
        "What did Bill Gates say about AI?",
        "What are the pros and cons of AI according to Bill Gates?",
        "What keeps Bill Gates optimistic about the future?",
        "Who inspired Bill Gates during Microsoft's early years?",
        "What did Bill Gates say about healthcare and education?",
        "What are Bill Gates' concerns regarding AI?",
        "What did MrBeast say about Feastables?",
        "How did MrBeast build his YouTube channel?",
        "What advice did MrBeast give to aspiring creators?",
    ]

    output_file = "hybrid_search_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, query in enumerate(TEST_QUERIES, start=1):
            print(f"[{idx}/{len(TEST_QUERIES)}] {query}")

            bm25_results = bm25_search(query)
            dense_results = vector_search(query)
            fused = fuse_rrf(bm25_results, dense_results)
            selected_files, ranked_files = select_relevant_files(fused)
            print(f"  file scores: {ranked_files}")
            print(f"  selected file(s): {selected_files}")

            filtered = [r for r in fused if r["source_file"] in selected_files] or fused
            results = filtered[:5]

            f.write("=" * 100 + "\n")
            f.write(f"QUERY: {query}\n")
            f.write(f"FILE SCORES: {ranked_files}\n")
            f.write(f"SELECTED FILE(S): {selected_files}\n")
            f.write("=" * 100 + "\n\n")

            for rank, row in enumerate(results, start=1):
                f.write(f"Result #{rank}\n")
                f.write(f"Speaker : {row['speakers']}\n")
                f.write(f"Source  : {row['source_file']}\n")
                f.write(f"Time    : {row['start_time']} - {row['end_time']}\n")
                f.write(f"Score   : {row['rrf_score']:.5f}\n\n")
                f.write(row["text"])
                f.write("\n\n")

            f.write("\n")

    print(f"\nResults saved to: {output_file}")