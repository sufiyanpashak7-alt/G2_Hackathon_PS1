"""
eval_recall.py — automated retrieval-quality tests: recall@k, MRR, nDCG@k
against a multi-transcript labeled query set.

Run standalone for a printed report + an HTML report written to disk:
    python eval_recall.py
    python eval_recall.py --top-k 5 --html report.html

Run as a CI regression gate:
    pytest eval_recall.py -v

Requires the DB to already contain the transcript chunks (run ingest.py first).
"""
from __future__ import annotations

import argparse
import html as html_lib
import math
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from config import get_connection
from rag import hybrid_search_v2


@dataclass
class LabeledQuery:
    query: str
    relevant_anchors: list[str] = field(default_factory=list)
    source_file: str | None = None
    note: str = ""
    # Used only for report grouping. Existing labels default to "core".
    category: str = "core"


def _chunked_name(raw_source_file: str) -> str:
    stem = raw_source_file.rsplit(".json", 1)[0]
    if stem.endswith("_chunked"):
        return f"{stem}.json"
    return f"{stem}_chunked.json"


def _normalize_filename(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = "".join(" " if ch.isspace() else ch for ch in s)
    s = " ".join(s.split())
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return s.lower()


_source_file_index: dict[str, str] | None = None


def _load_source_file_index() -> dict[str, str]:
    global _source_file_index
    if _source_file_index is None:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT source_file FROM chunks")
                _source_file_index = {
                    _normalize_filename(row[0]): row[0] for row in cur.fetchall()
                }
        finally:
            conn.close()
    return _source_file_index


def resolve_source_file(raw_source_file: str) -> str:
    index = _load_source_file_index()
    target = _normalize_filename(_chunked_name(raw_source_file))
    if target in index:
        return index[target]
    raise ValueError(
        f"source_file label {raw_source_file!r} doesn't match any file in the "
        f"corpus after normalization (looked for: {target!r}).\n"
        f"Files actually in Postgres: {sorted(index.values())}"
    )


def resolve_relevant_chunk_indices(lq: LabeledQuery) -> set[int]:
    if not lq.relevant_anchors:
        return set()

    resolved_source_file = resolve_source_file(lq.source_file) if lq.source_file else None

    conn = get_connection()
    indices: set[int] = set()
    try:
        with conn.cursor() as cur:
            for anchor in lq.relevant_anchors:
                sql = "SELECT chunk_index FROM chunks WHERE text ILIKE %s"
                params: list = [f"%{anchor}%"]
                if resolved_source_file:
                    sql += " AND source_file = %s"
                    params.append(resolved_source_file)

                cur.execute(sql, params)
                rows = cur.fetchall()

                if not rows:
                    raise ValueError(
                        f"[{lq.query!r}] anchor not found in current corpus: {anchor!r}"
                        + (
                            f" within {resolved_source_file!r}"
                            if resolved_source_file
                            else ""
                        )
                        + ". Transcript or chunking changed since this label was "
                          "written -- update the anchor to match the current text."
                    )

                if len(rows) > 1:
                    raise ValueError(
                        f"[{lq.query!r}] anchor matched {len(rows)} chunks: {anchor!r}. "
                        "Re-chunking made this substring ambiguous -- narrow the anchor."
                    )

                indices.add(rows[0][0])
    finally:
        conn.close()

    return indices


# ---------------------------------------------------------------------------
# Labeled query set
# ---------------------------------------------------------------------------
LABELED_QUERIES: list[LabeledQuery] = [
    # -----------------------------------------------------------------------
    # VERIFIED LABELS FROM THE ORIGINAL TEST FILE
    # -----------------------------------------------------------------------
    # Keep these anchors unchanged unless they stop resolving in the current
    # corpus. Report-only test cases should not invent anchors that have not
    # been verified against the actual transcript chunks.
    LabeledQuery(
        query="What is the two-part plan for today's episode?",
        relevant_anchors=["two -part plan for today"],
        source_file="10-Minute English Conversation & Shadowing Practice (B1–B2 Podcast).json",
        note="Part 1 shadowing/listening, Part 2 vocabulary breakdown",
        category="core",
    ),
    LabeledQuery(
        query="What changes did Maya make to her morning routine to stop scrolling on her phone?",
        relevant_anchors=["leave my phone charging in the kitchen overnight"],
        source_file="10-Minute English Conversation & Shadowing Practice (B1–B2 Podcast).json",
        note="Phone in kitchen overnight, glass of water, fresh air, reading 10 pages",
        category="core",
    ),
    LabeledQuery(
        query="What is the 70th, 50th, and 25th milestone Bill Gates mentions?",
        relevant_anchors=["Microsoft turns 50"],
        source_file="Bill Gates Joked with Steve Jobs About Taking the Wrong LSD, Talks AI and Optimism for the Future.json",
        note="Gates turning 70, Microsoft turning 50, Foundation turning 25",
        category="core",
    ),
    LabeledQuery(
        query="What joke did Bill Gates make regarding Steve Jobs' comment about him taking LSD?",
        relevant_anchors=["he should have taken acid"],
        source_file="Bill Gates Joked with Steve Jobs About Taking the Wrong LSD, Talks AI and Optimism for the Future.json",
        note="Joked that he took acid but got the batch about code rather than design",
        category="core",
    ),
    LabeledQuery(
        query="Why did MrBeast have to move his production shoot away from a lake when filming a video?",
        relevant_anchors=["federal crime"],
        source_file="MrBeast Counted to 100,000 in His First Viral Video, Leaves Another Message for Himself in 10 Years.json",
        note="Bald eagle's nest near the lake; federal crime to disturb",
        category="core",
    ),
    LabeledQuery(
        query="What message did teenage MrBeast leave for himself in the 10-year video from 2015?",
        relevant_anchors=["please, future me, please"],
        source_file="MrBeast Counted to 100,000 in His First Viral Video, Leaves Another Message for Himself in 10 Years.json",
        note="Comparing 2015 numbers (8k subs, 1.8M views) to future self, hoped for 1M subs",
        category="core",
    ),
    LabeledQuery(
        query="What shift does Satya Nadella describe regarding application architecture and SaaS applications?",
        relevant_anchors=["crud database will then get orchestrated"],
        source_file="Satya Nadella on the Future of SaaS, How 2025 is the year of Agents, Advice for Indian Engineers.json",
        note="Shift to agents orchestrating tools across SaaS apps, decoupling CRUD/logic",
        category="core",
    ),
    LabeledQuery(
        query="What two simultaneous gears should software developers work in according to Satya Nadella?",
        relevant_anchors=["two gears you have to simultaneously"],
        source_file="Satya Nadella on the Future of SaaS, How 2025 is the year of Agents, Advice for Indian Engineers.json",
        note="Frontier gear (experimentation) and optimization gear (cost, latency, deployment)",
        category="core",
    ),
    LabeledQuery(
        query="What advice does Sundar Pichai give to Indian engineers focusing on competitive exam mindsets and rote learning?",
        relevant_anchors=["Three Idiots"],
        source_file="Sundar Pichai’s advice for Indian Engineers, AI and India, Wrapper Startups, and More!.json",
        note="Focus on deeper fundamental understanding rather than rote learning specific tools",
        category="core",
    ),
    LabeledQuery(
        query="What makes a startup a 'shallow wrapper' according to Sundar Pichai, and how can startups add value?",
        relevant_anchors=["shouldn't be a shallow wrapper"],
        source_file="Sundar Pichai’s advice for Indian Engineers, AI and India, Wrapper Startups, and More!.json",
        note="Shallow wrappers get replaced by native model updates; add value by solving real workflows",
        category="core",
    ),

    # -----------------------------------------------------------------------
    # NEGATIVE CONTROLS
    # -----------------------------------------------------------------------
    LabeledQuery(
        query="What is the step-by-step recipe for baking a sourdough bread loaf at high altitude?",
        relevant_anchors=[],
        note="NEGATIVE CONTROL — topic absent from corpus; pipeline should abstain",
        category="negative_control",
    ),
    LabeledQuery(
        query="How does the James Webb Space Telescope calculate orbital corrections near Lagrange Point 2?",
        relevant_anchors=[],
        note="NEGATIVE CONTROL — topic absent from corpus; pipeline should abstain",
        category="negative_control",
    ),
]

# ---------------------------------------------------------------------------
# Result metadata helpers
# ---------------------------------------------------------------------------
def _load_chunk_metadata(source_file: str | None, chunk_index: int | None) -> dict:
    """Load canonical file/timestamp/speaker/text metadata from Postgres."""
    if not source_file or chunk_index is None:
        return {}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_file, chunk_index, speakers, start_time, end_time, text
                FROM chunks
                WHERE source_file = %s AND chunk_index = %s
                """,
                (source_file, chunk_index),
            )
            row = cur.fetchone()
            if not row:
                return {}
            return {
                "source_file": row[0],
                "chunk_index": row[1],
                "speakers": row[2],
                "start_time": row[3],
                "end_time": row[4],
                "text": row[5],
            }
    finally:
        conn.close()


def _hydrate_result_metadata(results: list[dict]) -> list[dict]:
    """Ensure search results expose file, timestamp and speaker metadata."""
    hydrated = []
    for result in results:
        item = dict(result)
        source_file = item.get("source_file")
        chunk_index = item.get("chunk_index")

        missing = any(
            item.get(key) is None
            for key in ("speakers", "start_time", "end_time", "text")
        )
        if missing:
            item.update({
                key: value
                for key, value in _load_chunk_metadata(source_file, chunk_index).items()
                if item.get(key) is None
            })

        if item.get("speakers") is None and item.get("speaker") is not None:
            item["speakers"] = item["speaker"]

        hydrated.append(item)
    return hydrated


def _format_timestamp(start_time, end_time) -> str:
    if start_time is None and end_time is None:
        return "timestamp unavailable"
    if end_time is None:
        return f"{float(start_time):.2f}s"
    if start_time is None:
        return f"to {float(end_time):.2f}s"
    return f"{float(start_time):.2f}s – {float(end_time):.2f}s"


def _format_speakers(result: dict) -> str:
    speakers = result.get("speakers")
    if speakers is None:
        speakers = result.get("speaker")
    if speakers is None or not str(speakers).strip():
        return "speaker unavailable"
    return str(speakers).strip()


def _format_result_line(result: dict) -> str:
    """Console representation matching the required retrieval output."""
    file_label = _short_file_name(
        result.get("source_file"), result.get("chunk_index")
    )
    timestamp = _format_timestamp(result.get("start_time"), result.get("end_time"))
    speaker = _format_speakers(result)
    return f"{file_label} | {timestamp} | {speaker}"


def _format_result_html(result: dict) -> str:
    """HTML result card: file + timestamp + speaker + text snippet."""
    file_label = _short_file_name(
        result.get("source_file"), result.get("chunk_index")
    )
    timestamp = _format_timestamp(result.get("start_time"), result.get("end_time"))
    speaker = _format_speakers(result)
    text = str(result.get("text") or "")
    snippet = text[:220] + ("…" if len(text) > 220 else "")

    return (
        '<div class="result-item">'
        f'<div class="result-file">{html_lib.escape(file_label)}</div>'
        f'<div class="result-meta">'
        f'<span class="result-time">{html_lib.escape(timestamp)}</span>'
        f'<span class="result-speaker">{html_lib.escape(speaker)}</span>'
        f'</div>'
        f'<div class="result-text">{html_lib.escape(snippet)}</div>'
        '</div>'
    )

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def mrr(retrieved: list[int], relevant: set[int]) -> float:
    for rank, idx in enumerate(retrieved, start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, idx in enumerate(retrieved[:k], start=1)
        if idx in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)
    )
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_eval(top_k: int = 5) -> list[dict]:
    report = []

    for lq in LABELED_QUERIES:
        relevant_indices = resolve_relevant_chunk_indices(lq)

        out = hybrid_search_v2(lq.query, top_k=top_k)
        retrieved = _hydrate_result_metadata(out["results"])

        if lq.source_file:
            expected_file = resolve_source_file(lq.source_file)
            retrieved = [
                r for r in retrieved if r.get("source_file") == expected_file
            ]

        retrieved_indices = [r["chunk_index"] for r in retrieved]

        relevant_results = []
        if relevant_indices:
            expected_file = resolve_source_file(lq.source_file) if lq.source_file else None
            for idx in sorted(relevant_indices):
                meta = _load_chunk_metadata(expected_file, idx) if expected_file else {}
                if not meta:
                    meta = next(
                        (dict(r) for r in retrieved if r.get("chunk_index") == idx),
                        {},
                    )
                if meta:
                    relevant_results.append(meta)

        if not relevant_indices:
            passed = (
                len(retrieved) == 0
                or not out.get("confident", True)
            )
            report.append({
                "query": lq.query,
                "type": "negative_control",
                "category": "negative_control",
                "passed": passed,
                "retrieved": retrieved_indices,
                "retrieved_results": retrieved,
                "relevant_results": [],
                "note": lq.note,
            })
            continue

        report.append({
            "query": lq.query,
            "type": "positive",
            "category": lq.category,
            "recall@k": recall_at_k(retrieved_indices, relevant_indices, top_k),
            "mrr": mrr(retrieved_indices, relevant_indices),
            "ndcg@k": ndcg_at_k(retrieved_indices, relevant_indices, top_k),
            "retrieved": retrieved_indices,
            "retrieved_results": retrieved,
            "relevant": sorted(relevant_indices),
            "relevant_results": relevant_results,
            "note": lq.note,
        })

    return report


def print_report(report: list[dict], top_k: int) -> None:
    print(f"\n=== Retrieval eval @k={top_k} ({len(report)} labeled queries) ===\n")

    for r in report:
        if r["type"] == "negative_control":
            status = "PASS" if r["passed"] else "FAIL"
            print(f"[{status}] (negative control) {r['query']!r}")
            for result in r.get("retrieved_results", []):
                print(f"    -> {_format_result_line(result)}")
            if not r.get("retrieved_results"):
                print("    -> no confident retrieval")
        else:
            print(
                f"       recall={r['recall@k']:.2f}  "
                f"mrr={r['mrr']:.2f}  "
                f"ndcg={r['ndcg@k']:.2f}  "
                f"{r['query']!r}"
            )
            print("       Retrieved:")
            for result in r.get("retrieved_results", [])[:top_k]:
                print(f"         - {_format_result_line(result)}")
            print("       Expected:")
            for result in r.get("relevant_results", []):
                print(f"         - {_format_result_line(result)}")

    positives = [r for r in report if r["type"] == "positive"]
    negatives = [r for r in report if r["type"] == "negative_control"]

    if positives:
        n = len(positives)
        print(
            f"\nMean recall@{top_k}: "
            f"{sum(r['recall@k'] for r in positives) / n:.3f}  "
            f"Mean MRR: {sum(r['mrr'] for r in positives) / n:.3f}  "
            f"Mean nDCG@{top_k}: "
            f"{sum(r['ndcg@k'] for r in positives) / n:.3f}"
        )

    if negatives:
        rate = sum(r["passed"] for r in negatives) / len(negatives)
        print(
            f"Negative-control pass rate (correctly abstained): "
            f"{rate:.0%}"
        )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def _grade_class(value: float) -> str:
    if value >= 0.75:
        return "good"
    if value >= 0.40:
        return "mid"
    return "bad"


def _metric_summary(rows: list[dict], key: str) -> float:
    return (
        sum(r[key] for r in rows) / len(rows)
        if rows else 0.0
    )


def _short_file_name(source_file: str | None, chunk_index: int | None = None) -> str:
    if not source_file:
        label = "unknown"
    else:
        stem = source_file.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        stem = stem.replace("_chunked.json", "").replace(".json", "")
        label = stem.replace(" ", "_")
    if len(label) > 24:
        label = label[:24] + "…"
    return f"{label}#{chunk_index}" if chunk_index is not None else label


def _format_results(results: list[dict], max_items: int = 5) -> str:
    items = []
    for r in results[:max_items]:
        items.append(
            _short_file_name(r.get("source_file"), r.get("chunk_index"))
        )
    return ", ".join(items)


def generate_html_report(
    report: list[dict],
    top_k: int,
    generated_at: datetime | None = None,
) -> str:
    """Generate a self-contained HTML report styled like the supplied reference."""
    generated_at = generated_at or datetime.now()

    positives = [r for r in report if r["type"] == "positive"]
    negatives = [r for r in report if r["type"] == "negative_control"]

    mean_recall = _metric_summary(positives, "recall@k")
    mean_mrr = _metric_summary(positives, "mrr")
    mean_ndcg = _metric_summary(positives, "ndcg@k")
    neg_pass_rate = (
        sum(r["passed"] for r in negatives) / len(negatives)
        if negatives else None
    )

    def esc(value) -> str:
        return html_lib.escape(str(value))

    groups = [
        ("core", "Core queries", "The original core query set."),
        ("broad", "Broad coverage queries",
         "Additional questions covering different content to check that retrieval quality is not a fluke."),
        ("stress", "Stress test (reworded) queries",
         "The same answer anchors tested with different query wording."),
        ("cross_file", "Cross-file queries",
         "Questions whose answer spans more than one source file."),
    ]

    def group_rows(category: str) -> list[dict]:
        return [r for r in positives if r.get("category") == category]

    def render_positive_rows(rows: list[dict]) -> str:
        rendered = []
        for r in rows:
            row_class = _grade_class(r["recall@k"])
            retrieved_html = "".join(
                _format_result_html(result)
                for result in r.get("retrieved_results", [])[:top_k]
            ) or '<span class="muted-inline">No retrieved results</span>'
            expected_html = "".join(
                _format_result_html(result)
                for result in r.get("relevant_results", [])
            ) or '<span class="muted-inline">No expected result metadata</span>'

            rendered.append(
                f"""
        <tr class="{row_class}">
          <td class="qcell">{esc(r['query'])}
            <div class="note">{esc(r['note'])}</div>
          </td>
          <td class="num">{r['recall@k']:.2f}</td>
          <td class="num">{r['mrr']:.2f}</td>
          <td class="num">{r['ndcg@k']:.2f}</td>
          <td class="results-cell">{retrieved_html}</td>
          <td class="results-cell">{expected_html}</td>
        </tr>"""
            )
        return "".join(rendered)

    sections = []
    for category, title, explanation in groups:
        rows = group_rows(category)
        if not rows:
            continue

        avg_recall = _metric_summary(rows, "recall@k")
        sections.append(
            f"""
  <h2>{esc(title)} ({len(rows)})
    <span class="subhead">avg recall {avg_recall:.2f}</span>
  </h2>
  <p class="explain">{esc(explanation)}</p>
  <table>
    <thead>
      <tr>
        <th>Query</th>
        <th>Recall@{top_k}</th>
        <th>MRR</th>
        <th>nDCG@{top_k}</th>
        <th>Retrieved — file / timestamp / speaker</th>
        <th>Expected — file / timestamp / speaker</th>
      </tr>
    </thead>
    <tbody>{render_positive_rows(rows)}</tbody>
  </table>
"""
        )

    negative_rows = []
    for r in negatives:
        badge = (
            '<span class="badge pass">PASS</span>'
            if r["passed"]
            else '<span class="badge fail">FAIL</span>'
        )

        cited_html = "".join(
            _format_result_html(result)
            for result in r.get("retrieved_results", [])[:top_k]
        ) or '<span class="muted-inline">None</span>'

        negative_rows.append(
            f"""
        <tr class="{'good' if r['passed'] else 'bad'}">
          <td class="qcell">{esc(r['query'])}
            <div class="note">{esc(r['note'])}</div>
          </td>
          <td>{badge}</td>
          <td class="results-cell">{cited_html}</td>
        </tr>"""
        )

    neg_summary = (
        f"{neg_pass_rate:.0%}"
        if neg_pass_rate is not None
        else "—"
    )

    positive_count = len(positives)
    negative_count = len(negatives)

    # Avoid claiming a fixed corpus size: derive it from the results/labels.
    corpus_files = set()
    for r in report:
        for result in r.get("retrieved_results", []):
            if result.get("source_file"):
                corpus_files.add(result["source_file"])
    for lq in LABELED_QUERIES:
        if lq.source_file:
            try:
                corpus_files.add(resolve_source_file(lq.source_file))
            except Exception:
                pass

    # Keep the verdict descriptive rather than silently changing test semantics.
    recall_target = 0.70
    recall_verdict = (
        f"clears the ≥{recall_target:.2f} recall target"
        if mean_recall >= recall_target
        else f"is below the ≥{recall_target:.2f} recall target"
    )

    if neg_pass_rate == 1.0:
        neg_verdict = "all out-of-corpus queries were correctly abstained"
    elif neg_pass_rate is None:
        neg_verdict = "no negative controls were included"
    else:
        neg_verdict = (
            f"{neg_pass_rate:.0%} of out-of-corpus queries were correctly abstained"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Retrieval Eval Report — top-{top_k}</title>
<style>
  :root {{
    --bg:#0f1115; --panel:#171a21; --border:#2a2e38;
    --text:#e6e8ec; --muted:#9aa1ad;
    --good:#1f7a4d; --good-bg:#12251c;
    --mid:#9c7a12; --mid-bg:#241f10;
    --bad:#b3392f; --bad-bg:#2a1512;
    --accent:#5b8def;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:32px 24px 64px;
    background:var(--bg); color:var(--text);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    line-height:1.5;
  }}
  h1 {{ font-size:1.5rem; margin:0 0 4px; }}
  .meta {{ color:var(--muted); font-size:.85rem; margin-bottom:8px; }}
  .verdict {{
    background:var(--panel); border:1px solid var(--border);
    border-left:4px solid var(--accent); border-radius:8px;
    padding:14px 18px; margin:18px 0 28px; font-size:.95rem;
  }}
  .cards {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }}
  .card {{
    background:var(--panel); border:1px solid var(--border);
    border-radius:10px; padding:16px 20px; min-width:170px;
  }}
  .card .label {{
    font-size:.72rem; color:var(--muted); text-transform:uppercase;
    letter-spacing:.04em;
  }}
  .card .value {{ font-size:1.6rem; font-weight:600; margin-top:4px; }}
  .card .sub {{ font-size:.72rem; color:var(--muted); margin-top:2px; }}
  h2 {{ font-size:1.05rem; margin:34px 0 4px; }}
  .subhead {{ font-weight:400; color:var(--muted); font-size:.82rem; }}
  .explain {{ color:var(--muted); font-size:.85rem; margin:4px 0 12px; max-width:820px; }}
  table {{
    width:100%; border-collapse:collapse; background:var(--panel);
    border-radius:10px; overflow:hidden;
  }}
  th,td {{
    padding:10px 14px; text-align:left; border-bottom:1px solid var(--border);
    font-size:.86rem; vertical-align:top;
  }}
  th {{
    color:var(--muted); font-weight:600; font-size:.75rem;
    text-transform:uppercase; letter-spacing:.03em;
  }}
  td.num {{ font-variant-numeric:tabular-nums; width:65px; }}
  td.mono {{
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
    font-size:.78rem; color:var(--muted); max-width:260px;
  }}
  .results-cell {{ min-width:280px; max-width:430px; }}
  .result-item {{ padding:0 0 9px; margin:0 0 9px; border-bottom:1px solid var(--border); }}
  .result-item:last-child {{ border-bottom:0; padding-bottom:0; margin-bottom:0; }}
  .result-file {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.78rem; color:var(--text); word-break:break-word; }}
  .result-meta {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:2px; font-size:.75rem; }}
  .result-time {{ color:var(--accent); font-variant-numeric:tabular-nums; }}
  .result-speaker {{ color:var(--muted); font-weight:600; }}
  .result-text {{ margin-top:3px; color:var(--muted); font-size:.74rem; line-height:1.35; }}
  .muted-inline {{ color:var(--muted); font-size:.78rem; }}
  .qcell {{ max-width:380px; }}
  .note {{ color:var(--muted); font-size:.76rem; margin-top:3px; }}
  tr.good td {{ background:var(--good-bg); }}
  tr.mid td {{ background:var(--mid-bg); }}
  tr.bad td {{ background:var(--bad-bg); }}
  .badge {{
    display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:.72rem; font-weight:600;
  }}
  .badge.pass {{
    background:var(--good-bg); color:#4ade80; border:1px solid var(--good);
  }}
  .badge.fail {{
    background:var(--bad-bg); color:#f87171; border:1px solid var(--bad);
  }}
  .legend {{
    display:flex; gap:18px; flex-wrap:wrap; margin:10px 0 24px;
    font-size:.78rem; color:var(--muted);
  }}
  .legend span.dot {{
    display:inline-block; width:10px; height:10px; border-radius:50%;
    margin-right:6px; vertical-align:middle;
  }}
  footer {{
    margin-top:44px; color:var(--muted); font-size:.76rem;
    border-top:1px solid var(--border); padding-top:16px;
  }}
</style>
</head>
<body>
  <h1>Retrieval Eval Report</h1>
  <div class="meta">
    Generated {esc(generated_at.strftime('%Y-%m-%d %H:%M:%S'))} ·
    top-{top_k} · {len(report)} labeled queries
    ({positive_count} positive / {negative_count} negative control) ·
    corpus: {len(corpus_files)} source file(s)
  </div>

  <div class="verdict">
    <b>Read this first:</b> mean Recall@{top_k} is
    <b>{mean_recall:.3f}</b>, which {recall_verdict}.
    Negative-control abstention is <b>{neg_summary}</b> —
    {neg_verdict}. See the per-query tables below for the detailed results.
  </div>

  <div class="cards">
    <div class="card">
      <div class="label">Mean Recall@{top_k}</div>
      <div class="value">{mean_recall:.3f}</div>
      <div class="sub">fraction of relevant chunks found in top {top_k}</div>
    </div>
    <div class="card">
      <div class="label">Mean MRR</div>
      <div class="value">{mean_mrr:.3f}</div>
      <div class="sub">how high the first hit ranks</div>
    </div>
    <div class="card">
      <div class="label">Mean nDCG@{top_k}</div>
      <div class="value">{mean_ndcg:.3f}</div>
      <div class="sub">relevance weighted by rank position</div>
    </div>
    <div class="card">
      <div class="label">Negative-control pass rate</div>
      <div class="value">{neg_summary}</div>
      <div class="sub">correctly abstained on out-of-corpus queries</div>
    </div>
  </div>

  <div class="legend">
    <span><span class="dot" style="background:var(--good)"></span>Recall ≥ 0.75 (good)</span>
    <span><span class="dot" style="background:var(--mid)"></span>Recall 0.40–0.74 (partial)</span>
    <span><span class="dot" style="background:var(--bad)"></span>Recall &lt; 0.40 (miss)</span>
  </div>

  {''.join(sections)}

  <h2>Negative controls ({negative_count})</h2>
  <p class="explain">
    Questions with no answer anywhere in the corpus. A correct system returns
    nothing or flags low confidence rather than confidently citing an unrelated chunk.
  </p>
  <table>
    <thead>
      <tr>
        <th>Query</th>
        <th>Result</th>
        <th>Retrieved — file / timestamp / speaker</th>
      </tr>
    </thead>
    <tbody>
      {''.join(negative_rows) if negative_rows else '<tr><td colspan="3">No negative controls.</td></tr>'}
    </tbody>
  </table>

  <footer>
    Ground truth is resolved live from short text anchors against the current
    corpus — never a hardcoded chunk index — so result/index values reflect
    this run's chunking rather than a stale label.
  </footer>
</body>
</html>"""


def write_html_report(
    report: list[dict],
    top_k: int,
    path: str = "eval_report.html",
) -> str:
    content = generate_html_report(report, top_k)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ---------------------------------------------------------------------------
# Pytest integration
# ---------------------------------------------------------------------------
def test_recall_at_5_meets_floor():
    MIN_RECALL = 0.70
    positives = [
        r for r in run_eval(top_k=5) if r["type"] == "positive"
    ]
    avg_recall = sum(r["recall@k"] for r in positives) / len(positives)
    assert avg_recall >= MIN_RECALL, (
        f"Mean recall@5 {avg_recall:.2f} fell below floor {MIN_RECALL}"
    )


def test_negative_controls_abstain():
    negatives = [
        r for r in run_eval(top_k=5) if r["type"] == "negative_control"
    ]
    assert all(r["passed"] for r in negatives), (
        "Pipeline returned confident results for queries outside the corpus: "
        f"{negatives}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run retrieval eval and optionally write an HTML report."
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--html",
        default="eval_report.html",
        help="Path to write the HTML report (default: eval_report.html)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip writing the HTML report",
    )
    args = parser.parse_args()

    rep = run_eval(top_k=args.top_k)
    print_report(rep, top_k=args.top_k)

    if not args.no_html:
        out_path = write_html_report(
            rep,
            top_k=args.top_k,
            path=args.html,
        )
        print(f"\nHTML report written to: {out_path}")
