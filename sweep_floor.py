"""
sweep_floor.py -- grid-search ABS_SCORE_FLOOR / REL_SCORE_FLOOR against the
(now-fixed) eval_recall.py harness, and report which combination maximizes
recall@k WITHOUT breaking negative-control abstention.

Why this exists: rag.py's own docstring already says ABS_SCORE_FLOOR=0.58
was calibrated on a 21-chunk, single-file corpus and needs re-calibrating
now that the corpus is 110 chunks / 5 files -- this script does that
re-calibration by measurement instead of by guessing a new number.

NOTE on the monkeypatch: filter_by_score(reranked, abs_floor=ABS_SCORE_FLOOR,
rel_floor=REL_SCORE_FLOOR) binds its defaults at *function definition time*,
so reassigning rag.ABS_SCORE_FLOOR after import does NOT change what a
no-arg call to filter_by_score() uses (a classic Python late-binding trap).
This script mutates filter_by_score.__defaults__ directly instead, which
DOES take effect on subsequent no-arg calls -- exactly what
hybrid_search_v2() makes internally.

Usage:
    python sweep_floor.py
    python sweep_floor.py --top-k 5
"""
import argparse
import itertools

import rag
from eval_recall import run_eval

# Grid to search. Narrow this once you see roughly where the good region is.
ABS_CANDIDATES = [0.35, 0.40, 0.45, 0.50, 0.55, 0.58, 0.62]
REL_CANDIDATES = [0.35, 0.40, 0.45, 0.50, 0.55]


def _set_floors(abs_floor: float, rel_floor: float) -> None:
    """See module docstring re: late-binding defaults -- this is the part
    that actually takes effect, not a plain module-attribute assignment."""
    rag.ABS_SCORE_FLOOR = abs_floor
    rag.REL_SCORE_FLOOR = rel_floor
    rag.filter_by_score.__defaults__ = (abs_floor, rel_floor)


def sweep(top_k: int = 5) -> list[dict]:
    orig_defaults = rag.filter_by_score.__defaults__
    results = []
    try:
        for abs_floor, rel_floor in itertools.product(ABS_CANDIDATES, REL_CANDIDATES):
            _set_floors(abs_floor, rel_floor)

            report = run_eval(top_k=top_k)
            positives = [r for r in report if r["type"] == "positive"]
            negatives = [r for r in report if r["type"] == "negative_control"]

            mean_recall = sum(r["recall@k"] for r in positives) / len(positives)
            mean_mrr = sum(r["mrr"] for r in positives) / len(positives)
            neg_pass_rate = sum(r["passed"] for r in negatives) / len(negatives) if negatives else 1.0

            row = {
                "abs_floor": abs_floor, "rel_floor": rel_floor,
                "mean_recall": mean_recall, "mean_mrr": mean_mrr,
                "neg_pass_rate": neg_pass_rate,
            }
            results.append(row)
            print(f"abs={abs_floor:.2f}  rel={rel_floor:.2f}  "
                  f"recall@{top_k}={mean_recall:.3f}  mrr={mean_mrr:.3f}  "
                  f"neg_pass={neg_pass_rate:.0%}")
    finally:
        rag.filter_by_score.__defaults__ = orig_defaults  # restore, don't leave global state mutated

    # "Best" = highest recall among configs that don't sacrifice negative-control
    # abstention -- a lower floor will always look better on recall alone, but
    # if it also makes the pipeline confidently answer out-of-corpus questions,
    # that's not a win, it's a hallucination risk.
    safe = [r for r in results if r["neg_pass_rate"] == 1.0]
    pool = safe if safe else results
    best = max(pool, key=lambda r: (r["mean_recall"], r["mean_mrr"]))

    print(f"\nBest negative-control-safe config: abs_floor={best['abs_floor']}, "
          f"rel_floor={best['rel_floor']}  "
          f"-> recall@{top_k}={best['mean_recall']:.3f}, mrr={best['mean_mrr']:.3f}")
    if not safe:
        print("WARNING: no config in the grid kept a 100% negative-control pass rate -- "
              "widen the grid or inspect the negative controls individually.")
    print("\nUpdate rag.py's ABS_SCORE_FLOOR / REL_SCORE_FLOOR constants to these values "
          "(and note the corpus size next to them, same as the existing comment, so the "
          "next person knows when to re-run this).")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    sweep(top_k=args.top_k)