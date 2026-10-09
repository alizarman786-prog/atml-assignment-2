from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {"clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
                     "persuasive_filler_correct", "gold_distractor_wrong_final"}
# every perturbed variant is paired with the clean correct response; "better" = the clean one is preferred
KIND = {"corrupt_reasoning_correct_final": "reasoning", "good_reasoning_wrong_final": "outcome",
        "gold_distractor_wrong_final": "outcome", "persuasive_filler_correct": "style"}
O1 = {"A": "better", "B": "wrong", "TIE": "tie"}   # compare(clean, variant): A = clean preferred
O2 = {"A": "wrong", "B": "better", "TIE": "tie"}   # compare(variant, clean): B = clean preferred


def load_diagnostic_groups(path):
    by_problem = defaultdict(dict)
    for row in read_jsonl(path):
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def rates(verdicts):
    n = len(verdicts)
    return {"n": n, **{k: sum(v == k for v in verdicts) / n for k in ("better", "tie", "wrong")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--limit-problems", type=int, help="SMOKE TEST ONLY")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    tag = "_smoke" if args.limit_problems else ""
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    d.mkdir(parents=True, exist_ok=True)
    allrows = [r for v in groups.values() for r in v.values()]
    ver_ok = float(np.mean([exact_reward(r["response"], str(r["gold_final"])) == r["expected_exact_reward"] for r in allrows]))
    judge = PairwiseAIJudge(cfg, "results/task5_feedback/judge_cache.json")
    recs, t0 = [], time.time()
    for pid in sorted(groups)[: args.limit_problems or None]:
        clean = groups[pid]["clean_correct"]
        q = clean["question"]
        for vt, kind in KIND.items():
            var = groups[pid][vt]
            ec = exact_reward(clean["response"], str(clean["gold_final"]))
            ev = exact_reward(var["response"], str(var["gold_final"]))
            p1 = judge.compare(q, clean["response"], var["response"])
            p2 = judge.compare(q, var["response"], clean["response"])
            recs.append(dict(problem_id=pid, variant=vt, kind=kind, verifier="better" if ec > ev else ("wrong" if ec < ev else "tie"),
                             judge_o1=O1[p1], judge_o2=O2[p2], words_clean=len(clean["response"].split()),
                             words_variant=len(var["response"].split())))
        print(f"  problem {pid} done ({time.time() - t0:.0f}s)", flush=True)
    out = {"n_problems": len({r["problem_id"] for r in recs}), "verifier_matches_expected_reward": ver_ok, "by_variant": {}, "by_kind": {}}
    for vt in KIND:
        R = [r for r in recs if r["variant"] == vt]
        out["by_variant"][vt] = {"verifier": rates([r["verifier"] for r in R]),
                                 "judge_pooled": rates([r["judge_o1"] for r in R] + [r["judge_o2"] for r in R]),
                                 "judge_clean_first": rates([r["judge_o1"] for r in R]), "judge_variant_first": rates([r["judge_o2"] for r in R]),
                                 "judge_order_consistency": float(np.mean([r["judge_o1"] == r["judge_o2"] for r in R])),
                                 "mean_words_clean": float(np.mean([r["words_clean"] for r in R])),
                                 "mean_words_variant": float(np.mean([r["words_variant"] for r in R]))}
    for kind in ("reasoning", "outcome", "style"):
        R = [r for r in recs if r["kind"] == kind]
        out["by_kind"][kind] = {"verifier": rates([r["verifier"] for r in R]),
                                "judge_pooled": rates([r["judge_o1"] for r in R] + [r["judge_o2"] for r in R])}
    out["S_reason"] = {m: out["by_kind"]["reasoning"][m]["better"] for m in ("verifier", "judge_pooled")}
    out["S_outcome"] = {m: out["by_kind"]["outcome"][m]["better"] for m in ("verifier", "judge_pooled")}
    write_jsonl(d / f"perturbation_pairs{tag}.jsonl", recs)
    (d / f"perturbations{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"verifier matches course expected_exact_reward on {ver_ok:.3f} of the rows")
    for vt, v in out["by_variant"].items():
        print(f"{vt:34s} verifier {v['verifier']['better']:.2f}/{v['verifier']['tie']:.2f}/{v['verifier']['wrong']:.2f} | "
              f"judge {v['judge_pooled']['better']:.2f}/{v['judge_pooled']['tie']:.2f}/{v['judge_pooled']['wrong']:.2f} (better/tie/wrong) | order-consistent {v['judge_order_consistency']:.2f}")
    print("S_reason:", out["S_reason"], "| S_outcome:", out["S_outcome"])


if __name__ == "__main__":
    main()
