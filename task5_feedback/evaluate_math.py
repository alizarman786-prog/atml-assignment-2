from __future__ import annotations

import argparse
import json
import time

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.models import load_policy, load_tokenizer
from task5_feedback.rlvr import exact_reward, extract_designated_final

SUFFIX = "\n\nShow your reasoning and end your response with exactly `#### <number>`."
POLICIES = ["sft", "rlvr", "rlaif"]
PAIRS = [("rlaif", "sft"), ("rlvr", "sft"), ("rlvr", "rlaif")]


def policy_specs(cfg):
    return {"sft": None, "rlvr": cfg["policies"]["rlvr"], "rlaif": cfg["policies"]["rlaif"]}


def dataset_path(cfg, dataset):
    return cfg["paths"]["gsm_eval"] if dataset == "gsm" else cfg["paths"]["math_transfer_eval"]


def out_dir(cfg):
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    d.mkdir(parents=True, exist_ok=True)
    return d


def row_prompt(row):
    if isinstance(row.get("messages"), list):
        return row["messages"]
    return [{"role": "user", "content": str(row["question"]) + SUFFIX}]


def row_gold(row):
    if row.get("gold_final") not in (None, ""):
        return str(row["gold_final"])
    return extract_designated_final(row.get("gold_solution", ""))


def generate_stage(cfg, dataset, name, bs, limit):
    tag = "_smoke" if limit else ""
    path = out_dir(cfg) / f"generated_{dataset}_{name}{tag}.jsonl"
    if path.exists():
        print("skip (exists):", path)
        return
    rows = read_jsonl(dataset_path(cfg, dataset))[: limit or None]
    prompts = [row_prompt(r) for r in rows]
    src = "messages" if isinstance(rows[0].get("messages"), list) else "question + suffix"
    print(f"{dataset}/{name}: {len(rows)} rows, prompt built from {src}; keys {sorted(rows[0])}")
    print("first prompt:", prompts[0][-1]["content"][:300].replace("\n", " "))
    tok = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=policy_specs(cfg)[name], trainable=False)
    order = sorted(range(len(rows)), key=lambda i: len(prompts[i][-1]["content"]))
    recs, t0 = [None] * len(rows), time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        g = batch_generate(model, tok, [prompts[i] for i in idx], max_prompt_length=512,
                           max_new_tokens=int(cfg["math_max_new_tokens"]), temperature=0.0, top_p=1.0, do_sample=False)
        for k, i in enumerate(idx):
            gold, resp = row_gold(rows[i]), g["responses"][k]
            pred = extract_designated_final(resp)
            recs[i] = dict(idx=i, dataset=dataset, policy=name, prompt_id=rows[i].get("prompt_id", rows[i].get("source_index")),
                           question=rows[i]["question"], gold=gold, response=resp, pred=pred, exact=exact_reward(resp, gold),
                           formatted=pred is not None, n_tokens=int(g["response_lengths"][k]), truncated=bool(g["truncated"][k]))
        print(f"  {min(s + bs, len(order))}/{len(order)} in {time.time() - t0:.0f}s", flush=True)
    write_jsonl(path, recs)
    print("accuracy:", round(float(np.mean([r["exact"] for r in recs])), 3), "| formatted:",
          round(float(np.mean([r["formatted"] for r in recs])), 3), "->", path)


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def policy_metrics(recs):
    n, k = len(recs), int(sum(r["exact"] for r in recs))
    L = np.array([r["n_tokens"] for r in recs], float)
    return dict(n=n, accuracy=k / n, accuracy_ci95=wilson(k, n), format_compliance=float(np.mean([r["formatted"] for r in recs])),
                mean_tokens=float(L.mean()), std_tokens=float(L.std(ddof=1)), frac_truncated=float(np.mean([r["truncated"] for r in recs])),
                failure_no_final=int(sum(not r["formatted"] for r in recs)),
                failure_wrong_final=int(sum(r["formatted"] and not r["exact"] for r in recs)),
                failure_truncated_no_final=int(sum(r["truncated"] and not r["formatted"] for r in recs)))


def pair_summary(rows):
    s, n = np.array([r["score"] for r in rows]), len(rows)
    c = {p: sum(r["pref"] == p for r in rows) for p in ("A", "TIE", "B")}
    diff = [r for r in rows if r["exact_a"] != r["exact_b"]]
    agree = sum(r["pref"] == ("A" if r["exact_a"] > r["exact_b"] else "B") for r in diff)
    tie = sum(r["pref"] == "TIE" for r in diff)
    secs = [r["judge_s"] for r in rows if r["judge_s"] > 0]
    return dict(n=n, win_rate=float(s.mean()), se=float(s.std(ddof=1) / np.sqrt(n)), wins=c["A"], ties=c["TIE"], losses=c["B"],
                identical_pairs=int(sum(r["identical"] for r in rows)), n_verifier_distinguishes=len(diff),
                judge_agrees_with_verifier=agree, judge_tie_on_those=tie, judge_disagrees_on_those=len(diff) - agree - tie,
                agreement_rate=agree / len(diff) if diff else None, mean_judge_s_uncached=float(np.mean(secs)) if secs else None,
                n_uncached_judge_calls=len(secs))


def judge_stage(cfg, dataset, limit):
    from task5_feedback.rlaif import PairwiseAIJudge
    tag, d = ("_smoke" if limit else ""), out_dir(cfg)
    G = {n: read_jsonl(d / f"generated_{dataset}_{n}{tag}.jsonl") for n in POLICIES}
    t = time.perf_counter()
    for r in G["sft"]:
        exact_reward(r["response"], r["gold"])
    summary = {"policies": {n: policy_metrics(G[n]) for n in POLICIES}, "pairs": {},
               "verifier_seconds_per_call": (time.perf_counter() - t) / len(G["sft"])}
    judge = PairwiseAIJudge(cfg, "results/task5_feedback/judge_cache.json")
    for a, b in PAIRS:
        rows = []
        for ra, rb in zip(G[a], G[b]):
            if ra["response"] == rb["response"]:       # identical text: counted as a tie, judge not called
                pref, sec, ident = "TIE", 0.0, True
            else:
                n0, t0 = len(judge.cache), time.time()
                pref = judge.compare(ra["question"], ra["response"], rb["response"])
                sec, ident = (time.time() - t0 if len(judge.cache) > n0 else 0.0), False
            rows.append(dict(idx=ra["idx"], a=a, b=b, pref=pref, score={"A": 1.0, "TIE": 0.5, "B": 0.0}[pref], identical=ident,
                             exact_a=ra["exact"], exact_b=rb["exact"], judge_s=sec))
        write_jsonl(d / f"judged_{dataset}_{a}_vs_{b}{tag}.jsonl", rows)
        summary["pairs"][f"{a}_vs_{b}"] = pair_summary(rows)
        print(f"{a} vs {b}: win rate {summary['pairs'][f'{a}_vs_{b}']['win_rate']:.3f}", flush=True)
    (d / f"summary_{dataset}{tag}.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--stage", choices=["generate", "judge"], required=True)
    ap.add_argument("--policy", choices=POLICIES)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="SMOKE TEST ONLY: first N rows, writes *_smoke files")
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    if a.stage == "generate":
        if not a.policy:
            ap.error("--policy is required for --stage generate")
        generate_stage(cfg, a.dataset, a.policy, a.batch_size, a.limit)
    else:
        s = judge_stage(cfg, a.dataset, a.limit)
        for n, m in s["policies"].items():
            print(f"{n:6s} acc {m['accuracy']:.3f} fmt {m['format_compliance']:.3f} tokens {m['mean_tokens']:.0f}+-{m['std_tokens']:.0f} trunc {m['frac_truncated']:.3f}")


if __name__ == "__main__":
    main()
