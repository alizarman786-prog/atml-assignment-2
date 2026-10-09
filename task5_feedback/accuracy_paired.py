from __future__ import annotations

import json
import re

import pandas as pd
from scipy.stats import binomtest

from common.data import load_yaml, read_jsonl, repo_path

NUM = r"[-+]?\d[\d,]*(?:\.\d+)?"
NAMES = ["sft", "rlvr", "rlaif"]
PAIRS = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]


def lenient(text):
    for pat in (r"####\s*(" + NUM + ")", r"\\boxed\{\s*\\?\$?\s*(" + NUM + ")"):
        found = re.findall(pat, text)
        if found:
            return found[-1].replace(",", "")
    found = re.findall(NUM, text)
    return found[-1].replace(",", "") if found else None


def equal(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except Exception:
        return False


def mcnemar(x, y):
    only_a = int((x & ~y).sum())
    only_b = int((~x & y).sum())
    p = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return {"a_only_correct": only_a, "b_only_correct": only_b, "p_exact": float(p)}


def summarize(g):
    f = g[g.formatted]
    return {"n": len(g), "strict_acc": float(g.exact.mean()), "format": float(g.formatted.mean()),
            "n_formatted": int(g.formatted.sum()), "acc_given_formatted": float(f.exact.mean()),
            "lenient_acc": float(g.lenient_ok.mean()), "truncated": float(g.truncated.mean())}


def main():
    cfg = load_yaml("configs/feedback.yaml")
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    out = {}
    for ds in ("gsm", "transfer"):
        G = {n: pd.DataFrame(read_jsonl(d / f"generated_{ds}_{n}.jsonl")).set_index("idx") for n in NAMES}
        for g in G.values():
            g["lenient_ok"] = [equal(lenient(r), x) for r, x in zip(g.response, g.gold)]
        out[ds] = {"policies": {n: summarize(g) for n, g in G.items()}, "paired": {}}
        for n in NAMES:
            print(ds, n, {k: round(v, 3) for k, v in out[ds]["policies"][n].items()})
        for a, b in PAIRS:
            for col in ("exact", "lenient_ok"):
                res = mcnemar(G[a][col].astype(bool).values, G[b][col].astype(bool).values)
                out[ds]["paired"][a + "_vs_" + b + "_" + col] = res
                print("  ", a, "vs", b, col, res)
    (d / "accuracy_paired.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
