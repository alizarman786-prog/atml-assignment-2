from __future__ import annotations

import argparse
import json

from common.data import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    cfg = load_yaml(ap.parse_args().config)
    d = repo_path(cfg["results_dir"]) / "task5_feedback"
    G, T = json.load(open(d / "summary_gsm.json")), json.load(open(d / "summary_transfer.json"))
    P = json.load(open(d / "perturbations.json"))
    rows = []
    for n in ("sft", "rlvr", "rlaif"):
        g, t = G["policies"][n], T["policies"][n]
        rows.append(dict(policy=n, gsm_acc=g["accuracy"], svamp_acc=t["accuracy"], acc_drop=g["accuracy"] - t["accuracy"],
                         gsm_format=g["format_compliance"], svamp_format=t["format_compliance"], gsm_tokens=g["mean_tokens"],
                         svamp_tokens=t["mean_tokens"], svamp_no_final=t["failure_no_final"], svamp_wrong_final=t["failure_wrong_final"]))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()})
    for name, S in (("GSM8K", G), ("SVAMP", T)):
        for k, v in S["pairs"].items():
            print(f"{name} {k}: win {v['win_rate']:.3f}+-{v['se']:.3f} (W/T/L {v['wins']}/{v['ties']}/{v['losses']}, identical {v['identical_pairs']}) | "
                  f"judge agrees with verifier on {v['judge_agrees_with_verifier']}/{v['n_verifier_distinguishes']}, ties {v['judge_tie_on_those']}")
    print("verifier s/call:", G["verifier_seconds_per_call"], "| judge s/uncached call:", {k: v["mean_judge_s_uncached"] for k, v in G["pairs"].items()})
    print("S_reason:", P["S_reason"], "| S_outcome:", P["S_outcome"])
    (d / "final_comparison.json").write_text(json.dumps({"policies": rows, "gsm_pairs": G["pairs"], "svamp_pairs": T["pairs"], "perturbations": P}, indent=1, default=float))


if __name__ == "__main__":
    main()
