from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = repo_path(cfg["results_dir"]) / "task4_safety"
    ids = set(pd.read_csv(d / "manual_audit_ids.csv")["xstest_id"].tolist())
    rows = []
    for name in ["sft", "dpo", "ppo", "grpo"]:
        for r in read_jsonl(d / f"generated_{name}.jsonl"):
            if r["xstest_id"] in ids:
                rows.append(dict(xstest_id=r["xstest_id"], policy=name, prompt=r["prompt"],
                                 response=r["response"], cls=r["benchmark_class"]))
    df = pd.DataFrame(rows)
    items = [(k, sorted(g["policy"])) for k, g in df.groupby(["xstest_id", "prompt", "response", "cls"], sort=False)]
    order = np.random.default_rng(int(cfg["seed"])).permutation(len(items))  # shuffled: policy identity not visible
    sheet, key = [], []
    for new_id, j in enumerate(order, 1):
        (xid, pr, resp, cls), pols = items[j]
        sheet.append(dict(item_id=new_id, prompt=pr, response=resp, xstest_class=cls, manual_label=""))
        key.append(dict(item_id=new_id, xstest_id=xid, policies="|".join(pols)))
    pd.DataFrame(sheet).to_csv(d / "manual_audit_sheet.csv", index=False)
    pd.DataFrame(key).to_csv(d / "manual_audit_key.csv", index=False)
    print(f"{len(ids)} prompt IDs x 4 policies = {len(df)} responses -> {len(items)} unique items to label")


if __name__ == "__main__":
    main()
