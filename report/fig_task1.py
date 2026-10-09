from figures_common import *


def a(ax):
    d = jlines(R / "task1_dpo/train_standard.jsonl").drop_duplicates("step", keep="last").sort_values("step")
    ax.plot(d["step"], d["loss"].rolling(10, min_periods=1).mean(), color=BLUE)
    ax.axhline(np.log(2), color=GREY, lw=0.6, ls=":")
    for s in d["step"][d["grad_norm"].isna()]:
        ax.axvline(s, color=RED, lw=0.7, alpha=0.7)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("DPO loss (10-step mean)", color=BLUE)
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(d["step"], d["pref_acc"].rolling(10, min_periods=1).mean(), color=GREEN)
    ax2.set_ylabel("batch preference accuracy", color=GREEN)


def b(ax):
    S = pd.DataFrame(jload(R / "task1_dpo/beta_sweep_summary.json")).sort_values("beta")
    err = 1.96 * np.sqrt(S["pref_acc"] * (1 - S["pref_acc"]) / 290)
    ax.errorbar(S["beta"], S["pref_acc"], yerr=err, marker="o", color=BLUE, capsize=2, label="short forks (600 pairs)")
    st = jload(R / "task1_dpo/eval_standard.json")["pairs_standard"]
    p, n = st["pref_accuracy"], st["n_pairs"]
    ax.errorbar([0.108], [p], yerr=[1.96 * np.sqrt(p * (1 - p) / n)], marker="s", color=GREEN, capsize=2, label="standard (1446 pairs)")
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([0.03, 0.1, 0.3])
    ax.set_xticklabels(["0.03", "0.1", "0.3"])
    ax.set_xlabel("beta")
    ax.set_ylabel("held-out preference accuracy")
    ax.legend(frameon=False, loc="lower left")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(S["beta"], S["median_margin"], color=AMBER, ls="--", marker="^")
    ax2.set_ylabel("median margin (nats)", color=AMBER)


def c(ax):
    P = jload(R / "task1_dpo/length_analysis.json")["pairs"]
    order = ["preferred_longer", "length_matched", "rejected_longer"]
    x = np.arange(3)
    for off, key, lab, col in [(-0.2, "acc_standard", "standard", BLUE), (0.2, "acc_balanced", "length-balanced", AMBER)]:
        y = np.array([P[s][key] for s in order])
        n = [P[s]["n"] for s in order]
        ci = np.array([wilson(int(round(v * m)), m) for v, m in zip(y, n)])
        ax.bar(x + off, y, 0.38, color=col, label=lab, yerr=[y - ci[:, 0], ci[:, 1] - y], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(["preferred\nlonger", "length\nmatched", "rejected\nlonger"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("held-out preference accuracy")
    ax.legend(frameon=False, loc="upper left")


fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
run_panels([(axes[0], "(a) standard DPO training", a), (axes[1], "(b) beta sweep", b), (axes[2], "(c) accuracy by length stratum", c)])
finish(fig, "fig_task1")
