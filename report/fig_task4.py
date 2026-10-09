from figures_common import *

POL = ["sft", "dpo", "ppo", "grpo"]


def a(ax):
    r = pd.read_csv(R / "task4_safety/safety_rates.csv").set_index("policy").loc[POL]
    x = np.arange(4)
    for off, key, col, lab in [(-0.2, "safe_answer", GREEN, "safe: answered"),
                               (0.2, "unsafe_justified_refusal", BLUE, "unsafe: refused")]:
        y, lo, hi = r[key].values, r[key + "_lo"].values, r[key + "_hi"].values
        ax.bar(x + off, y, 0.38, color=col, label=lab, yerr=[y - lo, hi - y], capsize=2, error_kw={"lw": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POL])
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("rate (450 prompts)")
    ax.legend(frameon=False, loc="upper center", ncol=2)


def b(ax):
    A = jload(R / "task4_safety/safety_summary.json")["audit"]["policy_rates_on_audit_items"]
    x = np.arange(4)
    for off, key, col, lab in [(-0.12, "over_refusal_manual", RED, "manual labels"), (0.12, "over_refusal_judge", GREY, "AI judge")]:
        y = np.array([A[p][key] for p in POL])
        n = [A[p]["n_safe"] for p in POL]
        ci = np.array([wilson(int(round(v * m)), m) for v, m in zip(y, n)])
        ax.errorbar(x + off, y, yerr=[y - ci[:, 0], ci[:, 1] - y], fmt="o", color=col, capsize=2, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([p.upper() for p in POL])
    ax.set_ylim(-0.02, 0.5)
    ax.set_ylabel("over-refusal (30 safe prompts)")
    ax.legend(frameon=False, loc="upper right")


def c(ax):
    C = pd.read_csv(R / "task4_safety/manual_vs_judge_confusion.csv", index_col=0)
    ax.imshow(C.values, cmap="Blues")
    short = ["safe\nanswer", "justified\nrefusal", "unsafe\ncompl.", "over-\nrefusal", "ambig."]
    ax.set_xticks(range(5))
    ax.set_xticklabels(["safe answer", "justified refusal", "unsafe compl.", "over-refusal", "ambig."], rotation=40, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(short)
    for i in range(5):
        for j in range(5):
            v = C.values[i, j]
            ax.text(j, i, int(v), ha="center", va="center", color="white" if v > C.values.max() / 2 else "black")
    ax.tick_params(labelsize=5)
    ax.set_xlabel("AI judge")
    ax.set_ylabel("manual label")
    ax.spines[:].set_visible(False)


fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2))
run_panels([(axes[0], "(a) AI-judge rates", a), (axes[1], "(b) over-refusal, audit", b), (axes[2], "(c) manual vs judge", c)])
finish(fig, "fig_task4")
