from figures_common import *


def a(ax):
    P = jload(R / "task5_feedback/perturbations.json")["by_variant"]
    names = ["corrupt_reasoning_correct_final", "good_reasoning_wrong_final", "gold_distractor_wrong_final", "persuasive_filler_correct"]
    labels = ["reasoning\ncorrupted", "wrong\nfinal", "gold\ndistractor", "persuasive\nfiller"]
    for i, nm in enumerate(names):
        for j, (src, tag) in enumerate([("verifier", "V"), ("judge_pooled", "J")]):
            x, bot = i * 3 + j, 0.0
            for key, col in [("better", GREEN), ("tie", GREY), ("wrong", RED)]:
                v = P[nm][src][key]
                ax.bar(x, v, 0.9, bottom=bot, color=col, label=key if (i == 0 and j == 0) else None)
                bot += v
            ax.text(x, 1.02, tag, ha="center", fontsize=7)
    ax.set_xticks([i * 3 + 0.5 for i in range(4)])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.38)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("fraction of pairs (V verifier, J judge)")
    ax.legend(frameon=False, loc="upper center", ncol=3, title="preference for the clean response", title_fontsize=7)


def b(ax):
    A = jload(R / "task5_feedback/accuracy_paired.json")
    names = ["sft", "rlvr", "rlaif"]
    ticks, labs = [], []
    for di, ds in enumerate(["gsm", "transfer"]):
        for ni, n in enumerate(names):
            p, x = A[ds]["policies"][n], di * 4 + ni
            lo, hi = wilson(int(round(p["strict_acc"] * p["n"])), p["n"])
            ax.bar(x - 0.17, p["strict_acc"], 0.34, color=BLUE, yerr=[[p["strict_acc"] - lo], [hi - p["strict_acc"]]], capsize=2,
                   error_kw={"lw": 0.8}, label="strict (#### field)" if x == 0 else None)
            ax.bar(x + 0.17, p["lenient_acc"], 0.34, color=AMBER, label="lenient diagnostic" if x == 0 else None)
            ticks.append(x)
            labs.append(n.upper())
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, rotation=30)
    ax.text(1, -0.27, "GSM8K (300)", ha="center", transform=ax.get_xaxis_transform())
    ax.text(5, -0.27, "SVAMP (100)", ha="center", transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, loc="upper left")


fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
run_panels([(axes[0], "(a) controlled diagnostic pairs", a), (axes[1], "(b) accuracy, strict vs lenient", b)])
finish(fig, "fig_task5")
