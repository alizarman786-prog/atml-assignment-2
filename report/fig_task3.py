from figures_common import *

KS = [2, 4, 8]


def a(ax):
    G = jload(R / "task3_grpo/group_size_study.json")
    x = np.arange(3)
    for key, col, lab in [("informative_rate", BLUE, "informative"),
                          ("trainable_group_rate", RED, "trainable")]:
        y = np.array([G[f"K{k}"]["all"][key] for k in KS])
        ci = np.array([G[f"K{k}"]["all"][key + "_ci95"] for k in KS])
        ax.errorbar(x, y, yerr=[y - ci[:, 0], ci[:, 1] - y], marker="o", color=col, capsize=2, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in KS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of groups")
    ax.legend(frameon=False, loc="lower right")


def b(ax):
    G = jload(R / "task3_grpo/group_size_study.json")
    bins = ["hard", "medium", "easy"]
    for j, (k, col) in enumerate(zip(KS, [BLUE, GREEN, AMBER])):
        ax.bar(np.arange(3) + (j - 1) * 0.26, [G[f"K{k}"][bn]["trainable_group_rate"] for bn in bins], 0.26, color=col, label=f"K={k}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(bins)
    ax.set_ylabel("trainable-group rate")
    ax.set_xlabel("prompt difficulty tercile")
    ax.set_ylim(0, 1.25)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.legend(frameon=False, loc="upper center", ncol=3)


def c(ax):
    d = jlines(R / "task3_grpo/train_standard.jsonl").drop_duplicates("update", keep="last").sort_values("update")
    ax.bar(d["update"], d["grad_norm"], color=GREY)
    z = d[d["grad_norm"] == 0]
    ax.scatter(z["update"], [0] * len(z), marker="x", color=RED, s=30, zorder=3, clip_on=False,
               label="all masked (zero grad.)")
    ax.set_ylim(0, d["grad_norm"].max() * 1.4)
    ax.set_xlabel("GRPO update")
    ax.set_ylabel("gradient norm")
    ax.legend(frameon=False, loc="upper center")


fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.15))
run_panels([(axes[0], "(a) groups by group size", a), (axes[1], "(b) by difficulty", b),
            (axes[2], "(c) standard run", c)])
finish(fig, "fig_task3")
