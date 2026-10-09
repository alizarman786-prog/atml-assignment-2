from figures_common import *
from matplotlib.ticker import MaxNLocator


def std():
    d = jlines(R / "task2_ppo/train_standard.jsonl")
    return d.drop_duplicates("update", keep="last").sort_values("update")


def a(ax):
    d = std()
    ax.plot(d["update"], d["task_reward"], color=BLUE, marker="o", ms=3, lw=0.8)
    m = d[d["missing_eos"] > 0]
    ax.scatter(m["update"], m["task_reward"], color=RED, zorder=3, s=16, label="no EOS (hit 512 cap)")
    ax.set_xlabel("PPO update")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("effective terminal reward")
    ax.legend(frameon=False)


def b(ax):
    d = std()
    ax.bar(d["update"], d["value_explained_var"], color=GREY)
    ax.axhline(0, color="k", lw=0.6)
    sk = d[d["value_skipped_steps"] > 0]
    ax.scatter(sk["update"], [0.5] * len(sk), marker="v", color=RED, label="critic step skipped (fp16 overflow)")
    ax.set_xlabel("PPO update")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("critic explained variance")
    ax.set_ylim(ax.get_ylim()[0], 5)
    ax.legend(frameon=False, loc="upper right")


def c(ax):
    T = jload(R / "task2_ppo/cached_clipping.json")["trajectories"]["unclipped"]
    eps = ["0.05", "0.2", "0.5"]
    vals = [float(np.mean([t["affected_frac_eps" + e] for t in T])) for e in eps]
    ax.bar(range(3), vals, color=[BLUE, GREEN, AMBER])
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["eps=" + e for e in eps])
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_ylabel("affected-token fraction")


def d_(ax):
    names = ["base", "midpoint", "standard", "center", "eps0p05", "eps0p50", "kl0p00", "kl0p20"]
    E = {n: jload(R / f"task2_ppo/eval_{n}.json") for n in names}
    y = np.arange(len(names))[::-1]
    ax.errorbar([E[n]["reward_effective"]["mean"] for n in names], y, xerr=[1.96 * E[n]["reward_effective"]["sem"] for n in names],
                fmt="o", color=BLUE, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("held-out reward (mean, 95% CI)")


fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.0))
run_panels([(axes[0, 0], "(a) reward per update", a), (axes[0, 1], "(b) critic quality", b),
            (axes[1, 0], "(c) clipping, cached batch", c), (axes[1, 1], "(d) held-out reward, 8 policies", d_)])
finish(fig, "fig_task2")
