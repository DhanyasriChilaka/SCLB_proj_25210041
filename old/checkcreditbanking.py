"""Plan step 6: credit-banking attacker vs. one-sided OptGTM.

    python check_credit_banking.py
    python check_credit_banking.py --a_minus 0.2 --a_plus 0.3 --T0 1000

Arm 0 (the strong arm in Environment's controlled mode) plays, on the SAME
seed: truthful, OverReporterArm(a_plus), CreditBankerArm(a_minus, a_plus, T0).
All other arms are truthful. For each run it takes the (l, u, r) tuples the
learner stored at pull time and reports arm 0's slack

    slack = sum r - sum l        (eliminated when slack < -reward_slack(m))

per round, plus pulls, attack gain G (pulls minus the truthful run's pulls)
and the elimination round. The plot shows the three slack curves for the
first seed.
"""
import argparse
import numpy as np
from Environment import Environment
from Linucbperarm import run
from optgtm import OneSidedOptGTM
from Strategies import TruthfulArm, OverReporterArm, CreditBankerArm

ATTACKER = 0
def episode(seed, strategy, T0):
    """One run with `strategy` on the attacker arm; returns its slack trace."""
    env = Environment(seed=seed)
    strategies = [TruthfulArm() for _ in range(env.K)]
    strategies[ATTACKER] = strategy
    learner = OneSidedOptGTM(env.K, env.d, T=env.T, sigma=env.sigma)
    log = run(env, learner, strategies)

    rounds = np.flatnonzero(log["arm"] == ATTACKER)     # round of each pull
    rec = np.array(learner.records[ATTACKER])           # (l, u, r) per pull
    m_elim = learner.eliminated_at[ATTACKER]            # pull count or None
    return dict(
        name=strategy.name,
        t=rounds,
        slack=np.cumsum(rec[:, 2] - rec[:, 0]),
        trigger=-learner.reward_slack(np.arange(1, len(rec) + 1)),
        pulls=len(rec),
        pulls_before_T0=int((rounds < T0).sum()),
        elim_round=None if m_elim is None else int(rounds[m_elim - 1]),
        regret=float(log["cum_regret"][-1]),
        false_elims=int((~np.delete(learner.active, ATTACKER)).sum()),
    )


def slack_at(res, t):
    """Slack held at the start of round t (0 if not pulled before t)."""
    k = int(np.searchsorted(res["t"], t))
    return float(res["slack"][k - 1]) if k > 0 else 0.0


def plot(runs, T0, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for res, color in zip(runs, ("0.55", "C3", "C0")):
        ax.plot(res["t"], res["slack"], color=color, lw=1.8, label=res["name"])
        if res["elim_round"] is not None:
            ax.plot(res["elim_round"], res["slack"][-1], "x", color=color,
                    ms=10, mew=2.5)
    ax.plot(runs[-1]["t"], runs[-1]["trigger"], "C0:", lw=1,
            label="banker's trigger level")
    ax.axvline(T0, color="k", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("round t  (dashed line = $T_0$)")
    ax.set_ylabel(r"arm 0 slack  $\sum r - \sum \ell$")
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--a_minus", type=float, default=0.2)
    p.add_argument("--a_plus", type=float, default=0.3)
    p.add_argument("--T0", type=int, default=1000)           # 0.2 T
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--plot", default="credit_banking_slack.png")
    a = p.parse_args()

    show = lambda e: "never" if e is None else f"t={e}"
    for n, seed in enumerate(a.seeds):
        tru = episode(seed, TruthfulArm(), a.T0)
        nai = episode(seed, OverReporterArm(a.a_plus), a.T0)
        ban = episode(seed, CreditBankerArm(a.a_minus, a.a_plus, a.T0), a.T0)

        print(f"\nseed {seed}")
        for r in (tru, nai, ban):
            print(f"  {r['name']:14s} pulls={r['pulls']:5d}  "
                  f"G={r['pulls'] - tru['pulls']:+5d}  "
                  f"eliminated: {show(r['elim_round']):7s} "
                  f"regret={r['regret']:6.1f}  "
                  f"truthful arms eliminated={r['false_elims']}")
        print(f"  pulls before T0: banker {ban['pulls_before_T0']}, "
              f"truthful {tru['pulls_before_T0']}")
        s_ban, s_tru = slack_at(ban, a.T0), slack_at(tru, a.T0)
        print(f"  slack at T0: banker {s_ban:.1f}, truthful {s_tru:.1f}  "
              f"-> banked credit {s_ban - s_tru:+.1f}")
        after = ban["slack"][ban["t"] >= a.T0]
        if len(after):
            print(f"  banker slack after T0: lowest {after.min():.1f}, "
                  f"final {after[-1]:.1f}")
        if n == 0:
            plot((tru, nai, ban), a.T0, a.plot,
                 f"one-sided OptGTM, seed {seed}: $a_-$={a.a_minus}, "
                 f"$a_+$={a.a_plus}, $T_0$={a.T0}")
            print(f"  plot -> {a.plot}")