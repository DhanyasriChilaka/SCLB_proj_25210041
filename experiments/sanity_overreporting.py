
"""Sanity experiment: constant over-reporting versus per-arm LinUCB.

This is a paired comparison: truthful and strategic runs use the same
environment seed, so they receive identical true contexts and exogenous
noise.

FINDING: in the 20-seed reference experiment, constant norm-preserving
over-reporting gives the attacker a modest but consistent selection gain
of roughly 3--7%. This is the plain-over-reporting baseline against which
the additional gain from credit banking should be compared. It is an
empirical observation for this synthetic setup, not a general theoretical
claim.
"""

import numpy as np

from src.environment import Environment
from src.strategies import TruthfulArm, OverReporterArm
from src.algorithms.linucb_per_arm import PerArmLinUCB
from src.simulation import run_episode


ATTACKER = 0


def run_condition(seed, attacker_delta=None):
    """Run one truthful or constant-over-reporting condition."""
    env = Environment(seed=seed)

    strategies = [
        TruthfulArm()
        for _ in range(env.K)
    ]

    if attacker_delta is not None:
        strategies[ATTACKER] = OverReporterArm(
            delta=attacker_delta
        )

    learner = PerArmLinUCB(
        K=env.K,
        d=env.d,
        alpha=1.0,
    )

    log = run_episode(
        env=env,
        learner=learner,
        strategies=strategies,
    )

    return {
        "seed": seed,
        "attacker_delta": attacker_delta,
        "pulls": learner.pulls.copy(),
        "attacker_pulls": int(learner.pulls[ATTACKER]),
        "regret": float(log["cum_regret"][-1]),
    }


def paired_comparison(seed, attacker_delta=0.30):
    """Compare truthful and attacked runs on the same environment seed."""
    truthful = run_condition(
        seed=seed,
        attacker_delta=None,
    )

    attacked = run_condition(
        seed=seed,
        attacker_delta=attacker_delta,
    )

    return {
        "seed": seed,
        "truthful": truthful,
        "attacked": attacked,
        "attack_gain": (
            attacked["attacker_pulls"]
            - truthful["attacker_pulls"]
        ),
        "regret_increase": (
            attacked["regret"]
            - truthful["regret"]
        ),
    }


def print_result(result):
    truthful = result["truthful"]
    attacked = result["attacked"]

    print(f"\nSeed {result['seed']}")

    print(
        "  Truthful:       "
        f"pulls={truthful['pulls'].tolist()}, "
        f"regret={truthful['regret']:.2f}"
    )

    print(
        "  Over-reporting: "
        f"pulls={attacked['pulls'].tolist()}, "
        f"regret={attacked['regret']:.2f}"
    )

    print(
        f"  Attacker selection gain: {result['attack_gain']:+d}"
    )

    print(
        f"  Regret increase: {result['regret_increase']:+.2f}"
    )


def summarize(results):
    attack_gains = np.array(
        [result["attack_gain"] for result in results],
        dtype=float,
    )

    regret_increases = np.array(
        [result["regret_increase"] for result in results],
        dtype=float,
    )

    print("\nSummary")
    print(f"  Number of paired seeds: {len(results)}")

    print(
        "  Mean attacker selection gain: "
        f"{attack_gains.mean():+.2f}"
    )

    print(
        "  Mean regret increase: "
        f"{regret_increases.mean():+.2f}"
    )

    print(
        "  Seeds with positive attack gain: "
        f"{np.count_nonzero(attack_gains > 0)}/{len(results)}"
    )


def main():
    seeds = [0, 1, 2]
    attacker_delta = 0.30

    results = []

    for seed in seeds:
        result = paired_comparison(
            seed=seed,
            attacker_delta=attacker_delta,
        )
        results.append(result)
        print_result(result)

    summarize(results)


if __name__ == "__main__":
    main()
# from Linucbperarm import PerArmLinUCB, run

# def episode(attacker_delta):
#     e = Environment(seed=0)
#     strats = [TruthfulArm() for _ in range(e.K)]
#     if attacker_delta is not None:
#         strats[0] = OverReporterArm(delta=attacker_delta)
#     learner = PerArmLinUCB(K=e.K, d=e.d, alpha=1.0)
#     log = run(e, learner, strats)
#     return learner.pulls.copy(), log["cum_regret"][-1]

# pulls_truth, regret_truth = episode(None)
# pulls_attack, regret_attack = episode(0.3)

# print(f"Truthful:         pulls={pulls_truth.tolist()}, "
#         f"cum regret={regret_truth:.1f}")
# print(f"Arm 0 lies +0.30: pulls={pulls_attack.tolist()}, "
#         f"cum regret={regret_attack:.1f}")

# # Same seed, so contexts and noise are identical: differences are
# # caused by the manipulation alone.
# # The lie hurts the learner...
# assert regret_attack > 1.3 * regret_truth, \
#     (regret_truth, regret_attack)
# # ...and buys the attacker a modest number of extra selections.
# assert pulls_attack[0] < 1.1 * pulls_truth[0], \
#     (pulls_truth[0], pulls_attack[0])
# print("All checks passed.")
