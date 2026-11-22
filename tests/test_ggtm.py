import numpy as np
import pytest

from src.environment import Environment
from src.strategies import TruthfulArm, OverReporterArm
from src.simulation import run_episode
from src.algorithms.ggtm import GGTM


@pytest.mark.parametrize("invalid_T", [0, 1])
def test_ggtm_requires_horizon_at_least_two(invalid_T):
    with pytest.raises(ValueError, match="at least 2"):
        GGTM(
            K=2,
            theta_star=np.array([1.0, 0.0]),
            T=invalid_T,
        )


@pytest.fixture(scope="module")
def truthful_ggtm_runs():
    """Run GGTM with truthful arms for seeds 0 through 19."""
    results = []

    for seed in range(20):
        env = Environment(seed=seed)

        learner = GGTM(
            K=env.K,
            theta_star=env.theta,
            T=env.T,
            sigma=env.sigma,
        )

        strategies = [
            TruthfulArm()
            for _ in range(env.K)
        ]

        log = run_episode(
            env=env,
            learner=learner,
            strategies=strategies,
        )

        results.append(
            {
                "seed": seed,
                "learner": learner,
                "log": log,
            }
        )

    return results


def test_ggtm_does_not_eliminate_truthful_arms(
    truthful_ggtm_runs,
):
    for result in truthful_ggtm_runs:
        learner = result["learner"]
        seed = result["seed"]

        assert learner.active.all(), (
            f"GGTM eliminated a truthful arm for seed {seed}. "
            f"Active arms: {learner.active.tolist()}, "
            f"elimination pulls: {learner.eliminated_at}"
        )


def test_ggtm_has_zero_regret_with_known_theta_and_truthful_reports(
    truthful_ggtm_runs,
):
    for result in truthful_ggtm_runs:
        log = result["log"]
        seed = result["seed"]

        assert np.allclose(
            log["regret"],
            0.0,
            atol=1e-12,
        ), (
            f"Expected zero regret for truthful GGTM, "
            f"but seed {seed} produced final cumulative regret "
            f"{log['cum_regret'][-1]:.12f}."
        )

        assert np.isclose(
            log["cum_regret"][-1],
            0.0,
            atol=1e-12,
        )


@pytest.mark.parametrize("seed", range(5))
def test_ggtm_detects_constant_overreporting(seed):
    env = Environment(seed=seed)

    strategies = [
        TruthfulArm()
        for _ in range(env.K)
    ]

    attacker = 0

    strategies[attacker] = OverReporterArm(
        delta=0.30
    )

    learner = GGTM(
        K=env.K,
        theta_star=env.theta,
        T=env.T,
        sigma=env.sigma,
    )

    log = run_episode(
        env=env,
        learner=learner,
        strategies=strategies,
    )

    elimination_pull = learner.eliminated_at[attacker]

    assert elimination_pull is not None, (
        f"GGTM failed to eliminate the over-reporting arm "
        f"for seed {seed}."
    )

    # This threshold is specific to the current synthetic environment,
    # noise level and manipulation magnitude.
    assert elimination_pull <= 20, (
        f"Over-reporting arm was detected later than expected "
        f"for seed {seed}: detection at attacker pull "
        f"{elimination_pull}."
    )

    assert not learner.active[attacker]

    # Optional consistency check: the learner completed the episode.
    assert len(log["arm"]) == env.T


def test_all_arms_eliminated_returns_complete_zero_reward_log():
    env = Environment(seed=0, K=2, T=20)
    learner = GGTM(
        K=env.K,
        theta_star=env.theta,
        T=env.T,
        sigma=env.sigma,
    )
    learner.active[:] = False

    log = run_episode(
        env=env,
        learner=learner,
        strategies=[TruthfulArm() for _ in range(env.K)],
    )

    assert log["stopped_at"] == 0
    assert np.all(log["arm"] == -1)
    assert np.all(log["reward"] == 0.0)
    assert np.all(log["true_mean"] == 0.0)
    assert np.allclose(log["regret"], env._means.max(axis=1))
    assert np.allclose(log["cum_regret"], np.cumsum(log["regret"]))
