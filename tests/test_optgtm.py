import numpy as np
import pytest

from src.environment import Environment
from src.strategies import TruthfulArm, OverReporterArm
from src.simulation import run_episode
from src.algorithms.optgtm import OneSidedOptGTM


@pytest.mark.parametrize("invalid_T", [0, 1])
def test_optgtm_requires_horizon_at_least_two(invalid_T):
    with pytest.raises(ValueError, match="at least 2"):
        OneSidedOptGTM(K=2, d=2, T=invalid_T)


@pytest.mark.parametrize("invalid_scale", [0.0, -0.5, np.inf, np.nan])
def test_optgtm_rejects_invalid_radius_scale(invalid_scale):
    with pytest.raises(ValueError, match="radius_scale"):
        OneSidedOptGTM(
            K=2,
            d=2,
            T=100,
            radius_scale=invalid_scale,
        )


def test_radius_scale_multiplies_paper_radius():
    paper = OneSidedOptGTM(K=2, d=2, T=100, radius_scale=1.0)
    scaled = OneSidedOptGTM(K=2, d=2, T=100, radius_scale=0.25)

    assert np.isclose(
        scaled.beta_root(5),
        0.25 * paper.beta_root(5),
    )


@pytest.fixture(scope="module")
def truthful_optgtm_runs():
    """Run truthful OptGTM once for each seed and reuse the results."""
    results = []

    for seed in range(20):
        env = Environment(seed=seed)

        learner = OneSidedOptGTM(
            K=env.K,
            d=env.d,
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
                "env": env,
                "learner": learner,
                "log": log,
            }
        )

    return results


def test_optgtm_does_not_eliminate_truthful_arms(
    truthful_optgtm_runs,
):
    """Truthful arms should remain active under one-sided OptGTM."""
    for result in truthful_optgtm_runs:
        seed = result["seed"]
        learner = result["learner"]

        assert learner.active.all(), (
            f"OptGTM eliminated a truthful arm for seed {seed}. "
            f"Active arms: {learner.active.tolist()}, "
            f"elimination pulls: {learner.eliminated_at}"
        )

        assert all(
            elimination is None
            for elimination in learner.eliminated_at
        )


def test_truthful_optgtm_regret_decreases_over_time(
    truthful_optgtm_runs,
):
    """The average late regret should be below early regret."""
    # Preserve the original reference experiment using seed 0.
    result = next(
        result
        for result in truthful_optgtm_runs
        if result["seed"] == 0
    )

    env = result["env"]
    log = result["log"]

    quarter = env.T // 4

    early_mean_regret = np.mean(
        log["regret"][:quarter]
    )

    late_mean_regret = np.mean(
        log["regret"][-quarter:]
    )

    assert late_mean_regret < 0.5 * early_mean_regret, (
        "OptGTM regret did not decrease sufficiently. "
        f"Early mean regret={early_mean_regret:.6f}, "
        f"late mean regret={late_mean_regret:.6f}."
    )


def test_constant_realizable_overreporting_is_not_detected_in_reference_case():
    """Regression test for the known linearly realizable report sequence.

    This does not claim that every constant over-reporting strategy is
    undetectable. It checks the particular reference environment, seed,
    strategy and manipulation magnitude used by the project.
    """
    env = Environment(seed=0)

    attacker = 0

    strategies = [
        TruthfulArm()
        for _ in range(env.K)
    ]

    strategies[attacker] = OverReporterArm(
        delta=0.30
    )

    learner = OneSidedOptGTM(
        K=env.K,
        d=env.d,
        T=env.T,
        sigma=env.sigma,
    )

    log = run_episode(
        env=env,
        learner=learner,
        strategies=strategies,
    )

    assert learner.eliminated_at[attacker] is None, (
        "The reference constant over-reporter was unexpectedly "
        f"eliminated at attacker pull "
        f"{learner.eliminated_at[attacker]}."
    )

    assert learner.active[attacker]

    # In the existing reference experiment, no other arms should be
    # falsely eliminated either.
    assert learner.active.all(), (
        "At least one arm was unexpectedly eliminated. "
        f"Active arms: {learner.active.tolist()}, "
        f"elimination pulls: {learner.eliminated_at}"
    )

    assert np.isfinite(log["cum_regret"][-1])
