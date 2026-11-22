"""Tests for the independent per-arm LinUCB implementation."""

import numpy as np
import pytest

from src.environment import Environment
from src.strategies import TruthfulArm
from src.simulation import run_episode
from src.algorithms.linucb_per_arm import PerArmLinUCB


@pytest.fixture(scope="module")
def truthful_linucb_run():
    """Run the reference truthful LinUCB experiment once."""
    env = Environment(seed=0)

    learner = PerArmLinUCB(
        K=env.K,
        d=env.d,
        alpha=1.0,
        lam=1.0,
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

    return {
        "env": env,
        "learner": learner,
        "log": log,
    }


def test_log_has_expected_shapes(truthful_linucb_run):
    """All logged arrays should match the environment horizon."""
    env = truthful_linucb_run["env"]
    log = truthful_linucb_run["log"]

    assert log["round"].shape == (env.T,)
    assert log["arm"].shape == (env.T,)
    assert log["true_mean"].shape == (env.T,)
    assert log["reward"].shape == (env.T,)
    assert log["regret"].shape == (env.T,)
    assert log["cum_regret"].shape == (env.T,)
    assert log["ucb"].shape == (env.T, env.K)


def test_logged_round_numbers_are_correct(truthful_linucb_run):
    env = truthful_linucb_run["env"]
    log = truthful_linucb_run["log"]

    assert np.array_equal(
        log["round"],
        np.arange(env.T),
    )


def test_logged_arms_are_valid(truthful_linucb_run):
    env = truthful_linucb_run["env"]
    log = truthful_linucb_run["log"]

    assert np.all(log["arm"] >= 0)
    assert np.all(log["arm"] < env.K)


def test_regret_is_nonnegative(truthful_linucb_run):
    """Pseudo-regret cannot be negative."""
    log = truthful_linucb_run["log"]

    assert np.all(
        log["regret"] >= -1e-12
    )


def test_logged_rewards_contain_predrawn_environment_noise(
    truthful_linucb_run,
):
    """Observed reward minus true mean should equal round-level noise."""
    env = truthful_linucb_run["env"]
    log = truthful_linucb_run["log"]

    realized_noise = (
        log["reward"]
        - log["true_mean"]
    )

    assert np.allclose(
        realized_noise,
        env._noise,
        atol=1e-12,
    )


def test_cumulative_regret_is_correct(truthful_linucb_run):
    """cum_regret should be the cumulative sum of instantaneous regret."""
    log = truthful_linucb_run["log"]

    expected_cumulative_regret = np.cumsum(
        log["regret"]
    )

    assert np.allclose(
        log["cum_regret"],
        expected_cumulative_regret,
    )


def test_cumulative_regret_is_monotone(truthful_linucb_run):
    """Cumulative pseudo-regret should never decrease."""
    cumulative_regret = truthful_linucb_run["log"]["cum_regret"]

    assert np.all(
        np.diff(cumulative_regret) >= -1e-12
    )


def test_all_rounds_produce_finite_values(truthful_linucb_run):
    """The algorithm should not produce NaNs or infinite values."""
    log = truthful_linucb_run["log"]

    for key in (
        "true_mean",
        "reward",
        "regret",
        "cum_regret",
        "ucb",
    ):
        assert np.all(
            np.isfinite(log[key])
        ), f"Non-finite values found in log['{key}']."


def test_total_pull_count_equals_horizon(truthful_linucb_run):
    """Exactly one arm should be selected in every round."""
    env = truthful_linucb_run["env"]
    learner = truthful_linucb_run["learner"]

    assert learner.pulls.sum() == env.T


def test_pull_counts_match_logged_actions(truthful_linucb_run):
    """Learner pull counters should agree with the action log."""
    env = truthful_linucb_run["env"]
    learner = truthful_linucb_run["learner"]
    log = truthful_linucb_run["log"]

    logged_pull_counts = np.bincount(
        log["arm"],
        minlength=env.K,
    )

    assert np.array_equal(
        learner.pulls,
        logged_pull_counts,
    )


def test_regret_decreases_over_time(truthful_linucb_run):
    """Late per-round regret should be smaller than early regret."""
    env = truthful_linucb_run["env"]
    log = truthful_linucb_run["log"]

    quarter = env.T // 4

    early_mean_regret = np.mean(
        log["regret"][:quarter]
    )

    late_mean_regret = np.mean(
        log["regret"][-quarter:]
    )

    assert late_mean_regret < 0.35 * early_mean_regret, (
        "Per-arm LinUCB did not reduce regret sufficiently. "
        f"Early mean regret={early_mean_regret:.6f}, "
        f"late mean regret={late_mean_regret:.6f}."
    )


def test_second_half_adds_small_fraction_of_total_regret(
    truthful_linucb_run,
):
    """The regret curve should flatten rather than remain linear."""
    env = truthful_linucb_run["env"]
    cumulative_regret = truthful_linucb_run["log"]["cum_regret"]

    final_regret = cumulative_regret[-1]

    assert final_regret > 0.0

    regret_added_in_second_half = (
        final_regret
        - cumulative_regret[env.T // 2]
    )

    second_half_share = (
        regret_added_in_second_half
        / final_regret
    )

    assert second_half_share < 0.35, (
        "Too much regret was accumulated in the second half. "
        f"Second-half share={second_half_share:.3f}."
    )


def test_learner_concentrates_on_highest_quality_arm(
    truthful_linucb_run,
):
    """The learner should pull arm 0 more than any other arm."""
    learner = truthful_linucb_run["learner"]

    assert learner.pulls.argmax() == 0, (
        f"Expected arm 0 to receive the most pulls, "
        f"but pull counts were {learner.pulls.tolist()}."
    )


def test_best_arm_receives_substantial_fraction_of_pulls(
    truthful_linucb_run,
):
    """Arm 0 should receive more than 40% of all selections."""
    env = truthful_linucb_run["env"]
    learner = truthful_linucb_run["learner"]

    best_arm_pull_fraction = (
        learner.pulls[0]
        / env.T
    )

    assert best_arm_pull_fraction > 0.40, (
        f"Arm 0 received only "
        f"{best_arm_pull_fraction:.1%} of pulls. "
        f"Pull counts: {learner.pulls.tolist()}."
    )


def test_estimator_matrices_have_expected_shapes(
    truthful_linucb_run,
):
    env = truthful_linucb_run["env"]
    learner = truthful_linucb_run["learner"]

    assert learner.V.shape == (
        env.K,
        env.d,
        env.d,
    )

    assert learner.b.shape == (
        env.K,
        env.d,
    )

    assert learner.pulls.shape == (
        env.K,
    )


def test_covariance_matrices_remain_symmetric(
    truthful_linucb_run,
):
    """Every ridge covariance matrix should remain symmetric."""
    learner = truthful_linucb_run["learner"]

    for arm in range(learner.K):
        assert np.allclose(
            learner.V[arm],
            learner.V[arm].T,
            atol=1e-12,
        )


def test_covariance_matrices_remain_positive_definite(
    truthful_linucb_run,
):
    """Ridge regularization should keep all covariance matrices PD."""
    learner = truthful_linucb_run["learner"]

    for arm in range(learner.K):
        eigenvalues = np.linalg.eigvalsh(
            learner.V[arm]
        )

        assert np.all(eigenvalues > 0.0), (
            f"Arm {arm} has a non-positive-definite "
            f"covariance matrix: eigenvalues={eigenvalues}."
        )


@pytest.mark.parametrize(
    ("alpha", "lam"),
    [
        (-0.1, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
    ],
)
def test_invalid_hyperparameters_raise(alpha, lam):
    with pytest.raises(ValueError):
        PerArmLinUCB(
            K=3,
            d=2,
            alpha=alpha,
            lam=lam,
        )


def test_update_changes_only_selected_arm():
    """One update must not alter another arm's estimator."""
    learner = PerArmLinUCB(
        K=3,
        d=2,
        alpha=1.0,
        lam=1.0,
    )

    original_V = learner.V.copy()
    original_b = learner.b.copy()
    original_pulls = learner.pulls.copy()

    selected_arm = 1
    reported_context = np.array([0.6, 0.8])
    reward = 0.5

    learner.update(
        arm=selected_arm,
        x_reported=reported_context,
        reward=reward,
    )

    # Selected arm should change.
    assert not np.array_equal(
        learner.V[selected_arm],
        original_V[selected_arm],
    )

    assert not np.array_equal(
        learner.b[selected_arm],
        original_b[selected_arm],
    )

    assert learner.pulls[selected_arm] == 1

    # All other arms should remain unchanged.
    for arm in (0, 2):
        assert np.array_equal(
            learner.V[arm],
            original_V[arm],
        )

        assert np.array_equal(
            learner.b[arm],
            original_b[arm],
        )

        assert (
            learner.pulls[arm]
            == original_pulls[arm]
        )


def test_update_matches_ridge_sufficient_statistics():
    """V and b should receive the standard linear-regression update."""
    learner = PerArmLinUCB(
        K=2,
        d=2,
        alpha=1.0,
        lam=1.0,
    )

    arm = 0
    context = np.array([0.6, 0.8])
    reward = 0.5

    expected_V = (
        np.eye(2)
        + np.outer(context, context)
    )

    expected_b = reward * context

    learner.update(
        arm=arm,
        x_reported=context,
        reward=reward,
    )

    assert np.allclose(
        learner.V[arm],
        expected_V,
    )

    assert np.allclose(
        learner.b[arm],
        expected_b,
    )


def test_theta_hat_matches_linear_solve():
    learner = PerArmLinUCB(
        K=2,
        d=2,
        alpha=1.0,
        lam=1.0,
    )

    learner.update(
        arm=0,
        x_reported=np.array([1.0, 0.0]),
        reward=0.5,
    )

    expected_theta = np.linalg.solve(
        learner.V[0],
        learner.b[0],
    )

    assert np.allclose(
        learner.theta_hat(0),
        expected_theta,
    )


def test_initial_selection_uses_deterministic_tie_break():
    """Equal initial UCBs should select the lowest-index arm."""
    learner = PerArmLinUCB(
        K=3,
        d=2,
        alpha=1.0,
        lam=1.0,
    )

    # All reports have the same norm and all estimators are initially equal.
    reported = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    selected_arm, ucb_values = learner.select(
        reported
    )

    assert np.allclose(
        ucb_values,
        ucb_values[0],
    )

    assert selected_arm == 0