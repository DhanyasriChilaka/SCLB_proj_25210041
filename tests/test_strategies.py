import numpy as np
import pytest

from src.environment import Environment
from src.strategies import (
    TruthfulArm,
    UnderReporterArm,
    OverReporterArm,
    CreditBankerArm,
    PullCountCreditBankerArm,
)


@pytest.fixture
def sample_context():
    """Return one environment and one true arm context."""
    env = Environment(seed=0)
    x_true = env.contexts()[0]
    return env, x_true


def test_truthful_report_equals_true_context(sample_context):
    env, x_true = sample_context

    strategy = TruthfulArm()
    reported = strategy.report(
        t=0,
        x_true=x_true,
        theta_star=env.theta,
    )

    assert np.array_equal(reported, x_true)


def test_truthful_report_is_a_copy(sample_context):
    """Changing the report must not change the environment's context."""
    env, x_true = sample_context

    strategy = TruthfulArm()
    reported = strategy.report(
        t=0,
        x_true=x_true,
        theta_star=env.theta,
    )

    assert reported is not x_true
    assert not np.shares_memory(reported, x_true)

    original = x_true.copy()
    reported[0] += 1.0

    assert np.array_equal(x_true, original)


def test_underreporting_preserves_unit_norm(sample_context):
    env, x_true = sample_context

    strategy = UnderReporterArm(delta=0.10)
    reported = strategy.report(
        t=0,
        x_true=x_true,
        theta_star=env.theta,
    )

    assert np.isclose(
        np.linalg.norm(reported),
        1.0,
        atol=1e-10,
    )


def test_underreporting_reduces_theta_component(sample_context):
    env, x_true = sample_context

    delta = 0.10
    strategy = UnderReporterArm(delta=delta)

    reported = strategy.report(
        t=0,
        x_true=x_true,
        theta_star=env.theta,
    )

    true_component = float(x_true @ env.theta)
    reported_component = float(reported @ env.theta)

    expected_component = max(true_component - delta, -1.0)

    assert np.isclose(
        reported_component,
        expected_component,
        atol=1e-10,
    )

    assert reported_component <= true_component


def test_overreporting_preserves_unit_norm(sample_context):
    env, x_true = sample_context

    strategy = OverReporterArm(delta=0.15)
    reported = strategy.report(
        t=0,
        x_true=x_true,
        theta_star=env.theta,
    )

    assert np.isclose(
        np.linalg.norm(reported),
        1.0,
        atol=1e-10,
    )


def test_overreporting_increases_theta_component(sample_context):
    env, x_true = sample_context

    delta = 0.15
    strategy = OverReporterArm(delta=delta)

    reported = strategy.report(
        t=0,
        x_true=x_true,
        theta_star=env.theta,
    )

    true_component = float(x_true @ env.theta)
    reported_component = float(reported @ env.theta)

    expected_component = min(true_component + delta, 1.0)

    assert np.isclose(
        reported_component,
        expected_component,
        atol=1e-10,
    )

    assert reported_component >= true_component

def test_iid_reporting_preserves_true_context_norm():
    env = Environment(seed=0, context_mode="iid")
    x_true = env.contexts()[0]

    reported = OverReporterArm(delta=0.1).report(
        t=0,
        x_true=x_true,
        theta_star=env.theta,
    )

    assert np.isclose(
        np.linalg.norm(reported),
        np.linalg.norm(x_true),
        atol=1e-10,
    )


def test_shift_clipped_to_context_boundary_tolerates_roundoff():
    """A feasible clipped shift must not fail by a floating-point ulp."""
    env = Environment(seed=4, context_mode="iid", T=50)
    strategy = UnderReporterArm(delta=10.0)

    for x_true in env.contexts():
        reported = strategy.report(
            t=0,
            x_true=x_true,
            theta_star=env.theta,
        )

        radius = np.linalg.norm(x_true)
        assert np.isclose(np.linalg.norm(reported), radius, atol=1e-10)
        assert np.isclose(reported @ env.theta, -radius, atol=1e-10)


def test_reporting_does_not_mutate_theta_star():
    theta_star = np.array([2.0, 0.0, 0.0])
    original = theta_star.copy()
    x_true = np.array([0.5, np.sqrt(0.75), 0.0])

    OverReporterArm(delta=0.1).report(
        t=0,
        x_true=x_true,
        theta_star=theta_star,
    )

    assert np.array_equal(theta_star, original)


def test_credit_banker_underreports_before_switch(sample_context):
    env, x_true = sample_context

    banker = CreditBankerArm(
        a_minus=0.10,
        a_plus=0.15,
        T0=5,
    )

    reported = banker.report(
        t=4,
        x_true=x_true,
        theta_star=env.theta,
    )

    true_component = float(x_true @ env.theta)
    reported_component = float(reported @ env.theta)

    assert np.isclose(
        reported_component,
        max(true_component - 0.10, -1.0),
        atol=1e-10,
    )


def test_credit_banker_overreports_at_and_after_switch(sample_context):
    env, x_true = sample_context

    banker = CreditBankerArm(
        a_minus=0.10,
        a_plus=0.15,
        T0=5,
    )

    # t == T0 is the first over-reporting round.
    reported = banker.report(
        t=5,
        x_true=x_true,
        theta_star=env.theta,
    )

    true_component = float(x_true @ env.theta)
    reported_component = float(reported @ env.theta)

    assert np.isclose(
        reported_component,
        min(true_component + 0.15, 1.0),
        atol=1e-10,
    )


def test_credit_banker_phases_match_individual_strategies(
    sample_context,
):
    """The banker should reuse the same under/over-reporting behavior."""
    env, x_true = sample_context

    banker = CreditBankerArm(
        a_minus=0.10,
        a_plus=0.15,
        T0=5,
    )

    early = banker.report(
        t=4,
        x_true=x_true,
        theta_star=env.theta,
    )

    late = banker.report(
        t=5,
        x_true=x_true,
        theta_star=env.theta,
    )

    expected_early = UnderReporterArm(delta=0.10).report(
        t=4,
        x_true=x_true,
        theta_star=env.theta,
    )

    expected_late = OverReporterArm(delta=0.15).report(
        t=5,
        x_true=x_true,
        theta_star=env.theta,
    )

    assert np.allclose(early, expected_early)
    assert np.allclose(late, expected_late)


@pytest.mark.parametrize("t", [0, 4, 5, 10])
def test_credit_banker_with_zero_shifts_is_truthful(
    sample_context,
    t,
):
    env, x_true = sample_context

    banker = CreditBankerArm(
        a_minus=0.0,
        a_plus=0.0,
        T0=5,
    )

    reported = banker.report(
        t=t,
        x_true=x_true,
        theta_star=env.theta,
    )

    assert np.array_equal(reported, x_true)
    assert reported is not x_true


def test_pull_count_banker_switches_after_selected_pull_threshold(
    sample_context,
):
    env, x_true = sample_context
    banker = PullCountCreditBankerArm(
        a_minus=0.10,
        a_plus=0.15,
        bank_pulls=2,
    )

    first = banker.report(0, x_true, env.theta)
    banker.on_selected(0, 0.0, first)
    second = banker.report(5, x_true, env.theta)
    banker.on_selected(5, 0.0, second)
    third = banker.report(6, x_true, env.theta)

    q = float(x_true @ env.theta)
    assert np.isclose(first @ env.theta, max(q - 0.10, -1.0))
    assert np.isclose(second @ env.theta, max(q - 0.10, -1.0))
    assert np.isclose(third @ env.theta, min(q + 0.15, 1.0))


def test_pull_count_banker_reset_restores_banking_phase(sample_context):
    env, x_true = sample_context
    banker = PullCountCreditBankerArm(bank_pulls=1)

    report = banker.report(0, x_true, env.theta)
    banker.on_selected(0, 0.0, report)
    assert banker.selected_pulls == 1

    banker.reset()
    assert banker.selected_pulls == 0


def test_report_strategies_do_not_change_true_reward(sample_context):
    """Reports must not affect the environment's reward generation."""
    env, x_true = sample_context

    expected_reward = (
        env.expected_rewards(t=0)[0]
        + env._noise[0]
    )

    strategy = OverReporterArm(delta=0.15)

    # Produce a manipulated report, but never pass it to env.step().
    strategy.report(
        t=0,
        x_true=x_true,
        theta_star=env.theta,
    )

    actual_reward = env.step(0)

    assert np.isclose(actual_reward, expected_reward)
