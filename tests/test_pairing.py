"""Tests for paired experimental comparisons and common randomness."""

import numpy as np
import pytest

from experiments.credit_banking import (
    run_credit_banking_condition,
    run_paired_credit_banking_comparison,
)
from src.algorithms.optgtm import OneSidedOptGTM
from src.strategies import (
    TruthfulArm,
    CreditBankerArm,
    PullCountCreditBankerArm,
)


def make_one_sided_optgtm(env):
    """Construct a fresh learner for one experimental condition."""
    return OneSidedOptGTM(
        K=env.K,
        d=env.d,
        T=env.T,
        sigma=env.sigma,
    )


def test_condition_defaults_to_lightweight_summary():
    summary = run_credit_banking_condition(
        seed=3,
        learner_factory=make_one_sided_optgtm,
        strategy_factory=TruthfulArm,
        env_kwargs={"T": 30},
    )

    assert summary["seed"] == 3
    assert summary["radius_scale"] == 1.0
    assert "stopped_at" in summary
    assert "log" not in summary


def test_full_results_skip_traces_unless_requested():
    result = run_credit_banking_condition(
        seed=3,
        learner_factory=make_one_sided_optgtm,
        strategy_factory=TruthfulArm,
        env_kwargs={"T": 30},
        summary_only=False,
    )

    assert result["traces"] is None

    representative = run_credit_banking_condition(
        seed=3,
        learner_factory=make_one_sided_optgtm,
        strategy_factory=TruthfulArm,
        env_kwargs={"T": 30},
        summary_only=False,
        collect_traces=True,
    )

    assert representative["traces"] is not None


@pytest.fixture(scope="module")
def identical_pair():
    """Two truthful conditions on the same environment seed."""
    return run_paired_credit_banking_comparison(
        seed=7,
        learner_factory=make_one_sided_optgtm,
        baseline_strategy_factory=TruthfulArm,
        attack_strategy_factory=TruthfulArm,
        attacker=0,
        env_kwargs={
            "T": 300,
        },
        summary_only=False,
    )


@pytest.fixture(scope="module")
def truthful_vs_banker_pair():
    """Truthful baseline versus credit banker on the same seed."""
    return run_paired_credit_banking_comparison(
        seed=11,
        learner_factory=make_one_sided_optgtm,
        baseline_strategy_factory=TruthfulArm,
        attack_strategy_factory=lambda: CreditBankerArm(
            a_minus=0.20,
            a_plus=0.30,
            T0=100,
        ),
        attacker=0,
        env_kwargs={
            "T": 300,
        },
        summary_only=False,
    )


@pytest.fixture(scope="module")
def pull_count_banker_pair():
    """Zero-shift baseline versus pull-count credit banker."""
    return run_paired_credit_banking_comparison(
        seed=13,
        learner_factory=make_one_sided_optgtm,
        baseline_strategy_factory=lambda: PullCountCreditBankerArm(
            a_minus=0.0,
            a_plus=0.0,
            bank_pulls=5,
        ),
        attack_strategy_factory=lambda: PullCountCreditBankerArm(
            a_minus=0.20,
            a_plus=0.30,
            bank_pulls=5,
        ),
        attacker=0,
        env_kwargs={"T": 100},
        summary_only=False,
    )


def test_identical_conditions_have_identical_action_logs(
    identical_pair,
):
    baseline_log = identical_pair["baseline"]["log"]
    attack_log = identical_pair["attack"]["log"]

    assert np.array_equal(
        baseline_log["arm"],
        attack_log["arm"],
    )


def test_identical_conditions_have_identical_rewards(
    identical_pair,
):
    baseline_log = identical_pair["baseline"]["log"]
    attack_log = identical_pair["attack"]["log"]

    assert np.array_equal(
        baseline_log["reward"],
        attack_log["reward"],
    )

    assert np.array_equal(
        baseline_log["true_mean"],
        attack_log["true_mean"],
    )


def test_identical_conditions_have_identical_regret(
    identical_pair,
):
    baseline_log = identical_pair["baseline"]["log"]
    attack_log = identical_pair["attack"]["log"]

    assert np.array_equal(
        baseline_log["regret"],
        attack_log["regret"],
    )

    assert np.array_equal(
        baseline_log["cum_regret"],
        attack_log["cum_regret"],
    )


def test_identical_conditions_have_zero_attack_gain(
    identical_pair,
):
    paired_summary = identical_pair["paired_summary"]

    assert paired_summary["attack_gain"] == 0

    assert np.isclose(
        paired_summary["regret_increase"],
        0.0,
        atol=1e-12,
    )


def test_time_threshold_pipeline_reports_phase_split(
    truthful_vs_banker_pair,
):
    summary = truthful_vs_banker_pair["paired_summary"]

    assert summary["phase_switch_round"] == 100
    assert (
        summary["banking_phase_gain"]
        + summary["spending_phase_gain"]
        == summary["attack_gain"]
    )


def test_pull_count_banking_pipeline_is_connected(pull_count_banker_pair):
    attack = pull_count_banker_pair["attack"]
    summary = pull_count_banker_pair["paired_summary"]

    assert attack["summary"]["strategy"] == "pull_count_credit_banker"
    assert attack["summary"]["bank_pulls_target"] == 5
    assert (
        attack["strategy"].selected_pulls
        == attack["summary"]["attacker_pulls"]
    )
    assert summary["banking_specific_gain"] == summary["attack_gain"]
    assert summary["phase_type"] == "pull"
    assert not summary["pair_censored"]


def test_paired_conditions_use_distinct_environment_objects(
    identical_pair,
):
    """Pairing should share randomness, not mutable objects."""
    baseline_env = identical_pair["baseline"]["env"]
    attack_env = identical_pair["attack"]["env"]

    assert baseline_env is not attack_env


def test_paired_conditions_use_distinct_learner_objects(
    identical_pair,
):
    baseline_learner = identical_pair["baseline"]["learner"]
    attack_learner = identical_pair["attack"]["learner"]

    assert baseline_learner is not attack_learner


def test_paired_conditions_use_distinct_strategy_objects(
    identical_pair,
):
    baseline_strategy = identical_pair["baseline"]["strategy"]
    attack_strategy = identical_pair["attack"]["strategy"]

    assert baseline_strategy is not attack_strategy


def test_paired_environments_have_identical_theta(
    truthful_vs_banker_pair,
):
    baseline_env = truthful_vs_banker_pair["baseline"]["env"]
    attack_env = truthful_vs_banker_pair["attack"]["env"]

    assert np.array_equal(
        baseline_env.theta,
        attack_env.theta,
    )


def test_paired_environments_have_identical_true_contexts(
    truthful_vs_banker_pair,
):
    """Both policies must face the same complete context sequence."""
    baseline_env = truthful_vs_banker_pair["baseline"]["env"]
    attack_env = truthful_vs_banker_pair["attack"]["env"]

    assert np.array_equal(
        baseline_env._X,
        attack_env._X,
    )

    assert np.array_equal(
        baseline_env._means,
        attack_env._means,
    )


def test_paired_environments_have_identical_exogenous_noise(
    truthful_vs_banker_pair,
):
    baseline_env = truthful_vs_banker_pair["baseline"]["env"]
    attack_env = truthful_vs_banker_pair["attack"]["env"]

    assert np.array_equal(
        baseline_env._noise,
        attack_env._noise,
    )


def test_different_policies_receive_same_realized_noise(
    truthful_vs_banker_pair,
):
    """Reward-minus-mean must agree even if selected arms differ."""
    baseline = truthful_vs_banker_pair["baseline"]
    attack = truthful_vs_banker_pair["attack"]

    baseline_log = baseline["log"]
    attack_log = attack["log"]

    baseline_noise = (
        baseline_log["reward"]
        - baseline_log["true_mean"]
    )

    attack_noise = (
        attack_log["reward"]
        - attack_log["true_mean"]
    )

    assert np.allclose(
        baseline_noise,
        baseline["env"]._noise,
        atol=1e-12,
    )

    assert np.allclose(
        attack_noise,
        attack["env"]._noise,
        atol=1e-12,
    )

    assert np.allclose(
        baseline_noise,
        attack_noise,
        atol=1e-12,
    )


def test_paired_attack_gain_matches_raw_action_logs(
    truthful_vs_banker_pair,
):
    baseline_log = truthful_vs_banker_pair["baseline"]["log"]
    attack_log = truthful_vs_banker_pair["attack"]["log"]
    paired_summary = truthful_vs_banker_pair["paired_summary"]

    attacker = 0

    baseline_pulls = int(
        np.count_nonzero(
            baseline_log["arm"] == attacker
        )
    )

    attack_pulls = int(
        np.count_nonzero(
            attack_log["arm"] == attacker
        )
    )

    expected_attack_gain = (
        attack_pulls
        - baseline_pulls
    )

    assert (
        paired_summary["baseline_attacker_pulls"]
        == baseline_pulls
    )

    assert (
        paired_summary["attack_attacker_pulls"]
        == attack_pulls
    )

    assert (
        paired_summary["attack_gain"]
        == expected_attack_gain
    )


def test_paired_regret_increase_matches_raw_logs(
    truthful_vs_banker_pair,
):
    baseline_log = truthful_vs_banker_pair["baseline"]["log"]
    attack_log = truthful_vs_banker_pair["attack"]["log"]
    paired_summary = truthful_vs_banker_pair["paired_summary"]

    baseline_regret = float(
        baseline_log["cum_regret"][-1]
    )

    attack_regret = float(
        attack_log["cum_regret"][-1]
    )

    expected_regret_increase = (
        attack_regret
        - baseline_regret
    )

    assert np.isclose(
        paired_summary["baseline_regret"],
        baseline_regret,
    )

    assert np.isclose(
        paired_summary["attack_regret"],
        attack_regret,
    )

    assert np.isclose(
        paired_summary["regret_increase"],
        expected_regret_increase,
    )


def test_summary_seed_and_algorithm_agree_between_conditions(
    truthful_vs_banker_pair,
):
    baseline_summary = truthful_vs_banker_pair[
        "baseline"
    ]["summary"]

    attack_summary = truthful_vs_banker_pair[
        "attack"
    ]["summary"]

    paired_summary = truthful_vs_banker_pair[
        "paired_summary"
    ]

    assert baseline_summary["seed"] == 11
    assert attack_summary["seed"] == 11
    assert paired_summary["seed"] == 11

    assert (
        baseline_summary["algorithm"]
        == attack_summary["algorithm"]
        == paired_summary["algorithm"]
        == "OneSidedOptGTM"
    )


def test_repeating_paired_experiment_is_deterministic(
    truthful_vs_banker_pair,
):
    """Repeating the same paired configuration should reproduce it."""
    repeated = run_paired_credit_banking_comparison(
        seed=11,
        learner_factory=make_one_sided_optgtm,
        baseline_strategy_factory=TruthfulArm,
        attack_strategy_factory=lambda: CreditBankerArm(
            a_minus=0.20,
            a_plus=0.30,
            T0=100,
        ),
        attacker=0,
        env_kwargs={
            "T": 300,
        },
        summary_only=False,
    )

    for condition in ("baseline", "attack"):
        original_log = truthful_vs_banker_pair[
            condition
        ]["log"]

        repeated_log = repeated[
            condition
        ]["log"]

        for key in (
            "arm",
            "true_mean",
            "reward",
            "regret",
            "cum_regret",
            "ucb",
        ):
            assert np.array_equal(
                original_log[key],
                repeated_log[key],
            ), (
                f"Repeated {condition} condition produced "
                f"different values for log['{key}']."
            )

    assert (
        truthful_vs_banker_pair["paired_summary"]
        == repeated["paired_summary"]
    )
