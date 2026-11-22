import numpy as np
import pytest

from src.metrics import (
    count_false_eliminations,
    extract_switch_trace_metrics,
    get_elimination_round,
    get_phase_information,
    get_arm_rounds,
    get_arm_pull_count,
    get_value_at_round,
    build_condition_summary,
    build_paired_summary,
)
from src.strategies import CreditBankerArm


def test_get_arm_rounds():
    log = {
        "arm": np.array([0, 2, 0, 1, 0, 2]),
    }

    rounds = get_arm_rounds(
        log=log,
        arm=0,
    )

    assert np.array_equal(
        rounds,
        np.array([0, 2, 4]),
    )


def test_get_arm_pull_count():
    log = {
        "arm": np.array([0, 2, 0, 1, 0, 2]),
    }

    assert get_arm_pull_count(
        log=log,
        arm=0,
    ) == 3


def test_get_value_at_round():
    arm_rounds = np.array([2, 5, 9])
    values = np.array([10.0, 20.0, 30.0])

    assert get_value_at_round(
        values=values,
        arm_rounds=arm_rounds,
        round_index=0,
    ) == 0.0

    assert get_value_at_round(
        values=values,
        arm_rounds=arm_rounds,
        round_index=5,
    ) == 10.0

    assert get_value_at_round(
        values=values,
        arm_rounds=arm_rounds,
        round_index=6,
    ) == 20.0


def test_build_paired_summary():
    baseline = {
        "seed": 0,
        "algorithm": "OneSidedOptGTM",
        "radius_scale": 1.0,
        "attacker": 0,
        "strategy": "truthful",
        "strategy_parameters": {},
        "attacker_pulls": 100,
        "cumulative_regret": 20.0,
    }

    attack = {
        "seed": 0,
        "algorithm": "OneSidedOptGTM",
        "radius_scale": 1.0,
        "attacker": 0,
        "strategy": "credit_banker",
        "strategy_parameters": {
            "a_minus": 0.2,
            "a_plus": 0.3,
            "T0": 1000,
        },
        "attacker_pulls": 135,
        "cumulative_regret": 28.0,
    }

    result = build_paired_summary(
        baseline_summary=baseline,
        attack_summary=attack,
    )

    assert result["attack_gain"] == 35
    assert result["regret_increase"] == 8.0


def test_build_paired_summary_rejects_mismatched_pair():
    baseline = {
        "seed": 0,
        "algorithm": "OneSidedOptGTM",
        "radius_scale": 1.0,
        "attacker": 0,
        "strategy": "truthful",
        "strategy_parameters": {},
        "attacker_pulls": 10,
        "cumulative_regret": 1.0,
    }
    attack = dict(baseline)
    attack["seed"] = 1

    with pytest.raises(ValueError, match="seed"):
        build_paired_summary(baseline, attack)


def test_get_value_at_round_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        get_value_at_round(
            values=np.array([1.0]),
            arm_rounds=np.array([1, 2]),
            round_index=2,
        )


def test_false_eliminations_can_include_or_exclude_designated_arm():
    class Learner:
        active = np.array([False, True, False])

    learner = Learner()

    assert count_false_eliminations(learner) == 2
    assert count_false_eliminations(learner, excluded_arm=0) == 1


def test_incomplete_pull_banking_reports_last_realized_banking_round():
    class PullBanker:
        bank_pulls = 5

    information = get_phase_information(
        strategy=PullBanker(),
        arm_rounds=np.array([3, 9]),
    )

    assert information["banking_pulls_realized"] == 2
    assert information["last_banking_round"] == 9
    assert information["switch_round"] is None


def test_elimination_round_rejects_inconsistent_pull_count():
    class Learner:
        eliminated_at = [3]

    with pytest.raises(ValueError, match="inconsistent"):
        get_elimination_round(
            learner=Learner(),
            arm=0,
            arm_rounds=np.array([2, 8]),
        )


def _paired_summary_stub(strategy, pulls, phase_type=None):
    return {
        "seed": 0,
        "algorithm": "BidirectionalOptGTM",
        "radius_scale": 1.0,
        "attacker": 1,
        "strategy": strategy,
        "strategy_parameters": {},
        "attacker_pulls": pulls,
        "cumulative_regret": 10.0,
        "phase_type": phase_type,
    }


def test_time_banking_decomposition_matches_total_gain():
    delayed = _paired_summary_stub("credit_banker", 100, "time")
    delayed.update({
        "switch_round": 1000,
        "banking_pulls_realized": 30,
        "spending_pulls_realized": 70,
    })
    banker = _paired_summary_stub("credit_banker", 110, "time")
    banker.update({
        "switch_round": 1000,
        "banking_pulls_realized": 20,
        "spending_pulls_realized": 90,
    })

    result = build_paired_summary(delayed, banker)

    assert result["banking_cost"] == 10
    assert result["spending_phase_gain"] == 20
    assert result["banking_specific_gain"] == 10
    assert result["phase_identity_holds"]


def test_pull_banking_uses_condition_specific_completion_rounds():
    delayed = _paired_summary_stub("pull_count_credit_banker", 100, "pull")
    delayed.update({
        "bank_pulls_target": 30,
        "reached_bank_pulls": True,
        "bank_completion_round": 90,
        "post_banking_pulls": 70,
    })
    banker = _paired_summary_stub("pull_count_credit_banker", 110, "pull")
    banker.update({
        "bank_pulls_target": 30,
        "reached_bank_pulls": True,
        "bank_completion_round": 125,
        "post_banking_pulls": 80,
    })

    result = build_paired_summary(delayed, banker)

    assert not result["pair_censored"]
    assert result["banking_delay"] == 35
    assert result["post_banking_selection_gain"] == 10


def test_pull_banking_marks_incomplete_pair_as_censored():
    delayed = _paired_summary_stub("pull_count_credit_banker", 20, "pull")
    banker = _paired_summary_stub("pull_count_credit_banker", 18, "pull")
    for summary in (delayed, banker):
        summary.update({
            "bank_pulls_target": 30,
            "reached_bank_pulls": False,
            "bank_completion_round": None,
            "post_banking_pulls": None,
        })

    result = build_paired_summary(delayed, banker)

    assert result["pair_censored"]
    assert result["banking_delay"] is None
    assert result["post_banking_selection_gain"] is None


def test_extract_time_switch_trace_uses_last_pull_before_t0():
    class TimeBanker:
        T0 = 10

    traces = {
        "one_sided_slack": np.array([1.0, 2.0, 3.0]),
        "reverse_slack": np.array([-1.0, -2.0, -3.0]),
        "reward_boundary": np.array([0.5, 0.7, 0.9]),
        "credit_cap": np.array([2.0, 2.5, 3.0]),
    }
    result = extract_switch_trace_metrics(
        traces,
        TimeBanker(),
        arm_rounds=np.array([2, 7, 12]),
    )

    assert result["switch_trace_pull"] == 2
    assert result["switch_trace_round"] == 7
    assert result["one_sided_slack_at_switch"] == 2.0


def _time_elimination_summary(t0, selected_rounds, elimination_pull):
    class Learner:
        radius_scale = 1.0
        active = np.array([False, True])
        eliminated_at = [elimination_pull, None]
        elimination_side = ["under", None]

    horizon = max(selected_rounds) + 2
    arm_log = np.ones(horizon, dtype=int)
    arm_log[selected_rounds] = 0
    log = {
        "arm": arm_log,
        "cum_regret": np.arange(horizon, dtype=float),
        "stopped_at": None,
    }
    return build_condition_summary(
        seed=0,
        learner=Learner(),
        strategy=CreditBankerArm(
            a_minus=0.10,
            a_plus=0.15,
            T0=t0,
        ),
        log=log,
        attacker=0,
    )


def test_time_elimination_during_banking_is_not_a_negative_delay():
    summary = _time_elimination_summary(
        t0=10,
        selected_rounds=np.array([1, 5]),
        elimination_pull=2,
    )

    assert summary["eliminated_during_banking"]
    assert summary["detection_delay_global_signed"] == -5
    assert summary["detection_delay_global"] is None
    assert summary["post_switch_detection_delay_rounds"] is None
    assert summary["post_switch_detection_delay_pulls"] is None


def test_time_post_switch_delay_is_reported_in_rounds_and_pulls():
    summary = _time_elimination_summary(
        t0=3,
        selected_rounds=np.array([0, 2, 6, 8]),
        elimination_pull=4,
    )

    assert not summary["eliminated_during_banking"]
    assert summary["post_switch_detection_delay_rounds"] == 5
    assert summary["post_switch_detection_delay_pulls"] == 2
    
