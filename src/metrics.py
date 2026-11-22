
"""Metrics and result-extraction utilities for bandit experiments."""

import numpy as np


def get_elimination_pull(learner, arm):
    """Return an arm's elimination pull count, or None."""
    eliminated_at = getattr(
        learner,
        "eliminated_at",
        None,
    )

    if eliminated_at is None:
        return None

    value = eliminated_at[arm]

    if value is None:
        return None

    return int(value)


def get_elimination_round(
    learner,
    arm,
    arm_rounds,
):
    """Convert an elimination pull count to a global round number.

    Parameters
    ----------
    learner:
        Learner that may contain `eliminated_at`.

    arm:
        Arm index.

    arm_rounds:
        Global rounds on which the arm was selected.
    """
    elimination_pull = get_elimination_pull(
        learner=learner,
        arm=arm,
    )

    if elimination_pull is None:
        return None

    # Pull counts start at one; array positions start at zero.
    index = elimination_pull - 1

    if not 0 <= index < len(arm_rounds):
        raise ValueError(
            "Elimination pull is inconsistent with the arm-selection log: "
            f"pull={elimination_pull}, selections={len(arm_rounds)}."
        )

    return int(arm_rounds[index])


def get_elimination_side(learner, arm):
    """Return 'over', 'under', or None."""
    sides = getattr(
        learner,
        "elimination_side",
        None,
    )

    if sides is None:
        return None

    return sides[arm]


def count_false_eliminations(
    learner,
    excluded_arm=None,
):
    """Count eliminated truthful arms, optionally excluding one arm.

    Returns None for algorithms without an active-arm mechanism.
    """
    active = getattr(
        learner,
        "active",
        None,
    )

    if active is None:
        return None

    active = np.asarray(
        active,
        dtype=bool,
    )

    if excluded_arm is not None:
        active = np.delete(active, excluded_arm)

    return int((~active).sum())


def get_arm_rounds(log, arm):
    """Return global rounds on which `arm` was selected."""
    return np.flatnonzero(
        np.asarray(log["arm"]) == arm
    )


def get_arm_pull_count(log, arm):
    """Return the total number of selections received by an arm."""
    return int(
        np.count_nonzero(
            np.asarray(log["arm"]) == arm
        )
    )


def extract_arm_traces(
    learner,
    arm,
):
    """Extract the trigger quantities used by an OptGTM learner.

    OptGTM learners store one `(LCB, UCB, reward)` tuple for every
    selection of an arm. For a windowed learner, the primary sum/slack
    fields use the last W pulls, matching its actual trigger; cumulative
    sums are also returned under explicit `cumulative_*` keys. Algorithms
    without `records` return None. `credit_cap` is defined only for the
    still-active portion of a bidirectional learner's trace.
    """
    all_records = getattr(
        learner,
        "records",
        None,
    )

    if all_records is None:
        return None

    records = np.asarray(all_records[arm], dtype=float).reshape(-1, 3)

    lcb = records[:, 0]
    ucb = records[:, 1]
    reward = records[:, 2]

    cumulative_lcb_sum = np.cumsum(lcb)
    cumulative_ucb_sum = np.cumsum(ucb)
    cumulative_reward_sum = np.cumsum(reward)

    window = getattr(learner, "window", None)

    def rolling_sum(values, width):
        result = np.cumsum(values)
        if len(result) > width:
            result[width:] -= result[:-width].copy()
        return result

    if window is None:
        lcb_sum = cumulative_lcb_sum
        ucb_sum = cumulative_ucb_sum
        reward_sum = cumulative_reward_sum
        boundary_counts = np.arange(1, len(records) + 1)
    else:
        lcb_sum = rolling_sum(lcb, window)
        ucb_sum = rolling_sum(ucb, window)
        reward_sum = rolling_sum(reward, window)
        boundary_counts = np.minimum(
            np.arange(1, len(records) + 1),
            window,
        )

    # Original trigger:
    #
    # L_m > R_m + B_m
    #
    # Equivalently:
    #
    # R_m - L_m < -B_m
    one_sided_slack = reward_sum - lcb_sum

    # Added reverse trigger:
    #
    # U_m < R_m - B_m
    #
    # Equivalently:
    #
    # R_m - U_m > B_m
    reverse_slack = reward_sum - ucb_sum

    model_width = ucb_sum - lcb_sum

    if hasattr(learner, "reward_slack"):
        reward_boundary = np.asarray(
            learner.reward_slack(boundary_counts),
            dtype=float,
        )
    else:
        reward_boundary = np.full(
            len(records),
            np.nan,
        )

    credit_cap = np.full(len(records), np.nan)
    if hasattr(learner, "elimination_side"):
        # For a surviving bidirectional arm:
        # R_m - L_m <= B_m + (U_m - L_m)
        credit_cap = reward_boundary + model_width
        elimination_pull = get_elimination_pull(learner, arm)
        if elimination_pull is not None:
            credit_cap[elimination_pull - 1:] = np.nan

    return {
        "records": records,
        "lcb": lcb,
        "ucb": ucb,
        "reward": reward,
        "lcb_sum": lcb_sum,
        "ucb_sum": ucb_sum,
        "reward_sum": reward_sum,
        "cumulative_lcb_sum": cumulative_lcb_sum,
        "cumulative_ucb_sum": cumulative_ucb_sum,
        "cumulative_reward_sum": cumulative_reward_sum,
        "one_sided_slack": one_sided_slack,
        "reverse_slack": reverse_slack,
        "model_width": model_width,
        "reward_boundary": reward_boundary,
        "credit_cap": credit_cap,
    }


def get_strategy_parameters(strategy):
    """Return experiment-relevant strategy parameters."""
    if not hasattr(strategy, "params"):
        raise TypeError(
            f"{type(strategy).__name__} must implement params()."
        )

    parameters = strategy.params()
    if not isinstance(parameters, dict):
        raise TypeError("strategy.params() must return a dictionary.")

    return {
        name: value.item() if isinstance(value, np.generic) else value
        for name, value in parameters.items()
    }


def get_phase_information(
    strategy,
    arm_rounds,
):
    """Extract banking-phase information from either banker design."""
    information = {
        "phase_type": None,
        "switch_round": None,
        "bank_pulls_target": None,
        "banking_pulls_realized": None,
        "spending_pulls_realized": None,
        "last_banking_round": None,
        "reached_bank_pulls": None,
        "bank_completion_round": None,
        "post_banking_pulls": None,
    }

    # Global-round strategy:
    # under-report for t < T0.
    if hasattr(strategy, "T0"):
        switch_round = int(strategy.T0)

        banking_rounds = arm_rounds[
            arm_rounds < switch_round
        ]

        information["phase_type"] = "time"
        information["switch_round"] = switch_round
        information["banking_pulls_realized"] = int(
            len(banking_rounds)
        )
        information["spending_pulls_realized"] = int(
            len(arm_rounds) - len(banking_rounds)
        )

        if len(banking_rounds) > 0:
            information["last_banking_round"] = int(
                banking_rounds[-1]
            )

        return information

    # Pull-count strategy:
    # under-report for the first `bank_pulls` selections.
    if hasattr(strategy, "bank_pulls"):
        bank_pulls = int(
            strategy.bank_pulls
        )

        realized = min(
            bank_pulls,
            len(arm_rounds),
        )
        reached = len(arm_rounds) >= bank_pulls
        information["phase_type"] = "pull"
        information["bank_pulls_target"] = bank_pulls
        information["banking_pulls_realized"] = realized
        information["reached_bank_pulls"] = bool(reached)

        if realized > 0:
            information["last_banking_round"] = int(
                arm_rounds[realized - 1]
            )

        if bank_pulls == 0:
            information["switch_round"] = 0
            information["bank_completion_round"] = 0
            information["post_banking_pulls"] = int(len(arm_rounds))
            information["spending_pulls_realized"] = int(len(arm_rounds))
        elif realized == bank_pulls:
            completion_round = int(arm_rounds[realized - 1])
            post_banking_pulls = int(len(arm_rounds) - bank_pulls)
            information["switch_round"] = int(
                completion_round + 1
            )
            information["bank_completion_round"] = completion_round
            information["post_banking_pulls"] = post_banking_pulls
            information["spending_pulls_realized"] = post_banking_pulls

    return information


def get_value_at_round(
    values,
    arm_rounds,
    round_index,
    default=0.0,
):
    """Return the latest arm-level trace value before a global round.

    This replaces the old `slack_at()` helper.
    """
    values = np.asarray(values)
    arm_rounds = np.asarray(arm_rounds)

    if len(values) != len(arm_rounds):
        raise ValueError(
            "values and arm_rounds must have the same length."
        )

    position = int(
        np.searchsorted(
            arm_rounds,
            round_index,
        )
    )

    if position == 0:
        return float(default)

    return float(values[position - 1])


def build_condition_summary(
    seed,
    learner,
    strategy,
    log,
    attacker,
):
    """Build summary metrics for one experimental condition."""
    attacker_rounds = get_arm_rounds(
        log=log,
        arm=attacker,
    )

    stopped_at = log.get("stopped_at")

    summary = {
        "seed": int(seed),
        "algorithm": type(learner).__name__,
        "radius_scale": (
            float(learner.radius_scale)
            if hasattr(learner, "radius_scale")
            else None
        ),
        "strategy": getattr(
            strategy,
            "name",
            type(strategy).__name__,
        ),
        "strategy_parameters": get_strategy_parameters(
            strategy
        ),
        "attacker": int(attacker),
        "attacker_pulls": int(
            len(attacker_rounds)
        ),
        "cumulative_regret": float(
            log["cum_regret"][-1]
        ),
        "stopped_at": (
            None if stopped_at is None else int(stopped_at)
        ),
        "complete_stop": stopped_at is not None,
        "elimination_pull": get_elimination_pull(
            learner=learner,
            arm=attacker,
        ),
        "elimination_round": get_elimination_round(
            learner=learner,
            arm=attacker,
            arm_rounds=attacker_rounds,
        ),
        "elimination_side": get_elimination_side(
            learner=learner,
            arm=attacker,
        ),
        "false_eliminations": count_false_eliminations(
            learner=learner,
            excluded_arm=(
                attacker
                if not getattr(strategy, "is_truthful", False)
                else None
            ),
        ),
    }

    summary.update(
        get_phase_information(
            strategy=strategy,
            arm_rounds=attacker_rounds,
        )
    )

    summary["attacker_eliminated"] = (
        summary["elimination_pull"] is not None
    )
    summary["any_false_elimination"] = (
        None
        if summary["false_eliminations"] is None
        else summary["false_eliminations"] > 0
    )

    summary["eliminated_during_banking"] = None
    summary["detection_delay_global_signed"] = None
    summary["detection_delay_global"] = None
    summary["post_switch_detection_delay_rounds"] = None
    summary["post_switch_detection_delay_pulls"] = None
    if (
        summary["phase_type"] == "time"
        and summary["elimination_round"] is not None
    ):
        signed_delay = int(
            summary["elimination_round"] - summary["switch_round"]
        )
        during_banking = signed_delay < 0
        summary["eliminated_during_banking"] = during_banking
        summary["detection_delay_global_signed"] = signed_delay
        if not during_banking:
            summary["detection_delay_global"] = signed_delay
            summary["post_switch_detection_delay_rounds"] = signed_delay
            summary["post_switch_detection_delay_pulls"] = int(
                summary["elimination_pull"]
                - summary["banking_pulls_realized"]
            )
    elif summary["phase_type"] == "time":
        summary["eliminated_during_banking"] = False

    summary["detection_delay_pull_signed"] = None
    summary["detection_delay_pull"] = None
    if (
        summary["phase_type"] == "pull"
        and summary["elimination_pull"] is not None
    ):
        signed_delay = int(
            summary["elimination_pull"]
            - summary["bank_pulls_target"]
        )
        during_banking = signed_delay <= 0
        summary["eliminated_during_banking"] = during_banking
        summary["detection_delay_pull_signed"] = signed_delay
        if not during_banking:
            summary["detection_delay_pull"] = signed_delay
    elif summary["phase_type"] == "pull":
        summary["eliminated_during_banking"] = False

    return summary


def build_paired_summary(
    baseline_summary,
    attack_summary,
    baseline_arm_rounds=None,
    attack_arm_rounds=None,
):
    """Calculate paired differences between attack and baseline."""
    for key in ("seed", "algorithm", "attacker", "radius_scale"):
        if baseline_summary[key] != attack_summary[key]:
            raise ValueError(
                f"Paired conditions disagree on {key}: "
                f"{baseline_summary[key]!r} != {attack_summary[key]!r}."
            )

    result = {
        "seed": attack_summary["seed"],
        "algorithm": attack_summary["algorithm"],
        "radius_scale": attack_summary["radius_scale"],
        "attacker": attack_summary["attacker"],
        "baseline_strategy": baseline_summary["strategy"],
        "baseline_parameters": baseline_summary[
            "strategy_parameters"
        ],
        "attack_strategy": attack_summary["strategy"],
        "attack_parameters": attack_summary[
            "strategy_parameters"
        ],
        "baseline_attacker_pulls": baseline_summary[
            "attacker_pulls"
        ],
        "attack_attacker_pulls": attack_summary[
            "attacker_pulls"
        ],
        "attack_gain": (
            attack_summary["attacker_pulls"]
            - baseline_summary["attacker_pulls"]
        ),
        "relative_attack_gain": (
            attack_summary["attacker_pulls"]
            - baseline_summary["attacker_pulls"]
        ) / max(baseline_summary["attacker_pulls"], 1),
        "baseline_regret": baseline_summary[
            "cumulative_regret"
        ],
        "attack_regret": attack_summary[
            "cumulative_regret"
        ],
        "regret_increase": (
            attack_summary["cumulative_regret"]
            - baseline_summary["cumulative_regret"]
        ),
    }

    for condition, summary in (
        ("baseline", baseline_summary),
        ("attack", attack_summary),
    ):
        for field in (
            "elimination_pull",
            "elimination_round",
            "elimination_side",
            "false_eliminations",
            "stopped_at",
            "attacker_eliminated",
            "complete_stop",
            "detection_delay_global",
            "detection_delay_pull",
            "detection_delay_global_signed",
            "detection_delay_pull_signed",
            "post_switch_detection_delay_rounds",
            "post_switch_detection_delay_pulls",
            "eliminated_during_banking",
        ):
            result[f"{condition}_{field}"] = summary.get(field)

    baseline_phase = baseline_summary.get("phase_type")
    attack_phase = attack_summary.get("phase_type")

    if baseline_phase == attack_phase == "time":
        baseline_switch = baseline_summary.get("switch_round")
        attack_switch = attack_summary.get("switch_round")
        if baseline_switch != attack_switch:
            raise ValueError(
                "Time-based paired conditions must use the same T0."
            )

        baseline_before = baseline_summary["banking_pulls_realized"]
        attack_before = attack_summary["banking_pulls_realized"]
        baseline_after = baseline_summary["spending_pulls_realized"]
        attack_after = attack_summary["spending_pulls_realized"]

        banking_cost = baseline_before - attack_before
        spending_gain = attack_after - baseline_after
        banking_specific_gain = result["attack_gain"]
        if banking_specific_gain != spending_gain - banking_cost:
            raise AssertionError(
                "Time-phase decomposition does not equal total paired gain."
            )

        result.update({
            "phase_type": "time",
            "phase_switch_round": attack_switch,
            "baseline_banking_phase_pulls": int(baseline_before),
            "attack_banking_phase_pulls": int(attack_before),
            "banking_phase_gain": int(attack_before - baseline_before),
            "banking_cost": int(banking_cost),
            "baseline_spending_phase_pulls": int(baseline_after),
            "attack_spending_phase_pulls": int(attack_after),
            "spending_phase_gain": int(spending_gain),
            "banking_specific_gain": int(banking_specific_gain),
            "phase_identity_holds": True,
            "pair_censored": False,
        })

    elif baseline_phase == attack_phase == "pull":
        baseline_target = baseline_summary.get("bank_pulls_target")
        attack_target = attack_summary.get("bank_pulls_target")
        if baseline_target != attack_target:
            raise ValueError(
                "Pull-count paired conditions must use the same bank_pulls."
            )

        censored = not (
            baseline_summary.get("reached_bank_pulls", False)
            and attack_summary.get("reached_bank_pulls", False)
        )
        result.update({
            "phase_type": "pull",
            "bank_pulls_target": attack_target,
            "banking_specific_gain": int(result["attack_gain"]),
            "pair_censored": bool(censored),
            "baseline_bank_completion_round": baseline_summary.get(
                "bank_completion_round"
            ),
            "attack_bank_completion_round": attack_summary.get(
                "bank_completion_round"
            ),
            "banking_delay": None,
            "post_banking_selection_gain": None,
        })
        if not censored:
            result["banking_delay"] = int(
                attack_summary["bank_completion_round"]
                - baseline_summary["bank_completion_round"]
            )
            result["post_banking_selection_gain"] = int(
                attack_summary["post_banking_pulls"]
                - baseline_summary["post_banking_pulls"]
            )

    elif (
        attack_phase == "time"
        and baseline_arm_rounds is not None
        and attack_arm_rounds is not None
    ):
        # Overall attack-versus-truthful split at the attack's global switch.
        baseline_arm_rounds = np.asarray(baseline_arm_rounds)
        attack_arm_rounds = np.asarray(attack_arm_rounds)
        switch_round = attack_summary["switch_round"]
        baseline_before = int(
            np.count_nonzero(baseline_arm_rounds < switch_round)
        )
        attack_before = int(
            np.count_nonzero(attack_arm_rounds < switch_round)
        )
        result.update({
            "phase_type": "time",
            "phase_switch_round": switch_round,
            "baseline_banking_phase_pulls": baseline_before,
            "attack_banking_phase_pulls": attack_before,
            "banking_phase_gain": attack_before - baseline_before,
            "baseline_spending_phase_pulls": (
                len(baseline_arm_rounds) - baseline_before
            ),
            "attack_spending_phase_pulls": (
                len(attack_arm_rounds) - attack_before
            ),
            "spending_phase_gain": (
                len(attack_arm_rounds) - attack_before
                - (len(baseline_arm_rounds) - baseline_before)
            ),
        })

    return result


def extract_switch_trace_metrics(traces, strategy, arm_rounds):
    """Return attacker trace values at the end of its banking phase.

    Time-based strategies use the latest attacker pull strictly before T0.
    Pull-count strategies use the configured m-th attacker pull. Missing or
    censored switch points produce None values rather than artificial zeros.
    """
    names = (
        "one_sided_slack",
        "reverse_slack",
        "reward_boundary",
        "credit_cap",
    )
    result = {
        "switch_trace_pull": None,
        "switch_trace_round": None,
        **{f"{name}_at_switch": None for name in names},
    }
    if traces is None:
        return result

    arm_rounds = np.asarray(arm_rounds)
    if hasattr(strategy, "T0"):
        position = int(np.searchsorted(arm_rounds, int(strategy.T0))) - 1
    elif hasattr(strategy, "bank_pulls"):
        position = int(strategy.bank_pulls) - 1
        if position < 0:
            position = None
    else:
        return result

    if position is None or not 0 <= position < len(arm_rounds):
        return result

    result["switch_trace_pull"] = int(position + 1)
    result["switch_trace_round"] = int(arm_rounds[position])
    for name in names:
        value = np.asarray(traces[name])[position]
        result[f"{name}_at_switch"] = (
            None if np.isnan(value) else float(value)
        )
    return result
