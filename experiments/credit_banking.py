"""Credit-banking experiment orchestration."""

from src.environment import Environment
from src.strategies import TruthfulArm
from src.simulation import run_episode
from src.metrics import (
    build_condition_summary,
    build_paired_summary,
    extract_arm_traces,
    extract_switch_trace_metrics,
    get_arm_rounds,
)


def run_credit_banking_condition(
    seed,
    learner_factory,
    strategy_factory,
    attacker=0,
    env_kwargs=None,
    log_contexts=False,
    summary_only=True,
    collect_traces=False,
):
    """Run one algorithm-strategy condition.

    By default, return only the lightweight summary for parameter sweeps.
    Set ``summary_only=False`` to retain the environment, learner and log.
    Set ``collect_traces=True`` only for representative runs that will be
    plotted or inspected in detail.
    """
    if summary_only and collect_traces:
        raise ValueError(
            "collect_traces=True requires summary_only=False."
        )

    if env_kwargs is None:
        env_kwargs = {}
    else:
        env_kwargs = dict(env_kwargs)

    if "seed" in env_kwargs:
        raise ValueError(
            "Pass seed through the seed argument, "
            "not through env_kwargs."
        )

    env = Environment(
        seed=seed,
        **env_kwargs,
    )

    if not 0 <= attacker < env.K:
        raise ValueError(
            f"attacker must be in [0, {env.K - 1}], "
            f"but received {attacker}."
        )

    # Every condition receives a fresh learner and strategy.
    learner = learner_factory(env)
    attacker_strategy = strategy_factory()

    strategies = [
        TruthfulArm()
        for _ in range(env.K)
    ]

    strategies[attacker] = attacker_strategy

    log = run_episode(
        env=env,
        learner=learner,
        strategies=strategies,
        log_contexts=log_contexts,
    )

    attacker_rounds = get_arm_rounds(
        log=log,
        arm=attacker,
    )

    traces = None
    if collect_traces:
        traces = extract_arm_traces(
            learner=learner,
            arm=attacker,
        )

    summary = build_condition_summary(
        seed=seed,
        learner=learner,
        strategy=attacker_strategy,
        log=log,
        attacker=attacker,
    )

    if collect_traces:
        summary.update(
            extract_switch_trace_metrics(
                traces=traces,
                strategy=attacker_strategy,
                arm_rounds=attacker_rounds,
            )
        )

    if summary_only:
        return summary

    return {
        "env": env,
        "learner": learner,
        "strategy": attacker_strategy,
        "strategies": strategies,
        "log": log,
        "attacker_rounds": attacker_rounds,
        "traces": traces,
        "summary": summary,
    }


def run_paired_credit_banking_comparison(
    seed,
    learner_factory,
    baseline_strategy_factory,
    attack_strategy_factory,
    attacker=0,
    env_kwargs=None,
    log_contexts=False,
    summary_only=True,
    collect_traces=False,
):
    """Run baseline and attack conditions with common randomness.

    Summary-only mode is the default for sweeps. Use full mode, optionally
    with trace collection, for representative runs and plots.
    """
    if summary_only and collect_traces:
        raise ValueError(
            "collect_traces=True requires summary_only=False."
        )

    baseline = run_credit_banking_condition(
        seed=seed,
        learner_factory=learner_factory,
        strategy_factory=baseline_strategy_factory,
        attacker=attacker,
        env_kwargs=env_kwargs,
        log_contexts=log_contexts,
        summary_only=False,
        collect_traces=collect_traces,
    )

    attack = run_credit_banking_condition(
        seed=seed,
        learner_factory=learner_factory,
        strategy_factory=attack_strategy_factory,
        attacker=attacker,
        env_kwargs=env_kwargs,
        log_contexts=log_contexts,
        summary_only=False,
        collect_traces=collect_traces,
    )

    paired_summary = build_paired_summary(
        baseline_summary=baseline["summary"],
        attack_summary=attack["summary"],
        baseline_arm_rounds=baseline["attacker_rounds"],
        attack_arm_rounds=attack["attacker_rounds"],
    )

    if summary_only:
        return {
            "baseline": baseline["summary"],
            "attack": attack["summary"],
            "paired_summary": paired_summary,
        }

    return {
        "baseline": baseline,
        "attack": attack,
        "paired_summary": paired_summary,
    }
