"""Simulation loop for strategic contextual-bandit experiments."""

import numpy as np
from .algorithms import AllArmsEliminated


def run_episode(
    env,
    learner,
    strategies,
    log_contexts=False,
):
    """Run one complete bandit episode.

    Parameters
    ----------
    env:
        Bandit environment.

    learner:
        Learning algorithm implementing:
            select(reported_contexts)
            update(arm, x_reported, reward)

    strategies:
        Sequence of K reporting strategies. Each strategy implements:
            report(t, x_true, theta_star)

        Stateful strategies may optionally implement:
            reset()
            on_selected(round_index, reward, reported_context)

    log_contexts:
        If True, store the selected arm's reported context each round.

    Returns
    -------
    dict
        Per-round logs and cumulative pseudo-regret.
    """
    T = env.T
    K = env.K
    d = env.d

    if len(strategies) != K:
        raise ValueError(
            f"Expected {K} strategies, "
            f"but received {len(strategies)}."
        )

    # This function expects a fresh or explicitly reset environment.
    if env.t != 0:
        raise ValueError(
            "run_episode expects env.t == 0. "
            "Create a new environment or call env.reset() first."
        )

    # Reset stateful strategies, such as pull-count credit bankers.
    for strategy in strategies:
        if hasattr(strategy, "reset"):
            strategy.reset()

    log = {
        "round": np.arange(T),
        "arm": np.full(T, -1, dtype=int),
        "true_mean": np.zeros(T),
        "reward": np.zeros(T),
        "regret": np.zeros(T),
        "ucb": np.zeros((T, K)),
        "stopped_at": None,
    }

    if log_contexts:
        log["reported"] = np.zeros((T, d))

    while not env.done:
        # Save the round number before env.step() increments env.t.
        t = env.t

        # True contexts remain private to the corresponding arms.
        X_true = env.contexts()

        # Every arm submits one reported context.
        reported = np.array(
            [
                strategies[i].report(
                    t=t,
                    x_true=X_true[i],
                    theta_star=env.theta,
                )
                for i in range(K)
            ],
            dtype=float,
        )

        if reported.shape != (K, d):
            raise ValueError(
                "Strategies must produce reported contexts with "
                f"shape ({K}, {d}), but received {reported.shape}."
            )

        # Learner selects an arm using only reported contexts.
        try:
            arm, ucb_values = learner.select(reported)
        except AllArmsEliminated:
            # Mechanisms 1 and 2 prescribe stopping with zero reward once
            # no active arm remains. Preserve a full-horizon evaluation log.
            log["stopped_at"] = t
            log["regret"][t:] = np.array(
                [env.expected_rewards(s).max() for s in range(t, T)]
            )
            break

        # Log quantities available before observing the reward.
        log["arm"][t] = arm
        log["ucb"][t] = ucb_values
        log["true_mean"][t] = env.expected_rewards(t)[arm]
        log["regret"][t] = env.regret(arm, t)

        if log_contexts:
            log["reported"][t] = reported[arm]

        # Reward is generated from the selected arm's true context.
        reward = env.step(arm)
        log["reward"][t] = reward

        # Update the learner using the reported context and true reward.
        learner.update(
            arm=arm,
            x_reported=reported[arm],
            reward=reward,
        )

        # Inform only the arm that was actually selected.
        selected_strategy = strategies[arm]

        if hasattr(selected_strategy, "on_selected"):
            selected_strategy.on_selected(
                round_index=t,
                reward=reward,
                reported_context=reported[arm],
            )

    log["cum_regret"] = np.cumsum(
        log["regret"]
    )

    return log
# import numpy as np
# def run_episode(env, learner, strategies, log_contexts=False):
#     """One full episode. `strategies[i]` produces arm i's report each round.

#     Returns a dict of per-round arrays. Reported contexts of the chosen arm
#     are logged only when log_contexts=True (a (T, d) array; off by default
#     to keep logs small).
#     """
#     T, K, d = env.T, env.K, env.d
#     if len(strategies) != K:
#         raise ValueError(
#             f"Expected {K} strategies, got {len(strategies)}."
#         )

#     for strategy in strategies:
#         if hasattr(strategy, "reset"):
#             strategy.reset()
#     log = {
#         "round": np.arange(T),
#         "arm": np.zeros(T, dtype=int),
#         "true_mean": np.zeros(T),
#         # "reward": np.zeros(T),
#         "regret": np.zeros(T),
#         "ucb": np.zeros((T, K)),
#     }
#     if log_contexts:
#         log["reported"] = np.zeros((T, d))

#     # while not env.done:
#     #     t = env.t                      # save before stepping
#     #     X_true = env.contexts()
#     #     reported = np.array([strategies[i].report(t, X_true[i], env.theta)
#     #                          for i in range(K)])
#     #     arm, u = learner.select(reported)
#     #     log["arm"][t] = arm
#     #     log["ucb"][t] = u
#     #     log["true_mean"][t] = env.expected_rewards(t)[arm]
#     #     log["regret"][t] = env.regret(arm, t)
#     #     if log_contexts:
#     #         log["reported"][t] = reported[arm]
#     #     r = env.step(arm)
#     #     log["reward"][t] = r
#     #     learner.update(arm, reported[arm], r)
#     while not env.done:
#         t = env.t
#         X_true = env.contexts()

#         reported = np.array([
#             strategies[i].report(
#                 t=t,
#                 x_true=X_true[i],
#                 theta_star=env.theta,
#             )
#             for i in range(K)
#         ])

#         arm, u = learner.select(reported)

#         log["arm"][t] = arm
#         log["ucb"][t] = u
#         log["true_mean"][t] = env.expected_rewards(t)[arm]
#         log["regret"][t] = env.regret(arm, t)

#         if log_contexts:
#             log["reported"][t] = reported[arm]

#         r = env.step(arm)
#         log["reward"][t] = r

#         learner.update(
#             arm=arm,
#             x_reported=reported[arm],
#             reward=r,
#         )

#     # Inform only the selected arm.
#     selected_strategy = strategies[arm]

#     if hasattr(selected_strategy, "on_selected"):
#         selected_strategy.on_selected(
#             round_index=t,
#             reward=r,
#             reported_context=reported[arm],
#         )
#     log["cum_regret"] = np.cumsum(log["regret"])
#     return log
