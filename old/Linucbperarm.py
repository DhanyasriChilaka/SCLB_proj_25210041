"""Per-arm LinUCB (independent estimators, OptGTM-style).

Each arm i keeps its own ridge estimator:

    V_i = lam * I + sum of x x^T over rounds where i was chosen,
    b_i = sum of r * x over those rounds,
    theta_hat_i = V_i^{-1} b_i   (computed via solve, never an explicit inverse)

where x is the REPORTED context of arm i in those rounds — the learner never
sees true contexts. Each round:

    UCB_{t,i} = <theta_hat_i, x_{t,i}> + alpha * sqrt(x_{t,i}^T V_i^{-1} x_{t,i}),

pick argmax, observe the true reward from the environment, update ONLY the
chosen arm with (its reported context, observed reward). An arm's reports
therefore corrupt only the learner's model of that arm.

Logging per round: round, chosen arm, true mean reward of the chosen arm,
observed reward, instantaneous pseudo-regret, reported context of the chosen
arm, and the full UCB vector.
"""

import numpy as np
# class PerArmLinUCB:
#     def __init__(self, K, d, alpha=1.0, lam=1.0):
#         if alpha < 0 or lam <= 0:
#             raise ValueError("alpha must be >= 0 and lam > 0.")
#         self.K, self.d, self.alpha, self.lam = K, d, alpha, lam
#         self.V = np.stack([lam * np.eye(d) for _ in range(K)])  # (K, d, d)
#         self.b = np.zeros((K, d))
#         self.pulls = np.zeros(K, dtype=int)

#     def theta_hat(self, i):
#         return np.linalg.solve(self.V[i], self.b[i])

#     def ucbs(self, reported):
#         """UCB for each arm given reported contexts, shape (K,)."""
#         reported = np.asarray(reported, dtype=float)
#         out = np.empty(self.K)
#         for i in range(self.K):
#             x = reported[i]
#             # Solve once, reuse for both the mean and the width:
#             # V^{-1} x gives theta_hat . x via b, and x^T V^{-1} x directly.
#             Vinv_x = np.linalg.solve(self.V[i], x)
#             mean = self.b[i] @ Vinv_x          # == theta_hat_i . x
#             width = np.sqrt(max(x @ Vinv_x, 0.0))
#             out[i] = mean + self.alpha * width
#         return out

#     def select(self, reported):
#         u = self.ucbs(reported)
#         # Deterministic tie-break: lowest index among maxima.
#         return int(np.argmax(u)), u

#     def update(self, arm, x_reported, reward):
#         x = np.asarray(x_reported, dtype=float)
#         self.V[arm] += np.outer(x, x)
#         self.b[arm] += reward * x
#         self.pulls[arm] += 1


# def run(env, learner, strategies, log_contexts=False):
#     """One full episode. `strategies[i]` produces arm i's report each round.

#     Returns a dict of per-round arrays. Reported contexts of the chosen arm
#     are logged only when log_contexts=True (a (T, d) array; off by default
#     to keep logs small).
#     """
#     T, K, d = env.T, env.K, env.d
#     log = {
#         "round": np.arange(T),
#         "arm": np.zeros(T, dtype=int),
#         "true_mean": np.zeros(T),
#         "reward": np.zeros(T),
#         "regret": np.zeros(T),
#         "ucb": np.zeros((T, K)),
#     }
#     if log_contexts:
#         log["reported"] = np.zeros((T, d))

#     while not env.done:
#         t = env.t                      # save before stepping
#         X_true = env.contexts()
#         reported = np.array([strategies[i].report(t, X_true[i], env.theta)
#                              for i in range(K)])
#         arm, u = learner.select(reported)
#         log["arm"][t] = arm
#         log["ucb"][t] = u
#         log["true_mean"][t] = env.expected_rewards(t)[arm]
#         log["regret"][t] = env.regret(arm, t)
#         if log_contexts:
#             log["reported"][t] = reported[arm]
#         r = env.step(arm)
#         log["reward"][t] = r
#         learner.update(arm, reported[arm], r)

#     log["cum_regret"] = np.cumsum(log["regret"])
#     return log


if __name__ == "__main__":
    from Environment import Environment
    from Strategies import TruthfulArm

    env = Environment(seed=0)
    learner = PerArmLinUCB(K=env.K, d=env.d, alpha=1.0)
    strategies = [TruthfulArm() for _ in range(env.K)]
    log = run(env, learner, strategies)

    cum = log["cum_regret"]
    T = env.T

    # Consistency of the log.
    assert np.all(log["regret"] >= 0)
    assert np.allclose(log["reward"] - log["true_mean"], env._noise)
    assert learner.pulls.sum() == T

    # Sublinearity: per-round regret in the last quarter is much smaller
    # than in the first quarter, and the second half adds a small share of
    # the total (a straight line would add half).
    q = T // 4
    early = log["regret"][:q].mean()
    late = log["regret"][-q:].mean()
    second_half_share = (cum[-1] - cum[T // 2]) / cum[-1]
    assert late < 0.35 * early, (early, late)
    assert second_half_share < 0.35, second_half_share

    # The learner concentrates on the best arm (per-arm estimators still
    # spend a noticeable share of pulls exploring the others).
    assert learner.pulls.argmax() == 0
    assert learner.pulls[0] / T > 0.4, learner.pulls

    print(f"Final cumulative regret: {cum[-1]:.1f}")
    print(f"Mean per-round regret, first quarter: {early:.4f}, "
          f"last quarter: {late:.4f}")
    print(f"Share of regret added in the second half: {second_half_share:.1%}")
    print(f"Pulls per arm: {learner.pulls.tolist()}")
    print("All checks passed.")