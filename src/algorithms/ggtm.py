"""GGTM (Buening et al., NeurIPS 2024, Mechanism 1): theta* known.

Greedy selection among active arms by reported reward <theta*, x_{t,i}>,
plus a grim trigger: after pulling arm i (its m-th pull), eliminate it
permanently if its total claimed reward exceeds the optimistic estimate of
its total observed reward,

    sum_{l <= t: i_l = i} <theta*, x_{l,i}>  >  rhat_{t,i} + 2*sigma*sqrt(m log T),

(paper's Eq. 1; the 2*sqrt(m log T) Hoeffding width scaled by sigma, since
the paper's constants assume unit-scale sub-Gaussian noise). Claims are
evaluated with the KNOWN theta*, so unlike OptGTM there is no estimator
that could calibrate a constant lie away: GGTM detects constant
over-reporting directly.

If the active set empties (Mechanism 1, lines 8-9: stop playing, zero
reward), select() raises; with valid widths this is unreachable under
truthful arms.

The learner-side use of theta* here is legitimate: GGTM is the paper's
warm-up mechanism for the theta*-known setting (Section 4). The arms'
strategies receive theta_star through the usual report() interface either
way.
"""
import numpy as np
from . import AllArmsEliminated
class GGTM:
    def __init__(self, K, theta_star, T, sigma=0.1):
        if T < 2:
            raise ValueError("T must be at least 2.")
        self.K, self.T, self.sigma = K, T, sigma
        theta = np.asarray(theta_star, dtype=float)
        self.theta = theta / np.linalg.norm(theta)
        self.active = np.ones(K, dtype=bool)
        self.pulls = np.zeros(K, dtype=int)
        self.sum_claim = np.zeros(K)   # sum of <theta*, reported x>
        self.sum_r = np.zeros(K)       # sum of observed rewards
        self.eliminated_at = [None] * K

    def reward_slack(self, m):
        """Hoeffding width on the summed rewards (paper's Eq. 1)."""
        return 2.0 * self.sigma * np.sqrt(m * np.log(self.T))

    def ucbs(self, reported):
        """Greedy scores: the reported rewards <theta*, x_{t,i}>.

        Named ucbs for drop-in compatibility with linucb.run's logging;
        GGTM is greedy, there is no exploration bonus (Mechanism 1, line 4).
        """
        return np.asarray(reported, dtype=float) @ self.theta

    def select(self, reported):
        if not self.active.any():
            raise AllArmsEliminated(
                "All arms eliminated (Mechanism 1, line 8): "
                "stop playing; zero reward henceforth."
            )
        u = self.ucbs(reported)
        return int(np.argmax(np.where(self.active, u, -np.inf))), u

    def update(self, arm, x_reported, reward):
        self.sum_claim[arm] += float(np.asarray(x_reported, float) @ self.theta)
        self.sum_r[arm] += reward
        self.pulls[arm] += 1
        m = self.pulls[arm]
        if self.active[arm] and \
                self.sum_claim[arm] > self.sum_r[arm] + self.reward_slack(m):
            self.active[arm] = False
            self.eliminated_at[arm] = m


# if __name__ == "__main__":
#     from Environment import Environment
#     from Strategies import TruthfulArm, OverReporterArm
#     from Linucbperarm import run

#     # Check 1 (paper Lemma D.2): truthful arms are never eliminated, and
#     # with theta* known the greedy rule picks the true best arm every
#     # round, so regret is exactly zero.
#     false_elims, max_reg = 0, 0.0
#     for seed in range(20):
#         env = Environment(seed=seed)
#         learner = GGTM(env.K, env.theta, T=env.T)
#         log = run(env, learner, [TruthfulArm() for _ in range(env.K)])
#         false_elims += int((~learner.active).sum())
#         max_reg = max(max_reg, log["cum_regret"][-1])
#     assert false_elims == 0, false_elims
#     assert max_reg == 0.0, max_reg
#     print("Check 1: zero false eliminations and zero regret "
#           "across 20 truthful seeds.")

#     # Check 2: the constant over-reporter IS eliminated by GGTM, and fast.
#     # With theta* known, claims are anchored to the truth and cannot be
#     # calibrated away: claims exceed rewards by ~delta per pull, while the
#     # slack grows only like sqrt(m). (Contrast with OptGTM, where the same
#     # lie is realizable and undetectable.)
#     for seed in range(5):
#         env = Environment(seed=seed)
#         strats = [TruthfulArm() for _ in range(env.K)]
#         strats[0] = OverReporterArm(delta=0.3)
#         learner = GGTM(env.K, env.theta, T=env.T)
#         log = run(env, learner, strats)
#         m = learner.eliminated_at[0]
#         assert m is not None and m <= 20, m
#         if seed == 0:
#             print(f"Over-reporter (+0.3): eliminated at pull {m}, "
#                   f"cum regret {log['cum_regret'][-1]:.1f} "
#                   f"(best arm lost; cost of the grim trigger, Sec. 4.1)")
#     print("All checks passed.")
