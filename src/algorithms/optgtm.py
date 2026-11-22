"""One-sided OptGTM (Buening et al., NeurIPS 2024, Mechanism 2).

Per-arm least-squares estimators on REPORTED contexts (Eq. 3), selection by
per-arm optimistic value among active arms, and a grim-trigger elimination:
after pulling arm i (its m-th pull), eliminate it permanently if

    sum_{l <= t: i_l = i} LCB_{l,i}(x_{l,i})  >  rhat_{t,i} + 2*sigma*sqrt(m log T)

where each LCB was recorded AT PULL TIME with the then-current ("outdated")
estimator, BEFORE updating on that round's data. The paper stresses this:
recomputing with the latest theta_hat makes the trigger vacuous, because
least squares minimizes exactly the claims-vs-rewards gap (Section 5).
Widths follow the paper, adapted to noise scale sigma (the paper's
constants are written for unit-scale sub-Gaussian noise):
  - confidence radius  sqrt(beta_{t,i}),
        beta-root = sigma * sqrt(d log((1 + m/lam)/delta)) + sqrt(lam) * S,
    with S = 1, delta = 1/T^2 (paper's choice), used for BOTH the selection
    UCB and the trigger's LCB;
  - reward slack 2 * sigma * sqrt(m log T) (Hoeffding, Eq. 1)
If the active set empties (lines 8-9 of Mechanism 2: stop playing, zero
reward), select() raises; with valid widths this is unreachable under
truthful arms.
Known departures from the paper, both deliberate:
  - sigma-scaling of the widths (above);
  - none other. Empirical width tuning (e.g. a constant radius) is NOT
    enabled here; add it explicitly in experiment code if needed, and
    report it as a departure.
"""
import numpy as np
from . import AllArmsEliminated
from  .linucb_per_arm import PerArmLinUCB
class OneSidedOptGTM(PerArmLinUCB):
    def __init__(
        self,
        K,
        d,
        T,
        lam=1.0,
        sigma=0.1,
        radius_scale=1.0,
    ):
        if T < 2:
            raise ValueError("T must be at least 2.")
        if not np.isfinite(radius_scale) or radius_scale <= 0:
            raise ValueError("radius_scale must be positive and finite.")
        # alpha is unused: selection uses the paper's sqrt(beta) radius.
        super().__init__(K, d, alpha=0.0, lam=lam)
        self.T, self.sigma = T, sigma
        self.radius_scale = float(radius_scale)
        self.delta = 1.0 / T ** 2
        self.active = np.ones(K, dtype=bool)
        self.records = [[] for _ in range(K)]   # (l, u, r) per pull
        self.sum_l = np.zeros(K)
        self.sum_r = np.zeros(K)
        self.eliminated_at = [None] * K

    # --- paper's confidence machinery ---

    def beta_root(self, m):
        """sqrt(beta_{t,i}) after m pulls; S = 1, delta = 1/T^2."""
        paper_radius = (
            self.sigma
            * np.sqrt(self.d * np.log((1 + m / self.lam) / self.delta))
            + np.sqrt(self.lam) * 1.0
        )
        return self.radius_scale * paper_radius

    def reward_slack(self, m):
        """Hoeffding width on the summed rewards (paper's Eq. 1)."""
        return 2.0 * self.sigma * np.sqrt(m * np.log(self.T))

    def bounds(self, i, x):
        """[LCB, UCB] of reported x under arm i's CURRENT estimator."""
        x = np.asarray(x, float)
        Vinv_x = np.linalg.solve(self.V[i], x)
        mean = self.b[i] @ Vinv_x
        w = self.beta_root(self.pulls[i]) * np.sqrt(max(x @ Vinv_x, 0.0))
        return mean - w, mean + w

    # --- selection: optimistic value among active arms (paper line 4) ---

    def ucbs(self, reported):
        reported = np.asarray(reported, dtype=float)
        return np.array([self.bounds(i, reported[i])[1]
                         for i in range(self.K)])

    def select(self, reported):
        if not self.active.any():
            raise AllArmsEliminated(
                "All arms eliminated (Mechanism 2, line 8): "
                "stop playing; zero reward henceforth."
            )
        u = self.ucbs(reported)
        return int(np.argmax(np.where(self.active, u, -np.inf))), u

    # --- record at pull time, then update, then trigger (lines 5-7) ---

    def _record(self, arm, l, u, reward):
        self.records[arm].append((l, u, reward))
        self.sum_l[arm] += l
        self.sum_r[arm] += reward

    def _triggered(self, arm):
        m = self.pulls[arm]
        return self.sum_l[arm] > self.sum_r[arm] + self.reward_slack(m)

    def update(self, arm, x_reported, reward):
        l, u = self.bounds(arm, x_reported)   # outdated estimator, pre-update
        self._record(arm, l, u, reward)
        super().update(arm, x_reported, reward)
        if self.active[arm] and self._triggered(arm):
            self.active[arm] = False
            self.eliminated_at[arm] = self.pulls[arm]


# if __name__ == "__main__":
#     from Environment import Environment
#     from Strategies import TruthfulArm, OverReporterArm
#     from Linucbperarm import run

#     # Check 1 (paper Lemma E.2): truthful arms are never eliminated.
#     false_elims = 0
#     for seed in range(20):
#         env = Environment(seed=seed)
#         learner = OneSidedOptGTM(env.K, env.d, T=env.T)
#         run(env, learner, [TruthfulArm() for _ in range(env.K)])
#         false_elims += int((~learner.active).sum())
#     assert false_elims == 0, false_elims
#     print("Check 1: zero false eliminations across 20 truthful seeds.")

#     # Truthful sanity: regret still flattens (larger constant than LinUCB's
#     # tuned alpha, as expected from the theoretical radius).
#     env = Environment(seed=0)
#     learner = OneSidedOptGTM(env.K, env.d, T=env.T)
#     log = run(env, learner, [TruthfulArm() for _ in range(env.K)])
#     cum = log["cum_regret"]
#     q = env.T // 4
#     assert log["regret"][-q:].mean() < 0.5 * log["regret"][:q].mean()
#     print(f"Truthful OptGTM: cum regret {cum[-1]:.1f}, "
#           f"pulls {learner.pulls.tolist()}")

#     # Check 2, honest version: a constant norm-preserving over-reporter is
#     # NOT eliminated. Its reports are linearly realizable (paper Lemma E.1 /
#     # Assumption 3), so the trigger has nothing to detect; OptGTM then runs
#     # as a clean per-arm bandit (Corollary E.9) and the lie buys nothing.
#     # Detecting under-report-then-over-report manipulation is exactly the
#     # paper's acknowledged limitation (Remark 4.3, Appendix C).
#     env = Environment(seed=0)
#     strats = [TruthfulArm() for _ in range(env.K)]
#     strats[0] = OverReporterArm(delta=0.3)
#     learner = OneSidedOptGTM(env.K, env.d, T=env.T)
#     log = run(env, learner, strats)
#     assert learner.eliminated_at[0] is None
#     assert learner.active.all()
#     print(f"Over-reporter (+0.3): not eliminated (realizable lie), "
#           f"cum regret {log['cum_regret'][-1]:.1f}")
#     print("All checks passed.")
