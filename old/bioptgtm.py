from collections import deque
import numpy as np

from .linucb_per_arm import PerArmLinUCB
from .optgtm import OneSidedOptGTM
"""-Paired experiment: naive over-reporting vs per-arm LinUCB ---
    FINDING (checked on seeds 0-2): against a per-arm estimator that
    regresses observed rewards on REPORTED contexts, a constant
    norm-preserving over-report does NOT increase the attacker's
    selections. Two channels neutralize it:
      1. The report keeps norm 1, so the initial exploration bonus
         alpha * sqrt(x V^{-1} x) is identical to the truthful one.
      2. The learner calibrates: it fits predictions to realized rewards
         on the reported contexts, so a constant shift is absorbed into
         theta_hat_i, and predicted means return to the truth on average.
    Large shifts even BACKFIRE (slope compression: the arm stops looking
    especially good in its best rounds, so it is pulled less). What the
    lie does do is distort the arm's model, which raises the learner's
    regret. A time-varying strategy (bank by under-reporting, then spend
    by over-reporting) is what actually buys extra selections -- that is
    the credit-banking experiment this project builds toward."""

class BidirectionalOptGTM(OneSidedOptGTM):
    """Two-sided trigger: eliminate arm i after its m-th pull when either

        sum_s l_s > sum_s r_s + c(m, delta)      (claims too HIGH: over-claiming)
        sum_s u_s < sum_s r_s - c(m, delta)      (claims too LOW: sandbagging)

    i.e. when the claimed interval [sum l, sum u] no longer overlaps the
    observed interval [sum r - c, sum r + c]. The second condition removes
    the paper's Assumption 2 blind spot on the detection side: an arm whose
    delivered rewards exceed even the optimistic reading of its reports
    must have been under-reporting. Everything else (estimators, recording
    at pull time with the outdated estimator, selection, permanence of
    elimination) is inherited unchanged.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sum_u = np.zeros(self.K)
        self.elimination_side = [None] * self.K   # "over" or "under"

    def update(self, arm, x_reported, reward):
        l, u = self.bounds(arm, x_reported)   # outdated estimator, pre-update
        self.records[arm].append((l, u, reward))
        self.sum_l[arm] += l
        self.sum_u[arm] += u
        self.sum_r[arm] += reward
        # Skip OneSidedOptGTM.update (it would re-record and re-test);
        # go straight to the estimator update.
        PerArmLinUCB_update(self, arm, x_reported, reward)
        m = self.pulls[arm]
        if not self.active[arm]:
            return
        c = self.reward_slack(m)
        if self.sum_l[arm] > self.sum_r[arm] + c:
            self.active[arm] = False
            self.eliminated_at[arm] = m
            self.elimination_side[arm] = "over"
        elif self.sum_u[arm] < self.sum_r[arm] - c:
            self.active[arm] = False
            self.eliminated_at[arm] = m
            self.elimination_side[arm] = "under"


class WindowedBidirectionalOptGTM(BidirectionalOptGTM):
    """Both trigger conditions evaluated on arm i's LAST W PULLS only.

    The window is in the arm's own pull count, not global rounds, so every
    arm's trigger rests on the same amount of evidence regardless of how
    often it is selected. Only the trigger statistics are windowed; the
    least-squares estimator stays cumulative, and each (l_s, u_s, r_s) is
    still recorded at pull time with the then-current estimator. The slack
    uses the number of tuples actually in the window, c(min(m, W), delta).
    Elimination remains permanent.
    """

    def __init__(self, *args, window=200, **kwargs):
        if window < 1:
            raise ValueError("window must be >= 1.")
        super().__init__(*args, **kwargs)
        self.window = window
        self._win = [deque(maxlen=window) for _ in range(self.K)]
        # Windowed running sums, maintained incrementally.
        self.win_l = np.zeros(self.K)
        self.win_u = np.zeros(self.K)
        self.win_r = np.zeros(self.K)

    def update(self, arm, x_reported, reward):
        l, u = self.bounds(arm, x_reported)   # outdated estimator, pre-update
        self.records[arm].append((l, u, reward))   # full history, for logs
        w = self._win[arm]
        if len(w) == self.window:                  # deque will evict: subtract
            ol, ou, orr = w[0]
            self.win_l[arm] -= ol
            self.win_u[arm] -= ou
            self.win_r[arm] -= orr
        w.append((l, u, reward))
        self.win_l[arm] += l
        self.win_u[arm] += u
        self.win_r[arm] += reward
        PerArmLinUCB.update(self, arm, x_reported, reward)   # cumulative estimator
        if not self.active[arm]:
            return
        c = self.reward_slack(len(w))
        if self.win_l[arm] > self.win_r[arm] + c:
            self.active[arm] = False
            self.eliminated_at[arm] = self.pulls[arm]
            self.elimination_side[arm] = "over"
        elif self.win_u[arm] < self.win_r[arm] - c:
            self.active[arm] = False
            self.eliminated_at[arm] = self.pulls[arm]
            self.elimination_side[arm] = "under"



