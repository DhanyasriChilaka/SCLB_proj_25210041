from collections import deque
import numpy as np

from .optgtm import OneSidedOptGTM


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

    def _record(self, arm, l, u, reward):
        super()._record(arm, l, u, reward)
        self.sum_u[arm] += u

    def _triggered(self, arm):
        m = self.pulls[arm]
        c = self.reward_slack(m)
        if self.sum_l[arm] > self.sum_r[arm] + c:
            self.elimination_side[arm] = "over"
            return True
        elif self.sum_u[arm] < self.sum_r[arm] - c:
            self.elimination_side[arm] = "under"
            return True
        return False


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

    def _record(self, arm, l, u, reward):
        # Retain the inherited full-history records and cumulative sums for
        # analysis, while maintaining separate sums for the trigger window.
        super()._record(arm, l, u, reward)
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

    def _triggered(self, arm):
        w = self._win[arm]
        c = self.reward_slack(len(w))
        if self.win_l[arm] > self.win_r[arm] + c:
            self.elimination_side[arm] = "over"
            return True
        elif self.win_u[arm] < self.win_r[arm] - c:
            self.elimination_side[arm] = "under"
            return True
        return False



