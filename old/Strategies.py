"""Arm reporting strategies.
 
A strategy is any object with one method:
 
    report(t, x_true, theta_star) -> x_reported
 
- t:          the round index (lets strategies behave differently over time,
              e.g. bank credit early and spend it late).
- x_true:     the arm's true context for round t, shape (d,). It is this
              arm's private information.
- theta_star: the true parameter, shape (d,), unit norm. Strategic arms are
              assumed to know the reward-relevant direction; the LEARNER
              never sees it. A truthful arm simply ignores it.
 
The returned report is what the learner sees in place of x_true. The
environment never sees reports: rewards always come from the true context. 
Strategies must not mutate x_true; return a new array (or x_true itself,
which the run loop must then treat as read-only — the truthful strategy
returns a copy to be safe).
"""
import numpy as np
from Environment import Environment,shifted_context
class TruthfulArm:
    """Reports the true context unchanged."""
    name = "truthful"
    def report(self, t, x_true, theta_star):
        return np.array(x_true, dtype=float, copy=True)
class UnderReporterArm:
    """Deflates the theta*-component by `delta` (norm-preserving).

    On its own this makes the arm look worse and is not rational; it is a
    building block for the later bank-then-spend strategy. If q - delta
    would fall below -1, the shift is clipped to land exactly at -1.
    """

    name = "under_reporter"

    def __init__(self, delta=0.10):
        if delta <= 0:
            raise ValueError("delta must be positive for an under-reporter.")
        self.delta = delta
    def report(self, t, x_true, theta_star):
        theta = np.asarray(theta_star, float)
        theta = theta / np.linalg.norm(theta)
        q = float(np.asarray(x_true, float) @ theta)
        d_eff = -min(self.delta, 1.0 + q)  # keep q - delta >= -1
        if d_eff >= 0:
            return np.array(x_true, dtype=float, copy=True)
        return shifted_context(x_true, theta, d_eff)
class OverReporterArm:
    """Naive attacker: always inflates the theta*-component by `delta`.

    The report is norm-preserving (Environment.shifted_context rescales the
    orthogonal part), so a norm check cannot unmask it. If the true
    component q is so large that q + delta would exceed 1, the shift is
    clipped to land exactly at 1 (the strongest feasible over-report).
    """
    name = "over_reporter"
    def __init__(self, delta=0.15):
        if delta <= 0:
            raise ValueError("delta must be positive for an over-reporter.")
        self.delta = delta

    def report(self, t, x_true, theta_star): 
        theta = np.asarray(theta_star, float)
        theta = theta / np.linalg.norm(theta)
        q = float(np.asarray(x_true, float) @ theta)
        d_eff = min(self.delta, 1.0 - q)  # keep q + delta <= 1
        if d_eff <= 0:
            return np.array(x_true, dtype=float, copy=True)
        return shifted_context(x_true, theta, d_eff)
class CreditBankerArm:
    """Credit-banking attacker: under-report before T0, over-report after.

    For t < T0 the theta*-component is deflated by `a_minus` (banking);
    for t >= T0 it is inflated by `a_plus` (spending). Both phases use the
    same norm-preserving shift and the same clipping as UnderReporterArm
    and OverReporterArm, so with the same `a_plus` the spending phase is
    report-for-report identical to OverReporterArm(a_plus): any difference
    in outcome between the two is due to the banking phase alone.

    A zero shift is allowed and means truthful in that phase, so
    CreditBankerArm(0, 0, T0) is the truthful arm (used by the paired
    attack-gain harness to check that G is zero without an attack).
    """
    name = "credit_banker"

    def __init__(self, a_minus=0.10, a_plus=0.15, T0=1000):
        if a_minus < 0 or a_plus < 0:
            raise ValueError("a_minus and a_plus must be non-negative.")
        if T0 < 0:
            raise ValueError("T0 must be non-negative.")
        self.a_minus, self.a_plus, self.T0 = a_minus, a_plus, T0

    def report(self, t, x_true, theta_star):
        theta = np.asarray(theta_star, float)
        theta = theta / np.linalg.norm(theta)
        q = float(np.asarray(x_true, float) @ theta)
        if t < self.T0:
            d_eff = -min(self.a_minus, 1.0 + q)   # keep q - a_minus >= -1
        else:
            d_eff = min(self.a_plus, 1.0 - q)     # keep q + a_plus <= 1
        if d_eff == 0:
            return np.array(x_true, dtype=float, copy=True)
        return shifted_context(x_true, theta, d_eff)
    



