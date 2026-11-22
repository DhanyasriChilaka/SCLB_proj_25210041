
import numpy as np
def shifted_context(x_true, theta, delta):
        """Shift the theta-component while preserving the context norm.

        Decomposes x_true into q*theta + an orthogonal part,
        replaces q with q + delta, and rescales the orthogonal part so the
        report has the same norm as x_true. Requires
        |q + delta| <= ||x_true||.
        Negative delta under-reports, positive delta over-reports.
        """
        theta = np.array(theta, dtype=float, copy=True)
        theta /= np.linalg.norm(theta)
        x_true = np.asarray(x_true, dtype=float)
        radius = np.linalg.norm(x_true)
        q = float(x_true @ theta)
        q_new = q + delta
        tolerance = 1e-12 * max(1.0, radius)
        if q_new < -radius - tolerance or q_new > radius + tolerance:
            raise ValueError(
                "q + delta must lie in "
                "[-||x_true||, ||x_true||]."
            )
        # report_with_shift clips to this interval, but recomputing the
        # projection can move an endpoint by a few floating-point ulps.
        q_new = float(np.clip(q_new, -radius, radius))
        orth = x_true - q * theta
        orth_norm = np.linalg.norm(orth)
        if orth_norm <= 1e-12:
            if abs(abs(q_new) - radius) > 1e-12:
                raise ValueError("No orthogonal component to rescale; "
                                 "cannot preserve the context norm.")
            return q_new * theta
        u = orth / orth_norm
        return q_new * theta + np.sqrt(
            max(0.0, radius ** 2 - q_new ** 2)
        ) * u

def report_with_shift(x_true, theta_star, delta):
    theta = np.array(theta_star, dtype=float, copy=True)
    theta /= np.linalg.norm(theta)

    x_true = np.asarray(x_true, dtype=float)
    radius = np.linalg.norm(x_true)
    q = float(x_true @ theta)
    delta = np.clip(delta, -radius - q, radius - q)

    if np.isclose(delta, 0.0):
        return np.array(x_true, dtype=float, copy=True)

    return shifted_context(x_true, theta, delta)

class TruthfulArm:
    """Reports the true context unchanged."""
    name = "truthful"
    is_truthful = True

    def params(self):
        return {}

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

    def params(self):
        return {"delta": self.delta}

    def report(self, t, x_true, theta_star):
        return report_with_shift(
            x_true=x_true,
            theta_star=theta_star,
            delta=-self.delta,
        )
class OverReporterArm:
    """Naive attacker: always inflates the theta*-component by `delta`.

    The report is norm-preserving (shifted_context rescales the
    orthogonal part), so a norm check cannot unmask it. If the true
    component q is so large that q + delta would exceed 1, the shift is
    clipped to land exactly at 1 (the strongest feasible over-report).
    """
    name = "over_reporter"
    def __init__(self, delta=0.15):
        if delta <= 0:
            raise ValueError("delta must be positive for an over-reporter.")
        self.delta = delta

    def params(self):
        return {"delta": self.delta}

    def report(self, t, x_true, theta_star): 
        return report_with_shift(
            x_true=x_true,
            theta_star=theta_star,
            delta=self.delta,
        )
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
        self.is_truthful = a_minus == 0 and a_plus == 0

    def params(self):
        return {
            "a_minus": self.a_minus,
            "a_plus": self.a_plus,
            "T0": self.T0,
        }

    def report(self, t, x_true, theta_star):
        if t < self.T0:
            delta = -self.a_minus
        else:
            delta = self.a_plus

        return report_with_shift(
            x_true=x_true,
            theta_star=theta_star,
            delta=delta,
        )


class PullCountCreditBankerArm:
    """Bank for the arm's first `bank_pulls` selections, then spend.

    Unlike CreditBankerArm, the phase transition depends on how many times
    this arm has actually been selected, rather than on the global round.
    """

    name = "pull_count_credit_banker"

    def __init__(self, a_minus=0.10, a_plus=0.15, bank_pulls=100):
        if a_minus < 0 or a_plus < 0:
            raise ValueError("a_minus and a_plus must be non-negative.")
        if bank_pulls < 0:
            raise ValueError("bank_pulls must be non-negative.")
        self.a_minus = a_minus
        self.a_plus = a_plus
        self.bank_pulls = int(bank_pulls)
        self.is_truthful = a_minus == 0 and a_plus == 0
        self.selected_pulls = 0

    def params(self):
        return {
            "a_minus": self.a_minus,
            "a_plus": self.a_plus,
            "bank_pulls": self.bank_pulls,
        }

    def reset(self):
        self.selected_pulls = 0

    def report(self, t, x_true, theta_star):
        if self.selected_pulls < self.bank_pulls:
            delta = -self.a_minus
        else:
            delta = self.a_plus

        return report_with_shift(
            x_true=x_true,
            theta_star=theta_star,
            delta=delta,
        )

    def on_selected(self, round_index, reward, reported_context):
        self.selected_pulls += 1
