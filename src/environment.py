"""Linear contextual bandit environment with true (unreported) contexts.

Rewards are always computed from the TRUE context of the chosen arm:
    r_t = <theta*, x*_{t, a_t}> + eta_t,   eta_t ~ N(0, sigma^2)

All environment randomness (theta*, contexts, noise) is pre-drawn from
independent streams derived from `seed`. Nothing an arm or learner does can
shift those streams, so two runs with the same seed face identical contexts
and identical noise (paired seeds / common random numbers).

Two context modes:

- "controlled" (default): each arm i has a persistent target mean reward
  qualities[i]. Each round, a round-specific target q_{t,i} = qualities[i]
  + jitter is drawn (clipped to [0.02, 0.98]) and the true context is
  constructed as
      x*_{t,i} = q_{t,i} * theta* + sqrt(1 - q_{t,i}^2) * u_{t,i},
  with u_{t,i} a random unit vector orthogonal to theta*. Hence
  ||x*_{t,i}|| = 1 and <theta*, x*_{t,i}> = q_{t,i} exactly. This is a
  data-generation procedure for experimental control, not information
  available to the learner; the learner must never receive the qualities,
  theta*, or true contexts.

- "iid": contexts uniform in the unit ball (all arms exchangeable). Kept
  as a robustness check that results do not depend on the controlled
  construction.

Note on clipping: after clipping, the realized mean reward of an arm whose
jittered targets hit the bounds is pulled slightly toward the interior, so
it is no longer exactly qualities[i]. Negligible for the default values.
"""

import numpy as np

DEFAULT_QUALITIES = np.array([0.60, 0.52, 0.48, 0.44, 0.40, 0.36])
Q_MIN, Q_MAX = 0.02, 0.98
class Environment:
    def __init__(self, seed, d=5, K=6, T=5000, sigma=0.1, theta=None,
                 context_mode="controlled", qualities=None,
                 quality_jitter=0.08):
        # --- Parameter validation ---
        if d < 2:
            raise ValueError("d must be at least 2 (needed for the "
                             "orthogonal-context construction).")
        if K < 2:
            raise ValueError("K must be at least 2.")
        if T <= 0:
            raise ValueError("T must be positive.")
        if sigma < 0:
            raise ValueError("sigma must be non-negative.")
        if context_mode not in ("controlled", "iid"):
            raise ValueError("context_mode must be 'controlled' or 'iid'.")
        if quality_jitter < 0:
            raise ValueError("quality_jitter must be non-negative.")

        self.seed, self.d, self.K, self.T, self.sigma = seed, d, K, T, sigma
        self.context_mode = context_mode
        self.quality_jitter = quality_jitter

        # Three independent child streams: theta, contexts, noise.
        ss_theta, ss_ctx, ss_noise = np.random.SeedSequence(seed).spawn(3)
        rng_theta = np.random.default_rng(ss_theta)
        rng_ctx = np.random.default_rng(ss_ctx)
        rng_noise = np.random.default_rng(ss_noise)

        # theta*: unit norm.
        if theta is None:
            theta = rng_theta.standard_normal(d)
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (d,):
            raise ValueError(f"theta must have shape ({d},).")
        theta_norm = np.linalg.norm(theta)
        if theta_norm <= 1e-12:
            raise ValueError("theta must have non-zero norm.")
        self.theta = theta / theta_norm

        # --- True contexts ---
        if context_mode == "controlled":
            if qualities is None:
                if K == len(DEFAULT_QUALITIES):
                    qualities = DEFAULT_QUALITIES.copy()
                else:
                    qualities = np.linspace(0.60, 0.36, K)
            qualities = np.asarray(qualities, dtype=float)
            if qualities.shape != (K,):
                raise ValueError(f"qualities must have shape ({K},).")
            if np.any(qualities < Q_MIN) or np.any(qualities > Q_MAX):
                raise ValueError(
                    f"qualities must lie in [{Q_MIN}, {Q_MAX}].")
            self.qualities = qualities

            # Round-specific target means, clipped so sqrt(1 - q^2) is real
            # and so an over-report of up to ~0.15 along theta* can stay on
            # the unit sphere.
            q = qualities[None, :] + quality_jitter * rng_ctx.standard_normal((T, K))
            q = np.clip(q, Q_MIN, Q_MAX)
            self._q = q

            # Random unit vectors orthogonal to theta*.
            z = rng_ctx.standard_normal((T, K, d))
            z -= (z @ self.theta)[..., None] * self.theta
            z /= np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)

            # ||x|| = 1 and <theta*, x> = q exactly.
            self._X = (q[..., None] * self.theta
                       + np.sqrt(1.0 - q ** 2)[..., None] * z)
        else:  # "iid": uniform in the unit ball.
            self.qualities = None
            self._q = None
            g = rng_ctx.standard_normal((T, K, d))
            g /= np.linalg.norm(g, axis=-1, keepdims=True)
            radius = rng_ctx.random((T, K, 1)) ** (1.0 / d)
            self._X = g * radius

        # One noise draw per round, shared across arms: shape (T,).
        # Matches the paper's r_t = <theta*, x*> + eta_t and gives
        # lower-variance paired comparisons across policies.
        self._noise = sigma * rng_noise.standard_normal(T)

        self._means = self._X @ self.theta  # (T, K) expected rewards
        self.t = 0

        # Prevent learners and reporting strategies from changing the true
        # environment data through shared array references.
        for array in (self.theta, self._X, self._means, self._noise):
            array.setflags(write=False)

    def reset(self):
        """Rewind to round 0. Contexts and noise are unchanged."""
        self.t = 0

    @property
    def done(self):
        return self.t >= self.T

    def contexts(self):
        """True contexts x*_{t,i} for the current round, shape (K, d)."""
        if self.done:
            raise RuntimeError("Horizon reached; call reset().")
        return self._X[self.t].copy()

    def step(self, arm):
        """Pull `arm` in the current round; return the noisy reward.

        The reward uses the true context, never a reported one, which is why
        this method takes only the arm index.
        """
        if self.done:
            raise RuntimeError("Horizon reached; call reset().")
        arm = int(arm)
        if not 0 <= arm < self.K:
            raise ValueError(f"arm must be in [0, {self.K - 1}].")
        r = self._means[self.t, arm] + self._noise[self.t]
        self.t += 1
        return float(r)
    
    # --- Oracle quantities for evaluation (not for the learner) ---
        
    def expected_rewards(self, t=None):
        """<theta*, x*_{t,i}> for all arms, shape (K,)."""
        return self._means[self.t if t is None else t].copy()

    def best_arm(self, t=None):
        return int(np.argmax(self._means[self.t if t is None else t]))

    def best_arm_share(self, arm=0):
        """Fraction of rounds in which `arm` is the true best arm."""
        return float((self._means.argmax(axis=1) == arm).mean())

    def regret(self, arm, t):
        """Instantaneous pseudo-regret of pulling `arm` in round t."""
        return self._means[t].max() - self._means[t, arm]
