
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

    # --- Utility for strategic arms (not used by the environment) ---

    @staticmethod
    def shifted_context(x_true, theta, delta):
        return shifted_context(x_true, theta, delta)

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


if __name__ == "__main__":
    def rollout(env, policy_seed):
        """Random policy with its own RNG; returns contexts, arms, rewards."""
        rng = np.random.default_rng(policy_seed)
        X, arms, rewards = [], [], []
        while not env.done:
            X.append(env.contexts())
            a = int(rng.integers(env.K))
            arms.append(a)
            rewards.append(env.step(a))
        return np.array(X), np.array(arms), np.array(rewards)

    # 1. Basic properties (controlled mode).
    env = Environment(seed=0)
    assert np.isclose(np.linalg.norm(env.theta), 1.0)
    assert env._X.shape == (5000, 6, 5)
    assert abs(env._noise.std() - 0.1) < 0.01

    # 2. Controlled construction: unit norms, means match q exactly,
    #    attacker leads on average but not every round.
    assert np.allclose(np.linalg.norm(env._X, axis=-1), 1.0, atol=1e-10)
    assert np.allclose(env._means, env._q)
    assert env._means[:, 0].mean() > env._means[:, 1:].mean(axis=0).max()
    share = env.best_arm_share(0)
    assert 0.4 < share < 0.95, f"best-arm share {share:.3f} out of range"
    print(f"Arm 0 (attacker) is the true best in {share:.1%} of rounds; "
          f"mean rewards per arm: {np.round(env._means.mean(0), 3)}")

    # 3. iid mode: norms <= 1, arms exchangeable (each best ~1/K of rounds).
    env_iid = Environment(seed=0, context_mode="iid")
    assert np.linalg.norm(env_iid._X, axis=-1).max() <= 1.0
    shares = [env_iid.best_arm_share(i) for i in range(6)]
    assert all(0.10 < s < 0.24 for s in shares)

    # 4. Same seed, same policy -> identical everything.
    X1, a1, r1 = rollout(Environment(seed=0), policy_seed=1)
    X2, a2, r2 = rollout(Environment(seed=0), policy_seed=1)
    assert np.array_equal(X1, X2) and np.array_equal(r1, r2)

    # 5. Same seed, DIFFERENT policy -> contexts and noise still identical.
    e1, e2 = Environment(seed=0), Environment(seed=0)
    X1, a1, r1 = rollout(e1, policy_seed=1)
    X2, a2, r2 = rollout(e2, policy_seed=2)
    assert not np.array_equal(a1, a2)
    assert np.array_equal(X1, X2)
    T = np.arange(e1.T)
    noise1 = r1 - e1._means[T, a1]
    noise2 = r2 - e2._means[T, a2]
    assert np.allclose(noise1, noise2, atol=1e-12)

    # 6. reset() replays the same run.
    e1.reset()
    _, _, r1b = rollout(e1, policy_seed=1)
    assert np.array_equal(r1, r1b)

    # 7. Different seed -> different environment.
    e3 = Environment(seed=1)
    assert not np.array_equal(e3._X, e1._X)
    assert not np.array_equal(e3._noise, e1._noise)

    # 8. Reports cannot affect rewards: step() never sees them.
    ea, eb = Environment(seed=0), Environment(seed=0)
    _true_x = ea.contexts()
    _fake_report = -_true_x  # deliberately never passed to step()
    assert ea.step(0) == eb.step(0)

    # 9. Norm-preserving report shift: unit norm, theta-component moves
    #    by exactly delta, true reward untouched.
    env.reset()
    x_true = env.contexts()[0]
    for delta in (-0.08, 0.15):
        x_rep = Environment.shifted_context(x_true, env.theta, delta)
        assert np.isclose(np.linalg.norm(x_rep), 1.0)
        assert np.isclose(x_rep @ env.theta, x_true @ env.theta + delta)
    assert np.isclose(env.step(0), env._means[0, 0] + env._noise[0])

    # 10. Guards: contexts()/step() after horizon, bad arm, bad params.
    done_env = Environment(seed=0, T=1)
    done_env.step(0)
    for fn in (done_env.contexts, lambda: done_env.step(0)):
        try:
            fn()
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
    for bad in (lambda: env.step(6), lambda: env.step(-1),
                lambda: Environment(seed=0, d=1),
                lambda: Environment(seed=0, K=1),
                lambda: Environment(seed=0, T=0),
                lambda: Environment(seed=0, sigma=-0.1),
                lambda: Environment(seed=0, theta=np.zeros(5)),
                lambda: Environment(seed=0, context_mode="nope"),
                lambda: Environment(seed=0, qualities=np.full(6, 0.99))):
        try:
            bad()
            raise AssertionError("expected an error")
        except (ValueError, RuntimeError):
            pass

    print("All checks passed.")