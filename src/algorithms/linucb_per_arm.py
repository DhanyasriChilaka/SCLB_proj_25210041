import numpy as np
class PerArmLinUCB:
    def __init__(self, K, d, alpha=1.0, lam=1.0):
        if alpha < 0 or lam <= 0:
            raise ValueError("alpha must be >= 0 and lam > 0.")
        self.K, self.d, self.alpha, self.lam = K, d, alpha, lam
        self.V = np.stack([lam * np.eye(d) for _ in range(K)])  # (K, d, d)
        self.b = np.zeros((K, d))
        self.pulls = np.zeros(K, dtype=int)

    def theta_hat(self, i):
        return np.linalg.solve(self.V[i], self.b[i])

    def ucbs(self, reported):
        """UCB for each arm given reported contexts, shape (K,)."""
        reported = np.asarray(reported, dtype=float)
        out = np.empty(self.K)
        for i in range(self.K):
            x = reported[i]
            # Solve once, reuse for both the mean and the width:
            # V^{-1} x gives theta_hat . x via b, and x^T V^{-1} x directly.
            Vinv_x = np.linalg.solve(self.V[i], x)
            mean = self.b[i] @ Vinv_x          # == theta_hat_i . x
            width = np.sqrt(max(x @ Vinv_x, 0.0))
            out[i] = mean + self.alpha * width
        return out

    def select(self, reported):
        u = self.ucbs(reported)
        # Deterministic tie-break: lowest index among maxima.
        return int(np.argmax(u)), u

    def update(self, arm, x_reported, reward):
        x = np.asarray(x_reported, dtype=float)
        self.V[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.pulls[arm] += 1