"""Tests for the strategic linear contextual-bandit environment."""

import numpy as np
import pytest

from src.environment import Environment


def rollout(env, policy_seed):
    """Run a random policy and return contexts, arms and rewards."""
    rng = np.random.default_rng(policy_seed)

    contexts = []
    arms = []
    rewards = []

    while not env.done:
        contexts.append(env.contexts())

        arm = int(rng.integers(env.K))
        arms.append(arm)

        rewards.append(env.step(arm))

    return (
        np.asarray(contexts),
        np.asarray(arms),
        np.asarray(rewards),
    )


def test_default_environment_properties():
    """The default environment should have the expected dimensions."""
    env = Environment(seed=0)

    assert np.isclose(np.linalg.norm(env.theta), 1.0)
    assert env._X.shape == (5000, 6, 5)
    assert env._means.shape == (5000, 6)
    assert env._noise.shape == (5000,)

    # The realized standard deviation should be close to sigma=0.1.
    assert abs(env._noise.std() - env.sigma) < 0.01


def test_controlled_context_construction():
    """Controlled contexts should have the requested mean rewards."""
    env = Environment(seed=0)

    context_norms = np.linalg.norm(
        env._X,
        axis=-1,
    )

    assert np.allclose(
        context_norms,
        1.0,
        atol=1e-10,
    )

    # By construction:
    # <theta*, x*_{t,i}> = q_{t,i}.
    assert np.allclose(
        env._means,
        env._q,
    )

    # Arm 0 should have the largest average quality.
    assert (
        env._means[:, 0].mean()
        >
        env._means[:, 1:].mean(axis=0).max()
    )

    # It should usually, but not always, be the best arm.
    best_share = env.best_arm_share(0)

    assert 0.4 < best_share < 0.95, (
        f"Arm 0 best-arm share {best_share:.3f} "
        "was outside the expected range."
    )


def test_controlled_means_equal_context_inner_products():
    """Stored means should equal theta* dotted with true contexts."""
    env = Environment(seed=0)

    reconstructed_means = env._X @ env.theta

    assert np.allclose(
        env._means,
        reconstructed_means,
    )


def test_iid_context_mode():
    """IID contexts should lie inside the unit ball."""
    env = Environment(
        seed=0,
        context_mode="iid",
    )

    context_norms = np.linalg.norm(
        env._X,
        axis=-1,
    )

    assert context_norms.max() <= 1.0 + 1e-12
    assert env._q is None
    assert env.qualities is None


def test_iid_arms_are_approximately_exchangeable():
    """No arm should dominate the IID environment."""
    env = Environment(
        seed=0,
        context_mode="iid",
    )

    shares = np.array(
        [
            env.best_arm_share(i)
            for i in range(env.K)
        ]
    )

    assert np.isclose(
        shares.sum(),
        1.0,
    )

    # With K=6, each arm should be best approximately 1/6 of the time.
    assert np.all(
        (shares > 0.10)
        & (shares < 0.24)
    ), shares


def test_same_seed_and_policy_reproduce_identical_run():
    """Identical seeds and policies should reproduce the same episode."""
    contexts_1, arms_1, rewards_1 = rollout(
        Environment(seed=0),
        policy_seed=1,
    )

    contexts_2, arms_2, rewards_2 = rollout(
        Environment(seed=0),
        policy_seed=1,
    )

    assert np.array_equal(
        contexts_1,
        contexts_2,
    )

    assert np.array_equal(
        arms_1,
        arms_2,
    )

    assert np.array_equal(
        rewards_1,
        rewards_2,
    )


def test_different_policies_receive_same_context_sequence_and_noise():
    """Policy choices must not alter environment randomness."""
    env_1 = Environment(seed=0)
    env_2 = Environment(seed=0)

    contexts_1, arms_1, rewards_1 = rollout(
        env_1,
        policy_seed=1,
    )

    contexts_2, arms_2, rewards_2 = rollout(
        env_2,
        policy_seed=2,
    )

    # The policies should actually be different.
    assert not np.array_equal(
        arms_1,
        arms_2,
    )

    # They should face the same true context arrays.
    assert np.array_equal(
        contexts_1,
        contexts_2,
    )

    rounds = np.arange(env_1.T)

    noise_1 = (
        rewards_1
        - env_1._means[rounds, arms_1]
    )

    noise_2 = (
        rewards_2
        - env_2._means[rounds, arms_2]
    )

    # Both policies receive the same exogenous round-level noise.
    assert np.allclose(
        noise_1,
        noise_2,
        atol=1e-12,
    )


def test_reset_replays_same_episode():
    """Resetting should rewind time without redrawing randomness."""
    env = Environment(seed=0)

    contexts_1, arms_1, rewards_1 = rollout(
        env,
        policy_seed=1,
    )

    env.reset()

    contexts_2, arms_2, rewards_2 = rollout(
        env,
        policy_seed=1,
    )

    assert np.array_equal(
        contexts_1,
        contexts_2,
    )

    assert np.array_equal(
        arms_1,
        arms_2,
    )

    assert np.array_equal(
        rewards_1,
        rewards_2,
    )


def test_reset_sets_round_to_zero():
    env = Environment(
        seed=0,
        T=10,
    )

    env.step(0)
    env.step(1)

    assert env.t == 2

    env.reset()

    assert env.t == 0
    assert not env.done


def test_different_seed_produces_different_environment():
    """Different seeds should change contexts and noise."""
    env_1 = Environment(seed=0)
    env_2 = Environment(seed=1)

    assert not np.array_equal(
        env_1._X,
        env_2._X,
    )

    assert not np.array_equal(
        env_1._noise,
        env_2._noise,
    )


def test_contexts_returns_a_copy():
    """Mutating returned contexts must not modify the environment."""
    env = Environment(
        seed=0,
        T=10,
    )

    returned_contexts = env.contexts()
    original_contexts = env._X[0].copy()

    returned_contexts[0, 0] += 100.0

    assert np.array_equal(
        env._X[0],
        original_contexts,
    )


def test_internal_environment_arrays_are_read_only():
    env = Environment(seed=0, T=10)

    assert not env.theta.flags.writeable
    assert not env._X.flags.writeable
    assert not env._means.flags.writeable
    assert not env._noise.flags.writeable


def test_reports_cannot_change_environment_reward():
    """The environment reward depends only on the selected true context."""
    env_1 = Environment(seed=0)
    env_2 = Environment(seed=0)

    true_contexts = env_1.contexts()

    # Construct a deliberately false report. It is never passed to step().
    fake_report = -true_contexts

    assert not np.array_equal(
        fake_report,
        true_contexts,
    )

    reward_1 = env_1.step(0)
    reward_2 = env_2.step(0)

    assert reward_1 == reward_2


def test_step_returns_expected_mean_plus_predrawn_noise():
    env = Environment(seed=0)

    arm = 0

    expected_reward = (
        env._means[0, arm]
        + env._noise[0]
    )

    actual_reward = env.step(arm)

    assert np.isclose(
        actual_reward,
        expected_reward,
    )


def test_regret_matches_oracle_definition():
    env = Environment(seed=0)

    round_index = 0
    arm = 2

    expected_regret = (
        env._means[round_index].max()
        - env._means[round_index, arm]
    )

    assert np.isclose(
        env.regret(arm, round_index),
        expected_regret,
    )


def test_best_arm_matches_largest_expected_reward():
    env = Environment(seed=0)

    expected_best_arm = int(
        np.argmax(env._means[0])
    )

    assert env.best_arm(t=0) == expected_best_arm


def test_contexts_raises_after_horizon():
    env = Environment(
        seed=0,
        T=1,
    )

    env.step(0)

    assert env.done

    with pytest.raises(
        RuntimeError,
        match="Horizon reached",
    ):
        env.contexts()


def test_step_raises_after_horizon():
    env = Environment(
        seed=0,
        T=1,
    )

    env.step(0)

    with pytest.raises(
        RuntimeError,
        match="Horizon reached",
    ):
        env.step(0)


@pytest.mark.parametrize(
    "invalid_arm",
    [-1, 6],
)
def test_step_rejects_invalid_arm(invalid_arm):
    env = Environment(seed=0)

    with pytest.raises(ValueError):
        env.step(invalid_arm)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d": 1},
        {"K": 1},
        {"T": 0},
        {"sigma": -0.1},
        {"theta": np.zeros(5)},
        {"context_mode": "invalid"},
        {"quality_jitter": -0.1},
        {"qualities": np.full(6, 0.99)},
    ],
)
def test_invalid_environment_parameters(kwargs):
    """Invalid constructor arguments should raise ValueError."""
    with pytest.raises(ValueError):
        Environment(
            seed=0,
            **kwargs,
        )


def test_quality_vector_must_match_number_of_arms():
    with pytest.raises(
        ValueError,
        match="qualities must have shape",
    ):
        Environment(
            seed=0,
            K=4,
            qualities=np.array([0.6, 0.5]),
        )


def test_supplied_theta_must_match_dimension():
    with pytest.raises(
        ValueError,
        match="theta must have shape",
    ):
        Environment(
            seed=0,
            d=5,
            theta=np.ones(4),
        )
