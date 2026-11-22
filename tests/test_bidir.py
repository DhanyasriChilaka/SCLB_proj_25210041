import numpy as np
import pytest

from src.environment import Environment
from src.strategies import TruthfulArm
from src.simulation import run_episode
from src.metrics import extract_arm_traces
from src.algorithms.optgtm import OneSidedOptGTM
from src.algorithms.bioptgtm import (
    BidirectionalOptGTM,
    WindowedBidirectionalOptGTM,
)


def test_two_sided_trigger_detects_high_claims(monkeypatch):
    """The upper-side trigger should detect excessive claims."""
    learner = BidirectionalOptGTM(
        K=1,
        d=2,
        T=100,
        sigma=0.1,
    )

    # Force the predicted interval to lie above the observed reward.
    monkeypatch.setattr(
        learner,
        "bounds",
        lambda arm, x: (2.0, 3.0),
    )

    # Remove the confidence allowance so this test checks only the
    # trigger inequality.
    monkeypatch.setattr(
        learner,
        "reward_slack",
        lambda m: 0.0,
    )

    x_reported = np.array([1.0, 0.0])
    observed_reward = 0.0

    learner.update(
        arm=0,
        x_reported=x_reported,
        reward=observed_reward,
    )

    assert not learner.active[0]
    assert learner.eliminated_at[0] == 1
    assert learner.elimination_side[0] == "over"


def test_two_sided_trigger_detects_low_claims(monkeypatch):
    """The lower-side trigger should detect sustained under-claiming."""
    learner = BidirectionalOptGTM(
        K=1,
        d=2,
        T=100,
        sigma=0.1,
    )

    # The reported-context model predicts at most 0.25, but the arm
    # delivers reward 1.0.
    monkeypatch.setattr(
        learner,
        "bounds",
        lambda arm, x: (0.0, 0.25),
    )

    monkeypatch.setattr(
        learner,
        "reward_slack",
        lambda m: 0.0,
    )

    x_reported = np.array([1.0, 0.0])
    observed_reward = 1.0

    learner.update(
        arm=0,
        x_reported=x_reported,
        reward=observed_reward,
    )

    assert not learner.active[0]
    assert learner.eliminated_at[0] == 1
    assert learner.elimination_side[0] == "under"


def test_one_sided_trigger_does_not_detect_low_claims(monkeypatch):
    """Original OptGTM has no lower-side trigger."""
    learner = OneSidedOptGTM(
        K=1,
        d=2,
        T=100,
        sigma=0.1,
    )

    monkeypatch.setattr(
        learner,
        "bounds",
        lambda arm, x: (0.0, 0.25),
    )

    monkeypatch.setattr(
        learner,
        "reward_slack",
        lambda m: 0.0,
    )

    x_reported = np.array([1.0, 0.0])
    observed_reward = 1.0

    learner.update(
        arm=0,
        x_reported=x_reported,
        reward=observed_reward,
    )

    assert learner.active[0]
    assert learner.eliminated_at[0] is None


def _windowed_learner_after_three_updates(monkeypatch):
    """Construct a two-entry window after three deterministic updates."""
    learner = WindowedBidirectionalOptGTM(
        K=1,
        d=2,
        T=100,
        sigma=0.1,
        window=2,
    )

    # Bounds returned on the first, second and third pulls.
    bounds_sequence = iter(
        [
            (1.0, 2.0),
            (3.0, 4.0),
            (5.0, 6.0),
        ]
    )

    monkeypatch.setattr(
        learner,
        "bounds",
        lambda arm, x: next(bounds_sequence),
    )

    # Prevent elimination so that the window behavior can be tested
    # independently of trigger behavior.
    monkeypatch.setattr(
        learner,
        "reward_slack",
        lambda m: 1_000_000.0,
    )

    x_reported = np.array([1.0, 0.0])

    learner.update(
        arm=0,
        x_reported=x_reported,
        reward=10.0,
    )

    learner.update(
        arm=0,
        x_reported=x_reported,
        reward=20.0,
    )

    learner.update(
        arm=0,
        x_reported=x_reported,
        reward=30.0,
    )

    return learner


def test_window_contains_last_w_arm_pulls(monkeypatch):
    learner = _windowed_learner_after_three_updates(
        monkeypatch
    )

    # Full records retain all three observations.
    assert len(learner.records[0]) == 3

    # The trigger window retains only the most recent two.
    assert len(learner._win[0]) == 2

    expected_window = [
        (3.0, 4.0, 20.0),
        (5.0, 6.0, 30.0),
    ]

    actual_window = list(learner._win[0])

    assert actual_window == expected_window


def test_window_sums_evict_oldest_observation(monkeypatch):
    learner = _windowed_learner_after_three_updates(
        monkeypatch
    )

    # The first tuple (1, 2, 10) should have been removed.
    expected_lcb_sum = 3.0 + 5.0
    expected_ucb_sum = 4.0 + 6.0
    expected_reward_sum = 20.0 + 30.0

    assert np.isclose(
        learner.win_l[0],
        expected_lcb_sum,
    )

    assert np.isclose(
        learner.win_u[0],
        expected_ucb_sum,
    )

    assert np.isclose(
        learner.win_r[0],
        expected_reward_sum,
    )


def test_windowed_learner_keeps_cumulative_sums(monkeypatch):
    learner = _windowed_learner_after_three_updates(monkeypatch)

    assert np.isclose(learner.sum_l[0], 1.0 + 3.0 + 5.0)
    assert np.isclose(learner.sum_u[0], 2.0 + 4.0 + 6.0)
    assert np.isclose(learner.sum_r[0], 10.0 + 20.0 + 30.0)


def test_windowed_traces_match_the_actual_trigger_window(monkeypatch):
    learner = _windowed_learner_after_three_updates(monkeypatch)
    traces = extract_arm_traces(learner, arm=0)

    assert np.allclose(traces["lcb_sum"], [1.0, 4.0, 8.0])
    assert np.allclose(traces["ucb_sum"], [2.0, 6.0, 10.0])
    assert np.allclose(traces["reward_sum"], [10.0, 30.0, 50.0])
    assert np.allclose(
        traces["cumulative_lcb_sum"],
        [1.0, 4.0, 9.0],
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    ("learner_class", "extra_kwargs"),
    [
        (BidirectionalOptGTM, {}),
        (
            WindowedBidirectionalOptGTM,
            {"window": 50},
        ),
    ],
)
def test_truthful_arm_survives_bidirectional_trigger(
    seed,
    learner_class,
    extra_kwargs,
):
    """Truthful arms should survive both bidirectional variants."""
    env = Environment(
        seed=seed,
        T=500,
    )

    learner = learner_class(
        K=env.K,
        d=env.d,
        T=env.T,
        sigma=env.sigma,
        **extra_kwargs,
    )

    strategies = [
        TruthfulArm()
        for _ in range(env.K)
    ]

    run_episode(
        env=env,
        learner=learner,
        strategies=strategies,
    )

    assert learner.active.all(), (
        f"{learner_class.__name__} eliminated a truthful arm "
        f"for seed {seed}. "
        f"Active={learner.active.tolist()}, "
        f"eliminated_at={learner.eliminated_at}, "
        f"sides={learner.elimination_side}"
    )

    assert all(
        elimination is None
        for elimination in learner.eliminated_at
    )
