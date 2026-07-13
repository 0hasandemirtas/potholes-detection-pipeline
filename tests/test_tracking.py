import numpy as np

from src.tracking import TrackState
from src.smoothing import EmaBoxSmoother, NoOpBoxSmoother



def test_track_is_confirmed_after_consecutive_frames():
    """Verilen track_id'nin n_confirm sayısı kadar ardışık karede görünür olduktan sonra onaylandığını test eder."""
    state = TrackState(n_confirm=3, m_persist=5)

    state.update({1})
    assert state.is_visible(1) is False

    state.update({1})
    assert state.is_visible(1) is False

    state.update({1})
    assert state.is_visible(1) is True

def test_confirmation_count_resets_when_track_is_missed():
    """Verilen track_id'nin n_confirm sayısı kadar ardışık karede görünür olduktan sonra onaylandığını test eder. Eğer track_id kaybolursa, onay sayısı sıfırlanır."""
    state = TrackState(n_confirm=3, m_persist=5)

    state.update({1})
    state.update({1})

    state.update(set())

    state.update({1})
    state.update({1})

    assert state.is_visible(1) is False

    state.update({1})

    assert state.is_visible(1) is True

def test_track_is_immediately_confirmed_when_n_confirm_is_one():
    """n_confirm 1 olduğunda, track_id'nin görünür olduğu ilk karede onaylandığını test eder."""
    state = TrackState(n_confirm=1, m_persist=5)

    state.update({1})

    assert state.is_visible(1) is True

def test_track_is_removed_after_persistence_limit():
    """Verilen track_id'nin m_persist sayısı kadar ardışık karede kaybolduktan sonra silindiğini test eder."""
    state = TrackState(n_confirm=1, m_persist=2)

    state.update({1})
    assert state.is_visible(1) is True

    first_miss = state.update(set())
    assert first_miss == []
    assert state.is_visible(1) is True

    second_miss = state.update(set())
    assert second_miss == []
    assert state.is_visible(1) is True

    third_miss = state.update(set())
    assert third_miss == [1]
    assert state.is_visible(1) is False

def test_noop_smoother_returns_box_unchanged():
    """NoOpBoxSmoother'ın kutuyu değiştirmeden döndürdüğünü test eder."""
    smoother = NoOpBoxSmoother()
    box = np.array([10, 20, 100, 200], dtype=np.float32)

    result = smoother.smooth(box, track_id=1)

    np.testing.assert_array_equal(result, box)

def test_ema_smoother_blends_current_and_previous_box():
    """EmaBoxSmoother'ın önceki ve mevcut kutuyu doğru şekilde harmanladığını test eder."""
    smoother = EmaBoxSmoother(alpha=0.5)

    first_box = np.array([0, 0, 100, 100], dtype=np.float32)
    second_box = np.array([10, 20, 120, 140], dtype=np.float32)

    smoother.smooth(first_box, track_id=1)
    result = smoother.smooth(second_box, track_id=1)

    expected = np.array([5, 10, 110, 120], dtype=np.float32)

    np.testing.assert_allclose(result, expected)

def test_noop_smoother_drops_last_box():
    smoother = NoOpBoxSmoother()
    box = np.array([10, 20, 100, 200], dtype=np.float32)

    smoother.smooth(box, track_id=1)

    np.testing.assert_array_equal(
        smoother.get_last_box(1),
        box,
    )

    smoother.drop(1)

    assert smoother.get_last_box(1) is None