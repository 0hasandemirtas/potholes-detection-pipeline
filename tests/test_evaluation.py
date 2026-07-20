import pytest

from src.evaluation import (
    VideoCountResult,
    calculate_count_metrics,
)


def test_calculate_count_metrics_aggregates_video_results():
    results = [
        VideoCountResult(
            video_name="video_1.mp4",
            expected_count=5,
            predicted_count=4,
        ),
        VideoCountResult(
            video_name="video_2.mp4",
            expected_count=2,
            predicted_count=4,
        ),
        VideoCountResult(
            video_name="video_3.mp4",
            expected_count=3,
            predicted_count=3,
        ),
    ]

    metrics = calculate_count_metrics(results)

    assert metrics.video_count == 3
    assert metrics.total_expected == 10
    assert metrics.total_predicted == 11
    assert metrics.mean_absolute_error == pytest.approx(1.0)
    assert metrics.count_bias == pytest.approx(1 / 3)
    assert metrics.exact_match_rate == pytest.approx(1 / 3)

def test_calculate_count_metrics_rejects_empty_results():
    with pytest.raises(
        ValueError,
        match="en az bir video sonucu",
    ):
        calculate_count_metrics([])


@pytest.mark.parametrize(
    ("expected_count", "predicted_count"),
    [
        (-1, 0),
        (0, -1),
    ],
)

def test_video_count_result_rejects_negative_counts(
    expected_count,
    predicted_count,
):
    with pytest.raises(
        ValueError,
        match="negatif olamaz",
    ):
        VideoCountResult(
            video_name="video.mp4",
            expected_count=expected_count,
            predicted_count=predicted_count,
        )