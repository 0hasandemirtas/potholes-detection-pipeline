from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class VideoCountResult:
    """Bir video için gerçek ve tahmin edilen çukur sayılarını tutar."""

    video_name: str
    expected_count: int
    predicted_count: int

    def __post_init__(self) -> None:
        if self.expected_count < 0:
            raise ValueError(
                "expected_count negatif olamaz"
            )

        if self.predicted_count < 0:
            raise ValueError(
                "predicted_count negatif olamaz"
            )


@dataclass(frozen=True)
class CountMetrics:
    """Birden fazla videodan hesaplanan sayım metriklerini tutar."""

    video_count: int
    total_expected: int
    total_predicted: int
    mean_absolute_error: float
    count_bias: float
    exact_match_rate: float


def calculate_count_metrics(
    results: Sequence[VideoCountResult],
) -> CountMetrics:
    """Video sonuçlarından toplu sayım metriklerini hesaplar."""

    if not results:
        raise ValueError(
            "Metrik hesaplamak için en az bir video sonucu gereklidir"
        )

    video_count = len(results)

    total_expected = sum(
        result.expected_count
        for result in results
    )
    total_predicted = sum(
        result.predicted_count
        for result in results
    )

    absolute_error_sum = sum(
        abs(result.predicted_count - result.expected_count)
        for result in results
    )

    signed_error_sum = sum(
        result.predicted_count - result.expected_count
        for result in results
    )

    exact_match_count = sum(
        result.predicted_count == result.expected_count
        for result in results
    )

    return CountMetrics(
        video_count=video_count,
        total_expected=total_expected,
        total_predicted=total_predicted,
        mean_absolute_error=absolute_error_sum / video_count,
        count_bias=signed_error_sum / video_count,
        exact_match_rate=exact_match_count / video_count,
    )

