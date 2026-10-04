from __future__ import annotations

from epub_to_m4b.progress import PercentSteps


def _due(total: float, values: list[float]) -> list[float]:
    steps = PercentSteps(total)
    return [v for v in values if steps.due(v)]


def test_due_once_per_ten_percent_step() -> None:
    assert _due(20, list(range(1, 21))) == [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def test_every_call_due_when_total_under_ten() -> None:
    assert _due(3, [1, 2, 3]) == [1, 2, 3]


def test_jump_over_several_steps_is_due_once() -> None:
    assert _due(100, [5, 47, 48, 100]) == [47, 100]


def test_end_is_due_once() -> None:
    assert _due(10, [10, 10, 12]) == [10]


def test_zero_total_is_due_once() -> None:
    assert _due(0, [0, 0]) == [0]
