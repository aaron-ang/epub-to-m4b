from __future__ import annotations

import pytest

from epub_to_m4b.synth.batching import PendingClip, make_batches


def _item(key: str, length: int) -> PendingClip:
    return PendingClip(key=key, text="x" * length)


def test_max_batch_cap_is_respected() -> None:
    items = [_item(str(i), 10) for i in range(5)]
    batches = make_batches(items, max_batch=2)
    assert [len(b) for b in batches] == [2, 2, 1]


def test_single_item_under_max_batch_is_its_own_batch() -> None:
    batches = make_batches([_item("only", 10)], max_batch=64)
    assert [len(b) for b in batches] == [1]


def test_empty_pool_yields_no_batches() -> None:
    assert make_batches([], max_batch=8) == []


def test_batches_are_length_sorted() -> None:
    items = [_item("long", 200), _item("short", 5), _item("medium", 50)]
    batches = make_batches(items, max_batch=1)
    assert [b[0].key for b in batches] == ["short", "medium", "long"]


def test_a_batch_never_mixes_wildly_different_lengths_when_pool_is_large() -> None:
    # 10 short items and 10 long items, batch size 5: sorting first means
    # no batch straddles the short/long boundary.
    items = [_item(f"short-{i}", 5) for i in range(10)]
    items += [_item(f"long-{i}", 200) for i in range(10)]
    batches = make_batches(items, max_batch=5)
    assert len(batches) == 4
    for batch in batches:
        assert len({len(item.text) for item in batch}) == 1


def test_every_item_dispatched_exactly_once() -> None:
    items = [_item(str(i), (i % 7) + 1) for i in range(37)]
    dispatched = [item for batch in make_batches(items, max_batch=8) for item in batch]
    assert len(dispatched) == len(items)
    assert {item.key for item in dispatched} == {item.key for item in items}


def test_similar_lengths_share_a_batch_regardless_of_origin() -> None:
    # The pool is book-wide: sentences from different chapters land in the
    # same call when their lengths are close.
    items = [_item("ch0-a", 10), _item("ch1-a", 11), _item("ch0-b", 200), _item("ch1-b", 201)]
    batches = make_batches(items, max_batch=2)
    assert [{item.key for item in b} for b in batches] == [{"ch0-a", "ch1-a"}, {"ch0-b", "ch1-b"}]


def test_max_batch_below_one_raises() -> None:
    with pytest.raises(ValueError, match="max_batch"):
        make_batches([_item("a", 1)], max_batch=0)
