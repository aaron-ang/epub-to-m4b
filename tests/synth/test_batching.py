from __future__ import annotations

import pytest

from epub_to_m4b.synth.batching import PendingClip, make_batches


def _item(key: str, chapter_index: int, position: int, length: int) -> PendingClip:
    return PendingClip(key=key, chapter_index=chapter_index, position=position, text="x" * length)


def test_max_batch_cap_is_respected() -> None:
    items = [_item(str(i), 0, i, 10) for i in range(5)]
    batches = make_batches(items, max_batch=2)
    assert [len(b) for b in batches] == [2, 2, 1]


def test_single_item_under_max_batch_is_its_own_batch() -> None:
    items = [_item("only", 0, 0, 10)]
    batches = make_batches(items, max_batch=64)
    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_empty_pool_yields_no_batches() -> None:
    assert make_batches([], max_batch=8) == []


def test_batches_are_length_sorted() -> None:
    items = [
        _item("long", 0, 0, 200),
        _item("short", 0, 1, 5),
        _item("medium", 0, 2, 50),
    ]
    batches = make_batches(items, max_batch=1)
    assert [b[0].key for b in batches] == ["short", "medium", "long"]


def test_a_batch_never_mixes_wildly_different_lengths_when_pool_is_large() -> None:
    # 10 short items and 10 long items, batch size 5: sorting first means
    # no batch straddles the short/long boundary.
    items = [_item(f"short-{i}", 0, i, 5) for i in range(10)]
    items += [_item(f"long-{i}", 0, 10 + i, 200) for i in range(10)]
    batches = make_batches(items, max_batch=5)
    assert len(batches) == 4
    for batch in batches:
        lengths = {len(item.text) for item in batch}
        assert len(lengths) == 1


def test_every_item_dispatched_exactly_once() -> None:
    items = [_item(str(i), i % 3, i, (i % 7) + 1) for i in range(37)]
    batches = make_batches(items, max_batch=8)
    dispatched = [item for batch in batches for item in batch]
    assert len(dispatched) == len(items)
    assert {item.key for item in dispatched} == {item.key for item in items}


def test_chapter_index_and_position_survive_the_round_trip() -> None:
    # sorting by length scrambles order, but each item still carries enough
    # to be zipped back to its originating (chapter, position) unambiguously.
    items = [
        _item("a", chapter_index=2, position=5, length=100),
        _item("b", chapter_index=0, position=1, length=1),
        _item("c", chapter_index=1, position=3, length=50),
    ]
    batches = make_batches(items, max_batch=1)
    by_key = {item.key: (item.chapter_index, item.position) for batch in batches for item in batch}
    assert by_key == {"a": (2, 5), "b": (0, 1), "c": (1, 3)}


def test_batches_stable_across_chapter_boundaries() -> None:
    # items from different chapters can legitimately land in the same batch
    # when their lengths are similar - that's the whole point of batching
    # "across chapters" rather than per-chapter.
    items = [
        _item("ch0-a", 0, 0, 10),
        _item("ch1-a", 1, 0, 11),
        _item("ch0-b", 0, 1, 200),
        _item("ch1-b", 1, 1, 201),
    ]
    batches = make_batches(items, max_batch=2)
    assert len(batches) == 2
    short_batch_keys = {item.key for item in batches[0]}
    assert short_batch_keys == {"ch0-a", "ch1-a"}


def test_max_batch_below_one_raises() -> None:
    with pytest.raises(ValueError, match="max_batch"):
        make_batches([_item("a", 0, 0, 1)], max_batch=0)
