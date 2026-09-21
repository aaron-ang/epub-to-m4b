from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from epub_to_m4b.book import AudioClip
from epub_to_m4b.synth import cache

_RATE = 8000


def _clip(seconds: float = 0.1, value: float = 0.5) -> AudioClip:
    n = max(1, int(seconds * _RATE))
    return AudioClip(samples=np.full(n, value, dtype=np.float32), sample_rate=_RATE)


# --- key stability -----------------------------------------------------


def test_clip_cache_key_is_stable_for_same_inputs() -> None:
    a = cache.clip_cache_key("v1", "Hello, world.")
    b = cache.clip_cache_key("v1", "Hello, world.")
    assert a == b
    assert len(a) == cache.CLIP_KEY_LENGTH


def test_clip_cache_key_changes_with_text() -> None:
    assert cache.clip_cache_key("v1", "a") != cache.clip_cache_key("v1", "b")


def test_clip_cache_key_changes_with_pipeline_version() -> None:
    assert cache.clip_cache_key("v1", "same text") != cache.clip_cache_key("v2", "same text")


def test_clip_cache_key_never_depends_on_engine_fingerprint() -> None:
    # the fingerprint is a directory partition, not folded into the hash -
    # the key function doesn't even take it as an argument.
    assert "engine_fingerprint" not in inspect.signature(cache.clip_cache_key).parameters


# --- clip store/load -----------------------------------------------------


def test_store_then_load_round_trips(tmp_path: Path) -> None:
    clip = _clip()
    key = cache.clip_cache_key("v1", "hello")
    cache.store_clip(tmp_path, "engine-fp", key, clip)
    loaded = cache.load_clip(tmp_path, "engine-fp", key)
    assert loaded is not None
    assert loaded.sample_rate == clip.sample_rate
    assert np.allclose(loaded.samples, clip.samples, atol=1e-4)


def test_load_missing_clip_is_a_miss(tmp_path: Path) -> None:
    assert cache.load_clip(tmp_path, "engine-fp", "0" * 32) is None


def test_store_clip_writes_via_temp_file_then_replace(tmp_path: Path, monkeypatch) -> None:
    key = cache.clip_cache_key("v1", "hello")
    seen_tmp_names: list[str] = []
    real_replace = cache.os.replace

    def spy_replace(src: object, dst: object) -> None:
        seen_tmp_names.append(str(src))
        real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", spy_replace)
    cache.store_clip(tmp_path, "engine-fp", key, _clip())

    assert len(seen_tmp_names) == 1
    # the temp file lived in the same directory as the final path (a
    # prerequisite for os.replace to be atomic on the same filesystem) and
    # was not itself named like the final key.
    final_path = cache.clip_path(tmp_path, "engine-fp", key)
    assert Path(seen_tmp_names[0]).parent == final_path.parent
    assert seen_tmp_names[0] != str(final_path)
    # and nothing is left behind afterwards - only the final file exists.
    assert list(final_path.parent.iterdir()) == [final_path]


def test_store_clip_crash_mid_write_leaves_no_partial_file_at_final_path(
    tmp_path: Path, monkeypatch
) -> None:
    key = cache.clip_cache_key("v1", "hello")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(sf, "write", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        cache.store_clip(tmp_path, "engine-fp", key, _clip())

    final_path = cache.clip_path(tmp_path, "engine-fp", key)
    assert not final_path.exists()
    # the temp file was cleaned up too, not left as debris.
    assert not list(final_path.parent.iterdir())


def test_load_clip_zero_length_file_counts_as_miss_and_is_deleted(tmp_path: Path) -> None:
    key = cache.clip_cache_key("v1", "hello")
    path = cache.clip_path(tmp_path, "engine-fp", key)
    path.parent.mkdir(parents=True)
    path.touch()
    assert cache.load_clip(tmp_path, "engine-fp", key) is None
    assert not path.exists()


def test_load_clip_corrupt_file_counts_as_miss_and_is_deleted(tmp_path: Path) -> None:
    key = cache.clip_cache_key("v1", "hello")
    path = cache.clip_path(tmp_path, "engine-fp", key)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"this is not a flac file, just garbage bytes")
    assert cache.load_clip(tmp_path, "engine-fp", key) is None
    assert not path.exists()


def test_corrupt_cache_entry_does_not_raise(tmp_path: Path) -> None:
    # the whole point of "corrupt counts as a miss" is that callers never
    # need a try/except around load_clip.
    key = cache.clip_cache_key("v1", "hello")
    path = cache.clip_path(tmp_path, "engine-fp", key)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\x00" * 4)
    result = cache.load_clip(tmp_path, "engine-fp", key)
    assert result is None


def test_engine_fingerprint_change_uses_a_different_directory(tmp_path: Path) -> None:
    key = cache.clip_cache_key("v1", "hello")
    cache.store_clip(tmp_path, "engine-a", key, _clip())
    # the old entry is untouched and still loadable under its own fingerprint...
    assert cache.load_clip(tmp_path, "engine-a", key) is not None
    # ...but a different fingerprint has never heard of it.
    assert cache.load_clip(tmp_path, "engine-b", key) is None
    path_a = cache.clip_path(tmp_path, "engine-a", key)
    path_b = cache.clip_path(tmp_path, "engine-b", key)
    assert path_a.parent != path_b.parent


def test_clip_path_partitions_by_first_16_chars_of_fingerprint(tmp_path: Path) -> None:
    long_fp = "x" * 40
    other_fp = "x" * 16 + "y" * 24  # same first 16 chars, different tail
    key = cache.clip_cache_key("v1", "hello")
    assert cache.clip_path(tmp_path, long_fp, key) == cache.clip_path(tmp_path, other_fp, key)


# --- chapter manifest / staleness -----------------------------------------


def _manifest(keys: tuple[str, ...], gaps: tuple[float, ...]) -> cache.ChapterManifest:
    return cache.ChapterManifest(
        clip_keys=keys, gap_after=gaps, offsets=tuple((0.0, 1.0) for _ in keys), duration=1.0
    )


def test_chapter_is_stale_when_never_written(tmp_path: Path) -> None:
    assert cache.chapter_is_stale(tmp_path, "book123", 0, clip_keys=("a",), gap_after=(0.1,))


def test_chapter_manifest_round_trips_through_json() -> None:
    manifest = cache.ChapterManifest(
        clip_keys=("a", "b"), gap_after=(0.1, 0.2), offsets=((0.0, 1.0), (1.1, 2.0)), duration=2.0
    )
    assert cache.ChapterManifest.from_json(manifest.to_json()) == manifest


def _write_chapter(tmp_path: Path, book_sha: str, idx: int, keys: tuple[str, ...]) -> Path:
    gaps = tuple(0.1 for _ in keys)
    scratch = tmp_path / "scratch.flac"
    sf.write(scratch, np.zeros(_RATE // 10, dtype=np.float32), _RATE)
    cache.store_chapter(
        tmp_path, book_sha, idx, audio_source=scratch, manifest=_manifest(keys, gaps)
    )
    return cache.chapter_flac_path(tmp_path, book_sha, idx)


def test_chapter_not_stale_when_manifest_matches_current_keys_and_gaps(tmp_path: Path) -> None:
    keys = ("k1", "k2")
    _write_chapter(tmp_path, "book123", 0, keys)
    assert not cache.chapter_is_stale(tmp_path, "book123", 0, clip_keys=keys, gap_after=(0.1, 0.1))


def test_chapter_stale_when_clip_keys_changed(tmp_path: Path) -> None:
    _write_chapter(tmp_path, "book123", 0, ("k1", "k2"))
    assert cache.chapter_is_stale(
        tmp_path, "book123", 0, clip_keys=("k1", "k3"), gap_after=(0.1, 0.1)
    )


def test_chapter_stale_when_gap_policy_changed(tmp_path: Path) -> None:
    # gap policy is not part of the *clip* cache key, but it does change
    # what the assembled chapter sounds like, so the chapter-level manifest
    # still needs to catch it.
    keys = ("k1", "k2")
    _write_chapter(tmp_path, "book123", 0, keys)
    assert cache.chapter_is_stale(tmp_path, "book123", 0, clip_keys=keys, gap_after=(0.9, 0.9))


def test_chapter_stale_when_flac_missing_but_manifest_present(tmp_path: Path) -> None:
    keys = ("k1",)
    flac_path = _write_chapter(tmp_path, "book123", 0, keys)
    flac_path.unlink()
    assert cache.chapter_is_stale(tmp_path, "book123", 0, clip_keys=keys, gap_after=(0.1,))


def test_chapter_stale_when_flac_is_corrupt(tmp_path: Path) -> None:
    keys = ("k1",)
    flac_path = _write_chapter(tmp_path, "book123", 0, keys)
    flac_path.write_bytes(b"garbage, not a flac file")
    assert cache.chapter_is_stale(tmp_path, "book123", 0, clip_keys=keys, gap_after=(0.1,))


def test_chapter_stale_when_flac_is_zero_length(tmp_path: Path) -> None:
    keys = ("k1",)
    flac_path = _write_chapter(tmp_path, "book123", 0, keys)
    flac_path.write_bytes(b"")
    assert cache.chapter_is_stale(tmp_path, "book123", 0, clip_keys=keys, gap_after=(0.1,))


def test_load_chapter_manifest_missing_returns_none(tmp_path: Path) -> None:
    assert cache.load_chapter_manifest(tmp_path, "book123", 0) is None


def test_load_chapter_manifest_corrupt_json_returns_none_not_raise(tmp_path: Path) -> None:
    path = cache.chapter_manifest_path(tmp_path, "book123", 0)
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert cache.load_chapter_manifest(tmp_path, "book123", 0) is None


def test_store_chapter_writes_manifest_only_after_flac_is_in_place(
    tmp_path: Path, monkeypatch
) -> None:
    keys = ("k1",)
    scratch = tmp_path / "scratch.flac"
    sf.write(scratch, np.zeros(_RATE // 10, dtype=np.float32), _RATE)

    real_atomic_replace = cache._atomic_replace
    calls: list[str] = []

    def spy_atomic_replace(final_path: Path, write_body: object) -> None:
        calls.append(str(final_path))
        if final_path.suffix == ".json":
            raise RuntimeError("simulated crash before manifest lands")
        real_atomic_replace(final_path, write_body)  # type: ignore[arg-type]

    monkeypatch.setattr(cache, "_atomic_replace", spy_atomic_replace)
    with pytest.raises(RuntimeError, match="simulated crash"):
        cache.store_chapter(
            tmp_path, "book123", 0, audio_source=scratch, manifest=_manifest(keys, (0.1,))
        )

    flac_path = cache.chapter_flac_path(tmp_path, "book123", 0)
    manifest_path = cache.chapter_manifest_path(tmp_path, "book123", 0)
    # the flac made it durably to disk...
    assert flac_path.is_file()
    # ...but the manifest never did, so a rerun must treat this chapter as
    # stale rather than trusting a partial write.
    assert not manifest_path.is_file()
    assert cache.chapter_is_stale(tmp_path, "book123", 0, clip_keys=keys, gap_after=(0.1,))
