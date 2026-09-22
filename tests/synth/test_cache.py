from __future__ import annotations

import inspect
import os
import stat
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from epub_to_m4b.book import AudioClip
from epub_to_m4b.synth import cache

_RATE = 8000
_FP = "engine-fp"
_SHA = "book123"


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
    cache.store_clip(tmp_path, _FP, key, clip)
    loaded = cache.load_clip(tmp_path, _FP, key)
    assert loaded is not None
    assert loaded.sample_rate == clip.sample_rate
    assert np.allclose(loaded.samples, clip.samples, atol=1e-4)


def test_load_missing_clip_is_a_miss(tmp_path: Path) -> None:
    assert cache.load_clip(tmp_path, _FP, "0" * 32) is None


def test_has_clip_tracks_presence_without_decoding(tmp_path: Path) -> None:
    key = cache.clip_cache_key("v1", "hello")
    assert not cache.has_clip(tmp_path, _FP, key)
    cache.store_clip(tmp_path, _FP, key, _clip())
    assert cache.has_clip(tmp_path, _FP, key)


def test_store_clip_writes_via_temp_file_then_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = cache.clip_cache_key("v1", "hello")
    seen_tmp_names: list[str] = []
    real_replace = cache.os.replace

    def spy_replace(src: object, dst: object) -> None:
        seen_tmp_names.append(str(src))
        real_replace(src, dst)

    monkeypatch.setattr(cache.os, "replace", spy_replace)
    cache.store_clip(tmp_path, _FP, key, _clip())

    assert len(seen_tmp_names) == 1
    # the temp file lived in the same directory as the final path (a
    # prerequisite for os.replace to be atomic on the same filesystem) and
    # was not itself named like the final key.
    final_path = cache.clip_path(tmp_path, _FP, key)
    assert Path(seen_tmp_names[0]).parent == final_path.parent
    assert seen_tmp_names[0] != str(final_path)
    # and nothing is left behind afterwards - only the final file exists.
    assert list(final_path.parent.iterdir()) == [final_path]


def test_store_clip_crash_mid_write_leaves_no_partial_file_at_final_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = cache.clip_cache_key("v1", "hello")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(sf, "write", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        cache.store_clip(tmp_path, _FP, key, _clip())

    final_path = cache.clip_path(tmp_path, _FP, key)
    assert not final_path.exists()
    # the temp file was cleaned up too, not left as debris.
    assert not list(final_path.parent.iterdir())


_BAD_PAYLOADS = [b"", b"this is not a flac file, just garbage bytes", b"\x00" * 4]


@pytest.mark.parametrize("payload", _BAD_PAYLOADS)
def test_bad_clip_file_is_a_miss_for_both_lookups_and_gets_deleted(
    tmp_path: Path, payload: bytes
) -> None:
    key = cache.clip_cache_key("v1", "hello")
    path = cache.clip_path(tmp_path, _FP, key)
    path.parent.mkdir(parents=True)

    path.write_bytes(payload)
    assert cache.load_clip(tmp_path, _FP, key) is None  # never raises
    assert not path.exists()

    path.write_bytes(payload)
    assert not cache.has_clip(tmp_path, _FP, key)
    assert not path.exists()


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


# --- permissions -----------------------------------------------------------


def _current_umask() -> int:
    old = os.umask(0)
    os.umask(old)
    return old


def _assert_umask_derived_mode(path: Path) -> None:
    umask = _current_umask()
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o666 & ~umask
    if umask != 0o077:
        # mkstemp's 0600 must not leak through to the landed file.
        assert mode != 0o600


def test_atomic_replace_lands_file_with_umask_derived_mode(tmp_path: Path) -> None:
    final = tmp_path / "out.bin"
    cache.atomic_replace(final, lambda p: p.write_bytes(b"data"))
    _assert_umask_derived_mode(final)


def test_store_clip_lands_flac_with_umask_derived_mode(tmp_path: Path) -> None:
    key = cache.clip_cache_key("v1", "hello")
    cache.store_clip(tmp_path, _FP, key, _clip())
    _assert_umask_derived_mode(cache.clip_path(tmp_path, _FP, key))


# --- chapter manifest / staleness -----------------------------------------

_KEYS = ("k1", "k2")
_GAPS = (0.1, 0.1)


def _write_zeros(tmp_path: Path) -> tuple[cache.Offsets, float]:
    sf.write(tmp_path, np.zeros(_RATE // 10, dtype=np.float32), _RATE)
    return ((0.0, 0.05), (0.05, 0.1)), 0.1


def _store(tmp_path: Path, *, fp: str = _FP, rate: int = _RATE) -> cache.ChapterManifest:
    return cache.store_chapter(
        tmp_path,
        _SHA,
        0,
        engine_fingerprint=fp,
        sample_rate=rate,
        clip_keys=_KEYS,
        gap_after=_GAPS,
        write_audio=_write_zeros,
    )


def _current(
    tmp_path: Path,
    *,
    fp: str = _FP,
    rate: int = _RATE,
    keys: tuple[str, ...] = _KEYS,
    gaps: tuple[float, ...] = _GAPS,
) -> cache.ChapterManifest | None:
    return cache.current_chapter_manifest(
        tmp_path, _SHA, 0, engine_fingerprint=fp, sample_rate=rate, clip_keys=keys, gap_after=gaps
    )


def test_chapter_never_written_is_not_current(tmp_path: Path) -> None:
    assert _current(tmp_path) is None


def test_chapter_manifest_round_trips_through_json() -> None:
    manifest = cache.ChapterManifest(
        engine_fingerprint=_FP,
        sample_rate=_RATE,
        clip_keys=("a", "b"),
        gap_after=(0.1, 0.2),
        offsets=((0.0, 1.0), (1.1, 2.0)),
        duration=2.0,
    )
    assert cache.ChapterManifest.from_json(manifest.to_json()) == manifest


def test_store_chapter_returns_what_write_audio_measured_and_lands_the_flac(
    tmp_path: Path,
) -> None:
    manifest = _store(tmp_path)
    assert manifest.offsets == ((0.0, 0.05), (0.05, 0.1))
    assert manifest.duration == 0.1
    assert manifest.engine_fingerprint == _FP
    assert manifest.sample_rate == _RATE
    assert cache.chapter_flac_path(tmp_path, _SHA, 0).is_file()
    assert cache.load_chapter_manifest(tmp_path, _SHA, 0) == manifest


def test_chapter_current_when_manifest_matches_everything(tmp_path: Path) -> None:
    stored = _store(tmp_path)
    assert _current(tmp_path) == stored


def test_chapter_not_current_when_clip_keys_changed(tmp_path: Path) -> None:
    _store(tmp_path)
    assert _current(tmp_path, keys=("k1", "k3")) is None


def test_chapter_not_current_when_gap_policy_changed(tmp_path: Path) -> None:
    # gap policy is not part of the *clip* cache key, but it does change
    # what the assembled chapter sounds like, so the chapter-level manifest
    # still needs to catch it.
    _store(tmp_path)
    assert _current(tmp_path, gaps=(0.9, 0.9)) is None


def test_chapter_not_current_when_engine_fingerprint_changed(tmp_path: Path) -> None:
    # Same sentences, same gaps, different voice: the assembled audio is a
    # different chapter even though every clip key matches.
    _store(tmp_path, fp="engine-a")
    assert _current(tmp_path, fp="engine-a") is not None
    assert _current(tmp_path, fp="engine-b") is None


def test_chapter_not_current_when_sample_rate_changed(tmp_path: Path) -> None:
    _store(tmp_path, rate=8000)
    assert _current(tmp_path, rate=16000) is None


@pytest.mark.parametrize("damage", ["unlink", b"garbage, not a flac file", b""])
def test_chapter_not_current_when_flac_is_missing_or_unreadable(
    tmp_path: Path, damage: str | bytes
) -> None:
    _store(tmp_path)
    flac_path = cache.chapter_flac_path(tmp_path, _SHA, 0)
    if damage == "unlink":
        flac_path.unlink()
    else:
        assert isinstance(damage, bytes)
        flac_path.write_bytes(damage)
    assert _current(tmp_path) is None


def test_load_chapter_manifest_missing_returns_none(tmp_path: Path) -> None:
    assert cache.load_chapter_manifest(tmp_path, _SHA, 0) is None


@pytest.mark.parametrize("text", ["{not valid json", '{"clip_keys": ["a"]}', "[]"])
def test_load_chapter_manifest_unusable_json_returns_none_not_raise(
    tmp_path: Path, text: str
) -> None:
    path = cache.chapter_manifest_path(tmp_path, _SHA, 0)
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding="utf-8")
    assert cache.load_chapter_manifest(tmp_path, _SHA, 0) is None


def test_store_chapter_write_audio_crash_leaves_no_flac_and_no_manifest(tmp_path: Path) -> None:
    def boom(_tmp_path: Path) -> tuple[cache.Offsets, float]:
        raise RuntimeError("simulated crash mid-assembly")

    with pytest.raises(RuntimeError, match="simulated crash"):
        cache.store_chapter(
            tmp_path,
            _SHA,
            0,
            engine_fingerprint=_FP,
            sample_rate=_RATE,
            clip_keys=_KEYS,
            gap_after=_GAPS,
            write_audio=boom,
        )
    chapters_dir = cache.chapter_flac_path(tmp_path, _SHA, 0).parent
    assert not list(chapters_dir.iterdir())  # no flac, no manifest, no temp debris


def test_store_chapter_writes_manifest_only_after_flac_is_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_atomic_replace = cache.atomic_replace

    def spy_atomic_replace(final_path: Path, write_body: object, **kwargs: object) -> None:
        if final_path.suffix == ".json":
            raise RuntimeError("simulated crash before manifest lands")
        real_atomic_replace(final_path, write_body, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cache, "atomic_replace", spy_atomic_replace)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _store(tmp_path)

    # the flac made it durably to disk...
    assert cache.chapter_flac_path(tmp_path, _SHA, 0).is_file()
    # ...but the manifest never did, so a rerun must treat this chapter as
    # stale rather than trusting a partial write.
    assert not cache.chapter_manifest_path(tmp_path, _SHA, 0).is_file()
    assert _current(tmp_path) is None
