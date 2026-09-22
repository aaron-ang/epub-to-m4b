"""Content-addressed FLAC clip store, plus a per-chapter reuse manifest.

Two layers, both flat files under ``os.replace``-atomic writes, no database:

``<cache_dir>/clips/<engine_fingerprint[:16]>/<key>.flac``
    One sentence's synthesized audio. ``key`` is
    ``sha256(text_pipeline_version + "\\0" + text)[:32]`` - the engine
    fingerprint is a *directory* partition, not folded into the hash, so
    switching engine/voice/model starts a fresh directory while the old one
    stays untouched and reusable if the run switches back. The key
    deliberately excludes gap policy: silence is added at assembly time
    (``audio/assemble.py``), never baked into a clip, so retuning
    ``GapPolicy`` must never invalidate a cached clip.

``<out_dir>/.work/<book_sha256[:16]>/chapters/<idx>.flac`` (+ sidecar ``.json``)
    One chapter's fully assembled audio, plus a manifest recording what went
    into it (engine fingerprint, sample rate, clip keys, gaps, sentence
    offsets, duration) so a rerun can tell whether the chapter is still
    current without re-running ``audio/assemble.py`` or touching ffmpeg.

Any cache hit that fails to open, or is zero-length, counts as a miss: the
bad file is deleted and the caller re-synthesizes/re-assembles rather than
letting a corrupt entry wedge the pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from epub_to_m4b.book import AudioClip

CLIP_KEY_LENGTH = 32
FINGERPRINT_DIR_LENGTH = 16


def _read_umask() -> int:
    # os.umask can only be read by setting it; do the toggle once at import
    # so concurrent writer threads never see a transiently zeroed umask.
    old = os.umask(0)
    os.umask(old)
    return old


# What a plain ``open(path, "wb")`` would produce - mkstemp always uses 0600,
# which would otherwise leak onto every landed clip, chapter and m4b.
_LANDED_FILE_MODE = 0o666 & ~_read_umask()

Offsets = tuple[tuple[float, float], ...]


def clip_cache_key(pipeline_version: str, text: str) -> str:
    """``sha256(text_pipeline_version, text)[:32]`` - never includes the engine
    fingerprint or gap policy; those are a directory partition and an
    assembly-time concern, respectively (see module docstring)."""
    digest = hashlib.sha256()
    digest.update(pipeline_version.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:CLIP_KEY_LENGTH]


def _fingerprint_dir(cache_dir: Path, engine_fingerprint: str) -> Path:
    return cache_dir / "clips" / engine_fingerprint[:FINGERPRINT_DIR_LENGTH]


def clip_path(cache_dir: Path, engine_fingerprint: str, key: str) -> Path:
    return _fingerprint_dir(cache_dir, engine_fingerprint) / f"{key}.flac"


def atomic_replace(
    final_path: Path, write_body: Callable[[Path], None], *, suffix: str = ".tmp"
) -> None:
    """Write via a temp file in the same directory, then ``os.replace`` into
    place - a crash mid-write can only ever leave the stale (or absent) final
    file, never a truncated one at the real path.

    ``suffix`` matters when the writer infers a container format from the
    extension (soundfile, ffmpeg): pass the real one, e.g. ``".flac"``.
    """
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=final_path.parent, prefix=f".{final_path.name}.", suffix=suffix
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        write_body(tmp_path)
        os.chmod(tmp_path, _LANDED_FILE_MODE)
        os.replace(tmp_path, final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _readable_and_nonempty(path: Path) -> bool:
    try:
        if path.stat().st_size == 0:
            return False
        sf.info(path)
    except OSError, sf.SoundFileError:
        return False
    return True


def has_clip(cache_dir: Path, engine_fingerprint: str, key: str) -> bool:
    """Cheap presence check (header only, no decode) for planning which
    sentences still need the engine. A corrupt or zero-length entry is
    deleted here so it doesn't need to be re-detected on every lookup."""
    path = clip_path(cache_dir, engine_fingerprint, key)
    if not path.is_file():
        return False
    if not _readable_and_nonempty(path):
        path.unlink(missing_ok=True)
        return False
    return True


def load_clip(cache_dir: Path, engine_fingerprint: str, key: str) -> AudioClip | None:
    """Return the cached clip, or ``None`` on a miss (including a corrupt or
    zero-length file, which is deleted rather than left to fail again)."""
    path = clip_path(cache_dir, engine_fingerprint, key)
    if not path.is_file():
        return None
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except OSError, sf.SoundFileError:
        path.unlink(missing_ok=True)
        return None
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:  # defensive: clips are always written mono
        samples = samples[:, 0]
    return AudioClip(samples=samples, sample_rate=int(sample_rate))


def store_clip(cache_dir: Path, engine_fingerprint: str, key: str, clip: AudioClip) -> None:
    path = clip_path(cache_dir, engine_fingerprint, key)

    def write_body(tmp_path: Path) -> None:
        sf.write(tmp_path, clip.samples, clip.sample_rate, format="FLAC")

    atomic_replace(path, write_body)


@dataclass(frozen=True, slots=True)
class ChapterManifest:
    """Everything needed to decide whether a chapter's cached flac is still
    current, and (if it is) to reconstruct per-sentence offsets without
    touching ffmpeg or re-running ``audio/assemble.py``.

    ``engine_fingerprint`` and ``sample_rate`` are recorded because the
    assembled flac bakes both in: the same sentences rendered by a different
    voice, or written at a different rate, is a different chapter even
    though every clip key and gap matches."""

    engine_fingerprint: str
    sample_rate: int
    clip_keys: tuple[str, ...]
    gap_after: tuple[float, ...]
    offsets: Offsets
    duration: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "engine_fingerprint": self.engine_fingerprint,
                "sample_rate": self.sample_rate,
                "clip_keys": list(self.clip_keys),
                "gap_after": list(self.gap_after),
                "offsets": [list(pair) for pair in self.offsets],
                "duration": self.duration,
            }
        )

    @staticmethod
    def from_json(text: str) -> ChapterManifest:
        data = json.loads(text)
        return ChapterManifest(
            engine_fingerprint=str(data["engine_fingerprint"]),
            sample_rate=int(data["sample_rate"]),
            clip_keys=tuple(data["clip_keys"]),
            gap_after=tuple(float(g) for g in data["gap_after"]),
            offsets=tuple((float(start), float(end)) for start, end in data["offsets"]),
            duration=float(data["duration"]),
        )

    def describes(
        self,
        *,
        engine_fingerprint: str,
        sample_rate: int,
        clip_keys: tuple[str, ...],
        gap_after: tuple[float, ...],
    ) -> bool:
        return (
            self.engine_fingerprint == engine_fingerprint
            and self.sample_rate == sample_rate
            and self.clip_keys == clip_keys
            and self.gap_after == gap_after
        )


def book_work_dir(out_dir: Path, book_sha256: str) -> Path:
    return out_dir / ".work" / book_sha256[:FINGERPRINT_DIR_LENGTH]


def chapter_flac_path(out_dir: Path, book_sha256: str, chapter_index: int) -> Path:
    return book_work_dir(out_dir, book_sha256) / "chapters" / f"{chapter_index:04d}.flac"


def chapter_manifest_path(out_dir: Path, book_sha256: str, chapter_index: int) -> Path:
    return book_work_dir(out_dir, book_sha256) / "chapters" / f"{chapter_index:04d}.json"


def load_chapter_manifest(
    out_dir: Path, book_sha256: str, chapter_index: int
) -> ChapterManifest | None:
    """A missing or unparseable manifest always means "treat as stale",
    never "trust a partial write" - there is no distinction between "never
    written" and "corrupt" here, both are just misses."""
    path = chapter_manifest_path(out_dir, book_sha256, chapter_index)
    if not path.is_file():
        return None
    try:
        return ChapterManifest.from_json(path.read_text(encoding="utf-8"))
    except OSError, ValueError, KeyError, TypeError:
        return None


def current_chapter_manifest(
    out_dir: Path,
    book_sha256: str,
    chapter_index: int,
    *,
    engine_fingerprint: str,
    sample_rate: int,
    clip_keys: tuple[str, ...],
    gap_after: tuple[float, ...],
) -> ChapterManifest | None:
    """The stored manifest if it describes exactly this chapter as it would
    be rendered now *and* the flac it describes still opens cleanly;
    ``None`` means the chapter must be (re)assembled."""
    manifest = load_chapter_manifest(out_dir, book_sha256, chapter_index)
    if manifest is None or not manifest.describes(
        engine_fingerprint=engine_fingerprint,
        sample_rate=sample_rate,
        clip_keys=clip_keys,
        gap_after=gap_after,
    ):
        return None
    flac_path = chapter_flac_path(out_dir, book_sha256, chapter_index)
    if not flac_path.is_file() or not _readable_and_nonempty(flac_path):
        return None
    return manifest


def store_chapter(
    out_dir: Path,
    book_sha256: str,
    chapter_index: int,
    *,
    engine_fingerprint: str,
    sample_rate: int,
    clip_keys: tuple[str, ...],
    gap_after: tuple[float, ...],
    write_audio: Callable[[Path], tuple[Offsets, float]],
) -> ChapterManifest:
    """Render the chapter flac through ``write_audio`` (which writes to the
    path it is given and returns the per-sentence offsets and total
    duration), land it atomically, then record its manifest.

    The manifest is only written after the flac is durably in place, so a
    crash between the two steps leaves, at worst, a flac with no matching
    manifest - which ``current_chapter_manifest`` (no manifest found)
    already treats as stale, never as a false "trust it" hit.
    """
    flac_path = chapter_flac_path(out_dir, book_sha256, chapter_index)
    rendered: list[tuple[Offsets, float]] = []

    def write_flac(tmp_path: Path) -> None:
        rendered.append(write_audio(tmp_path))

    atomic_replace(flac_path, write_flac, suffix=".flac")
    offsets, duration = rendered[0]
    manifest = ChapterManifest(
        engine_fingerprint=engine_fingerprint,
        sample_rate=sample_rate,
        clip_keys=clip_keys,
        gap_after=gap_after,
        offsets=offsets,
        duration=duration,
    )

    def write_manifest(tmp_path: Path) -> None:
        tmp_path.write_text(manifest.to_json(), encoding="utf-8")

    atomic_replace(chapter_manifest_path(out_dir, book_sha256, chapter_index), write_manifest)
    return manifest
