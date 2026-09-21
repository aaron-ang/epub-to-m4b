"""Sentences -> missing keys -> batches -> engine -> cache, with gap policy
in between.

Normalize + split each paragraph into sentence-sized strings, attach a
deterministic ``gap_after`` per the punctuation/position rules below, then
(``synthesize_book``) skip whatever is already cached, batch the rest
through ``synth/batching.py``, call the engine, and store each result via
``synth/cache.py`` immediately - so a crash after batch N keeps batches
1..N-1's work safe. Chapter-level assembly is skipped/reused per
``cache.chapter_is_stale`` if nothing in that chapter changed since the last
run.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from epub_to_m4b.audio.assemble import assemble_chapter
from epub_to_m4b.book import AudioClip, Book, Chapter, Paragraph, ParagraphKind, Sentence
from epub_to_m4b.synth import cache
from epub_to_m4b.synth.batching import PendingClip, make_batches
from epub_to_m4b.text import TEXT_PIPELINE_VERSION
from epub_to_m4b.text.normalize import normalize
from epub_to_m4b.text.split import split_paragraph
from epub_to_m4b.tts.base import TTSEngine

# A closing quote/bracket trailing the real terminator, e.g. the `"` in
# `He said "stop."` - look past it to find what actually ended the sentence.
_CLOSERS = "\"')]\u201d\u2019"
_CLAUSE_CHARS = ",;:"


@dataclass(frozen=True, slots=True)
class GapPolicy:
    sentence_end: float = 0.25
    clause: float = 0.10
    paragraph: float = 0.40
    heading: float = 0.80


_DEFAULT_POLICY = GapPolicy()


def sentence_gap(
    text: str,
    kind: ParagraphKind,
    *,
    is_last_in_paragraph: bool,
    is_last_in_chapter: bool,
    policy: GapPolicy = _DEFAULT_POLICY,
) -> float:
    """The gap that follows one sentence, per the punctuation/position rules.

    Heading sentences always get the heading gap. A paragraph-closing
    sentence gets the paragraph gap, unless it also closes the chapter -
    the chapter boundary itself provides the separation, so that case falls
    through to the ordinary punctuation-based gap instead of stacking an
    extra pause.
    """
    if kind is ParagraphKind.HEADING:
        return policy.heading
    if is_last_in_paragraph and not is_last_in_chapter:
        return policy.paragraph
    return policy.clause if _terminal_char(text) in _CLAUSE_CHARS else policy.sentence_end


def _terminal_char(text: str) -> str:
    stripped = text.rstrip()
    while stripped and stripped[-1] in _CLOSERS:
        stripped = stripped[:-1]
    return stripped[-1] if stripped else ""


def chapter_to_sentences(
    chapter: Chapter,
    chapter_index: int,
    *,
    policy: GapPolicy = _DEFAULT_POLICY,
    lang: str = "en",
) -> list[Sentence]:
    flat: list[tuple[str, ParagraphKind, bool]] = []
    for paragraph in chapter.paragraphs:
        normalized = normalize(paragraph.text, lang=lang)
        pieces = split_paragraph(Paragraph(text=normalized, kind=paragraph.kind))
        last = len(pieces) - 1
        for i, piece in enumerate(pieces):
            flat.append((piece, paragraph.kind, i == last))

    last_index = len(flat) - 1
    sentences: list[Sentence] = []
    for i, (text, kind, is_last_in_paragraph) in enumerate(flat):
        gap = sentence_gap(
            text,
            kind,
            is_last_in_paragraph=is_last_in_paragraph,
            is_last_in_chapter=i == last_index,
            policy=policy,
        )
        sentences.append(Sentence(text=text, gap_after=gap, chapter_index=chapter_index))
    return sentences


def batch_sentences(sentences: Sequence[Sentence], max_batch: int) -> list[list[Sentence]]:
    return [list(sentences[i : i + max_batch]) for i in range(0, len(sentences), max_batch)]


def synthesize_chapter(
    sentences: Sequence[Sentence], engine: TTSEngine
) -> list[tuple[Sentence, AudioClip]]:
    pairs: list[tuple[Sentence, AudioClip]] = []
    for batch in batch_sentences(sentences, engine.max_batch):
        clips = engine.synthesize([s.text for s in batch])
        if len(clips) != len(batch):
            raise ValueError(f"engine returned {len(clips)} clips for {len(batch)} texts")
        pairs.extend(zip(batch, clips, strict=True))
    return pairs


@dataclass(frozen=True, slots=True)
class ChapterResult:
    """One chapter's durable audio plus what the caller needs to place its
    sentences on the book-wide VTT timeline. ``cues`` offsets are local to
    the chapter (0 at the chapter's own start) - the caller accumulates a
    running book-wide cursor across chapters, same as it accumulates
    ``duration``."""

    flac_path: Path
    duration: float
    cues: tuple[tuple[str, float, float], ...]


def _clip_keys_and_gaps(
    sentences: Sequence[Sentence],
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    keys = tuple(cache.clip_cache_key(TEXT_PIPELINE_VERSION, s.text) for s in sentences)
    gaps = tuple(s.gap_after for s in sentences)
    return keys, gaps


@dataclass(frozen=True, slots=True)
class _ChapterPlan:
    """One chapter's precomputed sentences/keys/gaps, and whether the cached
    flac for it (if any) is still current."""

    sentences: list[Sentence]
    keys: tuple[str, ...]
    gaps: tuple[float, ...]
    stale: bool


def _plan_chapters(
    book: Book, out_dir: Path, *, policy: GapPolicy, lang: str
) -> list[_ChapterPlan]:
    plans = []
    for idx, chapter in enumerate(book.chapters):
        sentences = chapter_to_sentences(chapter, idx, policy=policy, lang=lang)
        keys, gaps = _clip_keys_and_gaps(sentences)
        stale = cache.chapter_is_stale(
            out_dir, book.source_sha256, idx, clip_keys=keys, gap_after=gaps
        )
        plans.append(_ChapterPlan(sentences=sentences, keys=keys, gaps=gaps, stale=stale))
    return plans


def _resolve_cached_and_pending(
    plans: Sequence[_ChapterPlan],
    *,
    cache_dir: Path,
    engine_fingerprint: str,
) -> tuple[dict[str, AudioClip], list[PendingClip]]:
    """Look up every stale chapter's sentences against the clip cache,
    pooling misses across chapter boundaries before any ``synthesize()``
    call is made."""
    clip_for_key: dict[str, AudioClip] = {}
    pending: list[PendingClip] = []
    for idx, plan in enumerate(plans):
        if not plan.stale:
            continue
        for position, (sentence, key) in enumerate(zip(plan.sentences, plan.keys, strict=True)):
            if key in clip_for_key:
                continue
            hit = cache.load_clip(cache_dir, engine_fingerprint, key)
            if hit is not None:
                clip_for_key[key] = hit
            else:
                pending.append(
                    PendingClip(key=key, chapter_index=idx, position=position, text=sentence.text)
                )
    return clip_for_key, pending


def _synthesize_pending(
    pending: Sequence[PendingClip],
    clip_for_key: dict[str, AudioClip],
    engine: TTSEngine,
    *,
    cache_dir: Path,
    engine_fingerprint: str,
) -> None:
    """Run every pending sentence through the engine, length-sorted and
    capped at ``engine.max_batch`` per call, storing each result to the
    clip cache the moment it comes back - so a crash partway through only
    ever costs the in-flight batch's work."""
    for batch in make_batches(pending, engine.max_batch):
        clips = engine.synthesize([item.text for item in batch])
        if len(clips) != len(batch):
            raise ValueError(f"engine returned {len(clips)} clips for {len(batch)} texts")
        for item, clip in zip(batch, clips, strict=True):
            cache.store_clip(cache_dir, engine_fingerprint, item.key, clip)
            clip_for_key[item.key] = clip


def _reuse_chapter(out_dir: Path, book_sha: str, idx: int, plan: _ChapterPlan) -> ChapterResult:
    manifest = cache.load_chapter_manifest(out_dir, book_sha, idx)
    assert manifest is not None  # plan.stale is False only when this loaded cleanly
    cues = tuple(
        (sentence.text, start, end)
        for sentence, (start, end) in zip(plan.sentences, manifest.offsets, strict=True)
    )
    flac_path = cache.chapter_flac_path(out_dir, book_sha, idx)
    return ChapterResult(flac_path=flac_path, duration=manifest.duration, cues=cues)


def _assemble_chapter_and_store(
    out_dir: Path,
    book_sha: str,
    idx: int,
    plan: _ChapterPlan,
    clip_for_key: dict[str, AudioClip],
    *,
    sample_rate: int,
) -> ChapterResult:
    pairs = [
        (sentence, clip_for_key[key])
        for sentence, key in zip(plan.sentences, plan.keys, strict=True)
    ]
    chapters_dir = cache.chapter_flac_path(out_dir, book_sha, idx).parent
    chapters_dir.mkdir(parents=True, exist_ok=True)
    # Suffix stays ".flac" (not ".flac.tmp") so soundfile can still infer the
    # format from the extension; the leading dot plus mkstemp's random
    # component keeps it out of the way of the real "<idx>.flac" path.
    fd, tmp_name = tempfile.mkstemp(dir=chapters_dir, prefix=f".{idx:04d}.", suffix=".flac")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        offsets, duration = assemble_chapter(pairs, tmp_path, sample_rate=sample_rate)
        manifest = cache.ChapterManifest(
            clip_keys=plan.keys, gap_after=plan.gaps, offsets=tuple(offsets), duration=duration
        )
        cache.store_chapter(out_dir, book_sha, idx, audio_source=tmp_path, manifest=manifest)
    finally:
        tmp_path.unlink(missing_ok=True)  # no-op once store_chapter has moved it

    flac_path = cache.chapter_flac_path(out_dir, book_sha, idx)
    cues = tuple(
        (sentence.text, start, end)
        for sentence, (start, end) in zip(plan.sentences, offsets, strict=True)
    )
    return ChapterResult(flac_path=flac_path, duration=duration, cues=cues)


def synthesize_book(
    book: Book,
    engine: TTSEngine,
    *,
    cache_dir: Path,
    out_dir: Path,
    policy: GapPolicy = _DEFAULT_POLICY,
    lang: str = "en",
    on_chapter_start: Callable[[int, int, str, int], None] | None = None,
) -> list[ChapterResult]:
    """Render every chapter of ``book``, resuming from ``cache_dir`` (clip
    cache, keyed by engine fingerprint + text pipeline version + text) and
    ``out_dir/.work`` (per-chapter flac + manifest) wherever possible.

    Chapters whose sentence set/order is unchanged since the last run (per
    ``cache.chapter_is_stale``) are skipped entirely - no clip lookups, no
    ``audio/assemble.py`` call. For the rest: every sentence's cache key is
    looked up first; hits are loaded from disk, misses are pooled *across
    all stale chapters* and run through ``synth/batching.py`` so a single
    ``synthesize()`` call never mixes wildly different sentence lengths.
    """
    engine_fingerprint = engine.fingerprint()
    book_sha = book.source_sha256

    plans = _plan_chapters(book, out_dir, policy=policy, lang=lang)
    clip_for_key, pending = _resolve_cached_and_pending(
        plans, cache_dir=cache_dir, engine_fingerprint=engine_fingerprint
    )
    _synthesize_pending(
        pending, clip_for_key, engine, cache_dir=cache_dir, engine_fingerprint=engine_fingerprint
    )

    results: list[ChapterResult] = []
    for idx, (chapter, plan) in enumerate(zip(book.chapters, plans, strict=True)):
        if on_chapter_start is not None:
            on_chapter_start(idx, len(plans), chapter.title, len(plan.sentences))
        if plan.stale:
            results.append(
                _assemble_chapter_and_store(
                    out_dir, book_sha, idx, plan, clip_for_key, sample_rate=engine.sample_rate
                )
            )
        else:
            results.append(_reuse_chapter(out_dir, book_sha, idx, plan))
    return results
