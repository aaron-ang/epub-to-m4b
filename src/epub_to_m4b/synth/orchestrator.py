"""Sentences -> missing keys -> batches -> engine -> cache, with gap policy
in between.

Normalize + split each paragraph into sentence-sized strings, attach a
deterministic ``gap_after`` per the punctuation/position rules below, then
(``synthesize_book``) skip whatever is already cached, batch the rest
through ``synth/batching.py``, call the engine, and store each result via
``synth/cache.py`` immediately - so a crash after batch N keeps batches
1..N-1's work safe. Chapter-level assembly is skipped/reused per
``cache.current_chapter_manifest`` if nothing in that chapter changed since
the last run.

Memory stays bounded to one chapter: planning only checks clip *presence*,
and each chapter's clips (cached or freshly synthesized - both live in the
cache by then) are loaded from disk just before that chapter is assembled.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from epub_to_m4b.audio.assemble import assemble_chapter
from epub_to_m4b.book import AudioClip, Book, Paragraph, ParagraphKind, Sentence
from epub_to_m4b.errors import EpubToM4bError
from epub_to_m4b.synth import cache
from epub_to_m4b.synth.batching import PendingClip, make_batches
from epub_to_m4b.text.normalize import normalize_all
from epub_to_m4b.text.split import CLOSERS, split_paragraph
from epub_to_m4b.tts.base import TTSEngine


class SynthesisError(EpubToM4bError):
    """The cache lost a clip between synthesis and chapter assembly; a rerun
    synthesizes it again."""


_CLAUSE_CHARS = ",;:"


@dataclass(frozen=True, slots=True)
class GapPolicy:
    """Seconds of silence inserted after a sentence when a chapter is
    assembled. Changing them re-assembles chapters from cached clips."""

    # Breath between sentences ending in . ! ?
    sentence_end: float = 0.25
    # Shorter pause after a clip ending in , ; : - a clause of the same
    # thought continues in the next clip.
    clause: float = 0.10
    # Longer pause after the last sentence of a paragraph (unless it also
    # closes the chapter, where the chapter boundary provides the break).
    paragraph: float = 0.40
    # Pause after a chapter or section title.
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
    # Look past a trailing closing quote/bracket to find what actually ended
    # the sentence, e.g. the `.` in `He said "stop."`.
    while stripped and stripped[-1] in CLOSERS:
        stripped = stripped[:-1]
    return stripped[-1] if stripped else ""


def book_to_sentences(
    book: Book, *, policy: GapPolicy = _DEFAULT_POLICY, lang: str = "en"
) -> list[list[Sentence]]:
    """Each chapter's sentences. The whole book is normalized in one call so
    every CPU core gets an equal share of it."""
    texts = iter(normalize_all([p.text for ch in book.chapters for p in ch.paragraphs], lang=lang))
    chapters: list[list[Sentence]] = []
    for idx, chapter in enumerate(book.chapters):
        flat: list[tuple[str, ParagraphKind, bool]] = []
        for paragraph in chapter.paragraphs:
            pieces = split_paragraph(Paragraph(text=next(texts), kind=paragraph.kind))
            last = len(pieces) - 1
            flat.extend((piece, paragraph.kind, i == last) for i, piece in enumerate(pieces))
        last_index = len(flat) - 1
        chapters.append(
            [
                Sentence(
                    text=text,
                    gap_after=sentence_gap(
                        text,
                        kind,
                        is_last_in_paragraph=is_last_in_paragraph,
                        is_last_in_chapter=i == last_index,
                        policy=policy,
                    ),
                    chapter_index=idx,
                )
                for i, (text, kind, is_last_in_paragraph) in enumerate(flat)
            ]
        )
    return chapters


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


Log = Callable[[str], None]


def _no_log(_message: str) -> None:
    return None


@dataclass(frozen=True, slots=True)
class _ChapterPlan:
    """One chapter's precomputed sentences/keys/gaps, plus the stored
    manifest if the cached flac for it is still current (``None`` means the
    chapter must be assembled this run)."""

    title: str
    sentences: list[Sentence]
    keys: tuple[str, ...]
    gaps: tuple[float, ...]
    manifest: cache.ChapterManifest | None

    @property
    def current(self) -> bool:
        return self.manifest is not None


def _plan_chapters(
    book: Book,
    out_dir: Path,
    *,
    engine_fingerprint: str,
    sample_rate: int,
    policy: GapPolicy,
    lang: str,
) -> list[_ChapterPlan]:
    plans = []
    for idx, (chapter, sentences) in enumerate(
        zip(book.chapters, book_to_sentences(book, policy=policy, lang=lang), strict=True)
    ):
        keys = tuple(cache.clip_cache_key(s.text) for s in sentences)
        gaps = tuple(s.gap_after for s in sentences)
        manifest = cache.current_chapter_manifest(
            out_dir,
            book.source_sha256,
            idx,
            engine_fingerprint=engine_fingerprint,
            sample_rate=sample_rate,
            clip_keys=keys,
            gap_after=gaps,
        )
        plans.append(
            _ChapterPlan(
                title=chapter.title, sentences=sentences, keys=keys, gaps=gaps, manifest=manifest
            )
        )
    return plans


def _resolve_pending(
    plans: Sequence[_ChapterPlan],
    *,
    cache_dir: Path,
    engine_fingerprint: str,
    log: Log,
) -> list[PendingClip]:
    """Check every non-current chapter's sentences for clip-cache presence
    (header only - nothing is decoded), pooling misses across chapter
    boundaries before any ``synthesize()`` call is made. Logs one line per
    chapter with its cached/to-synthesize split; a sentence repeated across
    chapters counts as "to synthesize" in each, but is queued only once."""
    pending: list[PendingClip] = []
    cached_by_key: dict[str, bool] = {}  # each key is checked on disk at most once
    for idx, plan in enumerate(plans):
        prefix = f"[{idx + 1}/{len(plans)}] {plan.title} — {len(plan.sentences)} sentences"
        if plan.current:
            log(f"{prefix}, chapter up to date")
            continue
        missing = 0
        for sentence, key in zip(plan.sentences, plan.keys, strict=True):
            if key not in cached_by_key:
                cached_by_key[key] = cache.has_clip(cache_dir, engine_fingerprint, key)
                if not cached_by_key[key]:
                    pending.append(PendingClip(key=key, text=sentence.text))
            if not cached_by_key[key]:
                missing += 1
        log(f"{prefix}, {len(plan.sentences) - missing} clips cached, {missing} to synthesize")
    return pending


def _synthesize_batch(engine: TTSEngine, batch: Sequence[PendingClip]) -> list[AudioClip]:
    clips = engine.synthesize([item.text for item in batch])
    if len(clips) != len(batch):
        raise ValueError(f"engine returned {len(clips)} clips for {len(batch)} texts")
    return clips


def _synthesize_pending(
    pending: Sequence[PendingClip],
    engine: TTSEngine,
    *,
    cache_dir: Path,
    engine_fingerprint: str,
    log: Log,
) -> None:
    """Run every pending sentence through the engine, length-sorted and
    capped at ``engine.max_batch`` per call, storing each result to the
    clip cache the moment it comes back - so a crash partway through only
    ever costs the in-flight batches' work.

    Engines that allow it (``max_concurrency`` above one - the hosted APIs)
    get their batches fanned out over a thread pool; results are stored as
    each finishes, in whatever order that is, since every clip travels with
    its own ``PendingClip`` key. Everything else stays a plain sequential
    loop with no threads involved.
    """
    batches = make_batches(pending, engine.max_batch)
    done = 0

    def store(n: int, batch: Sequence[PendingClip], clips: Sequence[AudioClip]) -> None:
        nonlocal done
        for item, clip in zip(batch, clips, strict=True):
            cache.store_clip(cache_dir, engine_fingerprint, item.key, clip)
        done += len(batch)
        log(f"batch {n}/{len(batches)}, {done}/{len(pending)} clips")

    if engine.max_concurrency <= 1:
        for n, batch in enumerate(batches, start=1):
            store(n, batch, _synthesize_batch(engine, batch))
        return

    with ThreadPoolExecutor(max_workers=engine.max_concurrency) as executor:
        futures = {executor.submit(_synthesize_batch, engine, batch): batch for batch in batches}
        try:
            for n, future in enumerate(as_completed(futures), start=1):
                store(n, futures[future], future.result())
        except BaseException:
            # Drop the batches still queued so the failure surfaces now
            # rather than after the whole pool drains.
            executor.shutdown(wait=False, cancel_futures=True)
            raise


def _cues(
    plan: _ChapterPlan, offsets: Sequence[tuple[float, float]]
) -> tuple[tuple[str, float, float], ...]:
    return tuple(
        (sentence.text, start, end)
        for sentence, (start, end) in zip(plan.sentences, offsets, strict=True)
    )


def _reuse_chapter(
    out_dir: Path, book_sha: str, idx: int, plan: _ChapterPlan, manifest: cache.ChapterManifest
) -> ChapterResult:
    flac_path = cache.chapter_flac_path(out_dir, book_sha, idx)
    return ChapterResult(
        flac_path=flac_path, duration=manifest.duration, cues=_cues(plan, manifest.offsets)
    )


def _assemble_chapter(
    out_dir: Path,
    book_sha: str,
    idx: int,
    plan: _ChapterPlan,
    *,
    cache_dir: Path,
    engine_fingerprint: str,
    sample_rate: int,
) -> ChapterResult:
    """Load this one chapter's clips from the cache (every one is there by
    now, cached earlier or stored by ``_synthesize_pending``) and write the
    assembled flac + manifest through ``cache.store_chapter``."""
    pairs: list[tuple[Sentence, AudioClip]] = []
    for sentence, key in zip(plan.sentences, plan.keys, strict=True):
        clip = cache.load_clip(cache_dir, engine_fingerprint, key)
        if clip is None:
            raise SynthesisError(
                f"clip {key} for chapter {idx} vanished from {cache_dir} between synthesis "
                "and assembly - rerun to synthesize it again"
            )
        pairs.append((sentence, clip))

    def write_audio(tmp_path: Path) -> tuple[cache.Offsets, float]:
        offsets, duration = assemble_chapter(pairs, tmp_path, sample_rate=sample_rate)
        return tuple(offsets), duration

    manifest = cache.store_chapter(
        out_dir,
        book_sha,
        idx,
        engine_fingerprint=engine_fingerprint,
        sample_rate=sample_rate,
        clip_keys=plan.keys,
        gap_after=plan.gaps,
        write_audio=write_audio,
    )
    return _reuse_chapter(out_dir, book_sha, idx, plan, manifest)


def synthesize_book(
    book: Book,
    engine: TTSEngine,
    *,
    cache_dir: Path,
    out_dir: Path,
    policy: GapPolicy = _DEFAULT_POLICY,
    lang: str = "en",
    log: Log = _no_log,
) -> list[ChapterResult]:
    """Render every chapter of ``book``, resuming from ``cache_dir`` (clip
    cache) and ``out_dir/.work`` (per-chapter flac + manifest) wherever possible.

    Chapters whose manifest still matches (same engine, sample rate,
    sentence set/order and gaps - see ``cache.current_chapter_manifest``)
    are reused as-is. For the rest: sentences without a cached clip are
    pooled *across all such chapters* and run through ``synth/batching.py``
    so a single ``synthesize()`` call never mixes wildly different sentence
    lengths. Progress goes to ``log`` one line at a time: a
    cached/to-synthesize split per chapter up front, then one line per
    engine batch, then one per assembled chapter.
    """
    engine_fingerprint = engine.fingerprint()
    sample_rate = engine.sample_rate
    book_sha = book.source_sha256

    plans = _plan_chapters(
        book,
        out_dir,
        engine_fingerprint=engine_fingerprint,
        sample_rate=sample_rate,
        policy=policy,
        lang=lang,
    )
    pending = _resolve_pending(
        plans, cache_dir=cache_dir, engine_fingerprint=engine_fingerprint, log=log
    )
    _synthesize_pending(
        pending, engine, cache_dir=cache_dir, engine_fingerprint=engine_fingerprint, log=log
    )

    results: list[ChapterResult] = []
    for idx, plan in enumerate(plans):
        if plan.manifest is not None:
            results.append(_reuse_chapter(out_dir, book_sha, idx, plan, plan.manifest))
            continue
        result = _assemble_chapter(
            out_dir,
            book_sha,
            idx,
            plan,
            cache_dir=cache_dir,
            engine_fingerprint=engine_fingerprint,
            sample_rate=sample_rate,
        )
        log(f"[{idx + 1}/{len(plans)}] assembled {plan.title} ({result.duration:.1f}s)")
        results.append(result)
    return results
