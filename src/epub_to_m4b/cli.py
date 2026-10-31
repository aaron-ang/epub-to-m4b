"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import soundfile as sf

from epub_to_m4b import __version__
from epub_to_m4b.audio.ffmpeg import (
    AAC_BITRATE,
    LOUDNESS_TARGET_LUFS,
    concat_command,
    concat_list,
    encode_m4b_command,
    encode_settings_digest,
    probe_chapters,
    require_ffmpeg,
    run_with_progress,
)
from epub_to_m4b.audio.metadata import build_ffmetadata, embed_cover
from epub_to_m4b.audio.vtt import write_vtt
from epub_to_m4b.book import Book, Paragraph, ParagraphKind
from epub_to_m4b.config import load_config, resolve_cache_dir
from epub_to_m4b.epub.chapters import DEFAULT_MIN_CHARS, DEFAULT_TOC_DEPTH
from epub_to_m4b.epub.reader import read_book
from epub_to_m4b.errors import EpubToM4bError
from epub_to_m4b.progress import PercentSteps
from epub_to_m4b.synth.cache import (
    atomic_replace,
    clear_encode_digest,
    load_encode_digest,
    store_encode_digest,
)
from epub_to_m4b.synth.orchestrator import GapPolicy, synthesize_book
from epub_to_m4b.text.normalize import normalize_all
from epub_to_m4b.text.split import DEFAULT_MAX_CHARS, split_paragraph
from epub_to_m4b.tts.base import TTSEngine
from epub_to_m4b.tts.registry import available_engines, create_engine

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="epub-to-m4b", description="Turn an EPUB into an M4B audiobook with a VTT transcript."
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", title="commands")

    chapters = sub.add_parser(
        "chapters",
        help="list detected chapters",
        description="Print the chapter table the converter would narrate, with paragraph and "
        "character counts, so TOC depth and merge threshold can be tuned before rendering.",
    )
    _add_book_args(chapters)
    chapters.set_defaults(func=_with_book(_cmd_chapters))

    dump = sub.add_parser(
        "dump-text",
        help="print chapter titles and paragraphs",
        description="Print the text of every chapter (or one chapter) as the narrator will see "
        "it, optionally after normalisation and sentence splitting.",
    )
    _add_book_args(dump)
    dump.add_argument("--chapter", type=int, default=None, help="1-based chapter index")
    dump.add_argument("--normalized", action="store_true", help="run each paragraph through NeMo")
    dump.add_argument(
        "--split",
        action="store_true",
        help="also show sentence boundaries (implies --normalized)",
    )
    dump.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="with --split, cut only sentences longer than this; match the engine's "
        f"max_chars to see its clips (default: {DEFAULT_MAX_CHARS})",
    )
    dump.set_defaults(func=_with_book(_cmd_dump_text))

    convert = sub.add_parser(
        "convert",
        help="render an audiobook (m4b + vtt) with chapter markers",
        description="Synthesise every chapter of each book with the chosen TTS engine and write "
        "an .m4b with chapter markers plus a WebVTT transcript into the output directory. One "
        "engine serves every book; a book that fails is skipped and the rest still convert.",
    )
    _add_book_args(convert, many=True)
    convert.add_argument(
        "--engine",
        required=True,
        choices=available_engines(),
        help="TTS engine (see README for configuration)",
    )
    convert.add_argument(
        "-o", "--out-dir", type=Path, required=True, help="directory for the .m4b and .vtt"
    )
    convert.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config file (default: $E2M_CONFIG or ~/.config/epub-to-m4b/config.toml)",
    )
    convert.set_defaults(func=_cmd_convert)
    return parser


def _add_book_args(parser: argparse.ArgumentParser, *, many: bool = False) -> None:
    if many:
        parser.add_argument("epub", type=Path, nargs="+", help="paths to one or more .epub files")
    else:
        parser.add_argument("epub", type=Path, help="path to the .epub file")
    parser.add_argument(
        "--toc-depth",
        type=int,
        default=DEFAULT_TOC_DEPTH,
        help="TOC nesting level that starts a new chapter (default: %(default)s)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=DEFAULT_MIN_CHARS,
        help="chapters with less body text than this merge into a neighbour (default: %(default)s)",
    )


def _configure_logging() -> None:
    """Route this package's INFO logs (engine guard notes, busy waits) to stderr.

    Only the ``epub_to_m4b`` logger gets a handler - not the root - so
    third-party INFO chatter such as httpx request lines stays quiet. Idempotent
    so repeated ``main()`` calls in one process don't stack handlers.
    """
    package_logger = logging.getLogger("epub_to_m4b")
    if package_logger.handlers:
        return

    # Binds the stream object that is current at call time, not the module
    # attribute, so a later reassignment of sys.stderr is not followed.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(logging.INFO)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    _configure_logging()
    result: int = args.func(args)
    return result


def _read_book(path: Path, args: argparse.Namespace) -> Book | None:
    """Parse ``path``, or print why it can't be read and return ``None``."""
    try:
        return read_book(path, toc_depth=args.toc_depth, min_chars=args.min_chars)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: cannot read {path}: {exc}", file=sys.stderr)
        return None


def _with_book(
    command: Callable[[Book, argparse.Namespace], int],
) -> Callable[[argparse.Namespace], int]:
    """Adapt a single-book command to ``main``: read ``args.epub`` first."""

    def run(args: argparse.Namespace) -> int:
        book = _read_book(args.epub, args)
        return 1 if book is None else command(book, args)

    return run


# Widest the title column in `chapters` output gets; longer titles are
# truncated with an ellipsis so the numeric columns stay aligned.
_TITLE_COLUMN_WIDTH = 50


def _cmd_chapters(book: Book, _args: argparse.Namespace) -> int:
    print(f"{book.title} — {book.author or 'unknown author'} ({len(book.chapters)} chapters)")
    longest = max((len(ch.title) for ch in book.chapters), default=0)
    width = max(len("title"), min(_TITLE_COLUMN_WIDTH, longest))
    print(f"{'#':>3}  {'title':<{width}}  {'paras':>5}  {'chars':>7}  {'docs':>4}")

    for n, ch in enumerate(book.chapters, start=1):
        chars = sum(len(p.text) for p in ch.paragraphs)
        title = ch.title if len(ch.title) <= width else ch.title[: width - 1] + "…"
        print(
            f"{n:>3}  {title:<{width}}  {len(ch.paragraphs):>5}"
            f"  {chars:>7}  {len(ch.source_ids):>4}"
        )

    return 0


def _cmd_dump_text(book: Book, args: argparse.Namespace) -> int:
    chapters = list(enumerate(book.chapters, start=1))
    if args.chapter is not None:
        if not 1 <= args.chapter <= len(chapters):
            print(f"error: chapter {args.chapter} out of range 1..{len(chapters)}", file=sys.stderr)
            return 1

        chapters = [chapters[args.chapter - 1]]

    paragraph_texts = [p.text for _, ch in chapters for p in ch.paragraphs]
    if args.normalized or args.split:
        paragraph_texts = normalize_all(paragraph_texts)

    texts = iter(paragraph_texts)
    for n, ch in chapters:
        print(f"=== [{n}] {ch.title}")
        for p in ch.paragraphs:
            prefix = "# " if p.kind is ParagraphKind.HEADING else ""
            text = next(texts)
            print(f"{prefix}{text}")
            if args.split:
                sentence_paragraph = Paragraph(text=text, kind=p.kind)
                pieces = split_paragraph(sentence_paragraph, max_chars=args.max_chars)
                for i, sentence in enumerate(pieces):
                    print(f"    [{i}] {sentence}")
        print()

    return 0


def _slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug or "book"


def _log(message: str) -> None:
    # Renders run for hours under nohup/redirect; block-buffered stdout
    # would show nothing until exit.
    print(message, flush=True)


def _cmd_convert(args: argparse.Namespace) -> int:
    """Convert every book in ``args.epub`` with one engine, so a sidecar
    server starts once for the batch. A book that cannot be read or fails to
    convert is reported and skipped; the exit code is 1 if any book failed.
    Books are read before the engine starts, so a batch with nothing to
    convert never starts a server."""
    paths: list[Path] = args.epub
    out_dir: Path = args.out_dir
    failed: list[Path] = []
    # Slug -> the book that claimed it, so two books with the same title
    # cannot overwrite each other's m4b in one run.
    claimed: dict[str, tuple[Path, str]] = {}
    pending: deque[tuple[Path, str, Book]] = deque()
    for path in paths:
        book = _read_book(path, args)
        if book is None:
            failed.append(path)
            continue
        slug = _slugify(book.title)
        if slug in claimed:
            first, sha = claimed[slug]
            if sha != book.source_sha256:
                print(f"error: {path}: {slug}.m4b is also the output of {first}", file=sys.stderr)
                failed.append(path)
            # The same book named twice converts once.
            continue
        claimed[slug] = (path, book.source_sha256)
        pending.append((path, slug, book))

    if pending:
        try:
            require_ffmpeg()
            cache_dir = resolve_cache_dir()
            app_config = load_config(args.config, cache_dir=cache_dir)
            engine = create_engine(args.engine, app_config)
        except EpubToM4bError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failed.extend(path for path, _, _ in pending)
            pending.clear()
        else:
            total = len(pending)
            with engine:
                for n in range(1, total + 1):
                    # Popped, so each Book is freed once converted.
                    path, slug, book = pending.popleft()
                    if total > 1:
                        _log(f"=== [{n}/{total}] {path}")
                    try:
                        _convert(book, slug, engine, out_dir=out_dir, cache_dir=cache_dir)
                    except (EpubToM4bError, OSError) as exc:
                        print(f"error: {exc}", file=sys.stderr)
                        failed.append(path)

    if failed and len(paths) > 1:
        print(f"failed {len(failed)} of {len(paths)} books:", file=sys.stderr)
        for path in failed:
            print(f"  {path}", file=sys.stderr)

    return 1 if failed else 0


def _convert(book: Book, slug: str, engine: TTSEngine, *, out_dir: Path, cache_dir: Path) -> None:
    """Convert one book with no error handling; ``_cmd_convert`` reports
    every :class:`EpubToM4bError` and moves on to the next book."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m4b_path = out_dir / f"{slug}.m4b"
    vtt_path = out_dir / f"{slug}.vtt"
    results = synthesize_book(
        book,
        engine,
        cache_dir=cache_dir,
        out_dir=out_dir,
        policy=GapPolicy(),
        log=_log,
    )

    cues: list[tuple[str, float, float]] = []
    durations: list[float] = []
    chapter_files: list[Path] = []
    book_cursor = 0.0
    for result in results:
        for text, start, end in result.cues:
            cues.append((text, book_cursor + start, book_cursor + end))
        durations.append(result.duration)
        chapter_files.append(result.flac_path)
        book_cursor += result.duration

    digest = encode_settings_digest()
    settings_match = load_encode_digest(out_dir, book.source_sha256) == digest
    if settings_match and _m4b_is_current(
        m4b_path, chapter_files, expected_chapters=len(book.chapters)
    ):
        print(f"{m4b_path} already up to date, skipping re-encode")
    else:
        clear_encode_digest(out_dir, book.source_sha256)
        _write_m4b(book, m4b_path, chapter_files, durations, log=_log)
        store_encode_digest(out_dir, book.source_sha256, digest)

    write_vtt(cues, vtt_path)
    print(f"wrote {m4b_path}")
    print(f"wrote {vtt_path}")


def _m4b_is_current(
    m4b_path: Path, chapter_files: Sequence[Path], *, expected_chapters: int
) -> bool:
    """Newer than every chapter flac *and* a container ffprobe can read with
    the right chapter count (the caller checks the encode-settings stamp
    first). mtime alone is not enough: an m4b left behind by an interrupted
    or failed encode could be newer than everything and still be garbage,
    and would then never be rebuilt."""
    if not m4b_path.is_file():
        return False

    m4b_mtime = m4b_path.stat().st_mtime
    if any(f.stat().st_mtime > m4b_mtime for f in chapter_files):
        return False

    try:
        probe = probe_chapters(m4b_path)
    except EpubToM4bError, ValueError:
        return False

    return len(probe.get("chapters", [])) == expected_chapters


def _hms(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def _run_logged(args: Sequence[str], label: str, total: float, log: Callable[[str], None]) -> None:
    """Run an ffmpeg command, logging ``label N% (pos / total)`` each
    ``PROGRESS_STEP_PERCENT`` of ``total`` seconds, and 100% once it exits:
    ffmpeg's last reported position can fall just short of the total."""
    steps = PercentSteps(total)

    def report(seconds: float) -> None:
        seconds = min(seconds, total)
        if steps.due(seconds):
            # True division: float ``//`` can floor ``total * 100 // total`` to 99.
            percent = 100 if seconds >= total else int(seconds * 100 / total)
            log(f"{label} {percent}% ({_hms(seconds)} / {_hms(total)})")

    run_with_progress(args, report)
    report(total)


def _write_m4b(
    book: Book,
    m4b_path: Path,
    chapter_files: Sequence[Path],
    durations: Sequence[float],
    *,
    log: Callable[[str], None],
) -> None:
    """Concat + loudness-normalized encode + cover-embed into a temp file
    next to ``m4b_path``, then ``os.replace`` it into place, so a crash
    mid-encode never leaves a partial m4b at the real path."""
    total = sum(durations)

    def write_body(tmp_m4b: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="epub-to-m4b-") as tmp_name:
            tmp_dir = Path(tmp_name)
            list_path = tmp_dir / "concat.txt"
            list_path.write_text(concat_list(chapter_files), encoding="utf-8")
            combined_path = tmp_dir / "combined.flac"
            log(f"concatenating {len(chapter_files)} chapters ({_hms(total)})")
            _run_logged(concat_command(list_path, combined_path), "concat", total, log)

            metadata_path = tmp_dir / "ffmetadata.txt"
            metadata_path.write_text(build_ffmetadata(book, durations), encoding="utf-8")
            sample_rate = sf.info(combined_path).samplerate
            log(f"encoding m4b: loudnorm to {LOUDNESS_TARGET_LUFS} LUFS + AAC {AAC_BITRATE}")
            _run_logged(
                encode_m4b_command(combined_path, metadata_path, tmp_m4b, sample_rate=sample_rate),
                "encode",
                total,
                log,
            )
        if book.cover and book.cover_mime:
            log("embedding cover")
            embed_cover(tmp_m4b, book.cover, book.cover_mime)

    atomic_replace(m4b_path, write_body, suffix=".m4b")
