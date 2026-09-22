"""Command-line entry point."""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from epub_to_m4b import __version__
from epub_to_m4b.audio.ffmpeg import (
    FFmpegNotFoundError,
    concat_command,
    concat_list,
    encode_m4b_command,
    probe_chapters,
    require_ffmpeg,
    run_command,
)
from epub_to_m4b.audio.metadata import build_ffmetadata, embed_cover
from epub_to_m4b.audio.vtt import write_vtt
from epub_to_m4b.book import Book, Paragraph, ParagraphKind
from epub_to_m4b.config import ConfigError, load_config, resolve_cache_dir
from epub_to_m4b.epub.reader import read_book
from epub_to_m4b.synth.cache import atomic_replace
from epub_to_m4b.synth.orchestrator import GapPolicy, synthesize_book
from epub_to_m4b.text.normalize import normalize
from epub_to_m4b.text.split import split_paragraph
from epub_to_m4b.tts.registry import available_engines, create_engine

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epub-to-m4b")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    chapters = sub.add_parser("chapters", help="list detected chapters")
    _add_book_args(chapters)
    chapters.set_defaults(func=_cmd_chapters)

    dump = sub.add_parser("dump-text", help="print chapter titles and paragraphs")
    _add_book_args(dump)
    dump.add_argument("--chapter", type=int, default=None, help="1-based chapter index")
    dump.add_argument(
        "--normalized", action="store_true", help="run each paragraph through normalize(text)"
    )
    dump.add_argument(
        "--split",
        action="store_true",
        help="also show sentence boundaries (implies --normalized)",
    )
    dump.set_defaults(func=_cmd_dump_text)

    convert = sub.add_parser("convert", help="render an audiobook (m4b + vtt) with chapter markers")
    _add_book_args(convert)
    convert.add_argument("--engine", required=True, choices=available_engines())
    convert.add_argument("-o", "--out-dir", type=Path, required=True)
    convert.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config file (default: $E2M_CONFIG or ~/.config/epub-to-m4b/config.toml)",
    )
    convert.set_defaults(func=_cmd_convert)
    return parser


def _add_book_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("epub", type=Path)
    parser.add_argument("--toc-depth", type=int, default=1)
    parser.add_argument("--min-chars", type=int, default=200)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        book = read_book(args.epub, toc_depth=args.toc_depth, min_chars=args.min_chars)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: cannot read {args.epub}: {exc}", file=sys.stderr)
        return 1
    result: int = args.func(book, args)
    return result


def _cmd_chapters(book: Book, _args: argparse.Namespace) -> int:
    print(f"{book.title} — {book.author or 'unknown author'} ({len(book.chapters)} chapters)")
    print(f"{'#':>3}  {'title':<50}  {'paras':>5}  {'chars':>7}  {'docs':>4}")
    for n, ch in enumerate(book.chapters, start=1):
        chars = sum(len(p.text) for p in ch.paragraphs)
        title = ch.title if len(ch.title) <= 50 else ch.title[:49] + "…"
        print(f"{n:>3}  {title:<50}  {len(ch.paragraphs):>5}  {chars:>7}  {len(ch.source_ids):>4}")
    return 0


def _cmd_dump_text(book: Book, args: argparse.Namespace) -> int:
    chapters = list(enumerate(book.chapters, start=1))
    if args.chapter is not None:
        if not 1 <= args.chapter <= len(chapters):
            print(f"error: chapter {args.chapter} out of range 1..{len(chapters)}", file=sys.stderr)
            return 1
        chapters = [chapters[args.chapter - 1]]
    normalized = args.normalized or args.split
    for n, ch in chapters:
        print(f"=== [{n}] {ch.title}")
        for p in ch.paragraphs:
            prefix = "# " if p.kind is ParagraphKind.HEADING else ""
            text = normalize(p.text) if normalized else p.text
            print(f"{prefix}{text}")
            if args.split:
                sentence_paragraph = Paragraph(text=text, kind=p.kind)
                for i, sentence in enumerate(split_paragraph(sentence_paragraph)):
                    print(f"    [{i}] {sentence}")
        print()
    return 0


def _slugify(title: str) -> str:
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug or "book"


def _cmd_convert(book: Book, args: argparse.Namespace) -> int:
    try:
        require_ffmpeg()
    except FFmpegNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(book.title)
    m4b_path = out_dir / f"{slug}.m4b"
    vtt_path = out_dir / f"{slug}.vtt"

    cache_dir = resolve_cache_dir()
    try:
        app_config = load_config(args.config, cache_dir=cache_dir)
        engine = create_engine(args.engine, app_config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    def log(message: str) -> None:
        # Renders run for hours under nohup/redirect; block-buffered stdout
        # would show nothing until exit.
        print(message, flush=True)

    with engine:
        results = synthesize_book(
            book, engine, cache_dir=cache_dir, out_dir=out_dir, policy=GapPolicy(), log=log
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

    if _m4b_is_current(m4b_path, chapter_files, expected_chapters=len(book.chapters)):
        print(f"{m4b_path} already up to date, skipping re-encode")
    else:
        _write_m4b(book, m4b_path, chapter_files, durations)

    write_vtt(cues, vtt_path)
    print(f"wrote {m4b_path}")
    print(f"wrote {vtt_path}")
    return 0


def _m4b_is_current(
    m4b_path: Path, chapter_files: Sequence[Path], *, expected_chapters: int
) -> bool:
    """Newer than every chapter flac *and* a container ffprobe can read with
    the right chapter count. mtime alone is not enough: an m4b left behind
    by an interrupted or failed encode could be newer than everything and
    still be garbage, and would then never be rebuilt."""
    if not m4b_path.is_file():
        return False
    m4b_mtime = m4b_path.stat().st_mtime
    if any(f.stat().st_mtime > m4b_mtime for f in chapter_files):
        return False
    try:
        probe = probe_chapters(m4b_path)
    except RuntimeError, ValueError:
        return False
    return len(probe.get("chapters", [])) == expected_chapters


def _write_m4b(
    book: Book, m4b_path: Path, chapter_files: Sequence[Path], durations: Sequence[float]
) -> None:
    """Concat + encode + cover-embed into a temp file next to ``m4b_path``,
    then ``os.replace`` it into place, so a crash mid-encode never leaves a
    partial m4b at the real path."""

    def write_body(tmp_m4b: Path) -> None:
        with tempfile.TemporaryDirectory(prefix="epub-to-m4b-") as tmp_name:
            tmp_dir = Path(tmp_name)
            list_path = tmp_dir / "concat.txt"
            list_path.write_text(concat_list(chapter_files), encoding="utf-8")
            combined_path = tmp_dir / "combined.flac"
            run_command(concat_command(list_path, combined_path))

            metadata_path = tmp_dir / "ffmetadata.txt"
            metadata_path.write_text(build_ffmetadata(book, durations), encoding="utf-8")
            run_command(encode_m4b_command(combined_path, metadata_path, tmp_m4b))
        if book.cover and book.cover_mime:
            embed_cover(tmp_m4b, book.cover, book.cover_mime)

    atomic_replace(m4b_path, write_body, suffix=".m4b")
