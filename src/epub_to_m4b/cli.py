"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from epub_to_m4b import __version__
from epub_to_m4b.book import Book, ParagraphKind
from epub_to_m4b.epub.reader import read_book


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
    dump.set_defaults(func=_cmd_dump_text)
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
    for n, ch in chapters:
        print(f"=== [{n}] {ch.title}")
        for p in ch.paragraphs:
            prefix = "# " if p.kind is ParagraphKind.HEADING else ""
            print(f"{prefix}{p.text}")
        print()
    return 0
