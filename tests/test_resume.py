"""M5 acceptance test: kill -9 mid-render, rerun, prove the resume cache works.

Runs the CLI as a real subprocess against ``--engine silence`` (no GPU/network
needed - this is a pipeline correctness test, not an engine test) so it can
be sent a genuine ``SIGKILL``: killing a thread or an in-process call would
not exercise the atomic-write crash safety this test is actually checking
(``synth/cache.py``'s temp-file-then-``os.replace`` discipline).

Two small, test-only seams make this possible across a process boundary
(both default to off and change nothing about normal use - see
``tts/fake.py`` and ``tts/registry.py``):

- ``E2M_SILENCE_DELAY_SECONDS`` spaces out ``synthesize()`` calls in
  wall-clock time, so a kill can land reliably mid-run instead of racing a
  fake engine that would otherwise finish in a few milliseconds.
- ``E2M_SILENCE_CALL_LOG`` appends every requested sentence to a file, one
  per call, giving this test (running in a different process than the one
  it kills) direct visibility into exactly what was asked for.

``E2M_CACHE_DIR`` (a normal, documented override, same precedent as
``E2M_CONFIG``) points the resume cache at a throwaway directory instead of
the real ``~/.cache/epub-to-m4b``.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from epub_to_m4b.audio.ffmpeg import probe_chapters
from epub_to_m4b.synth import cache as clip_cache
from epub_to_m4b.text import TEXT_PIPELINE_VERSION
from epub_to_m4b.tts.fake import SilenceEngine

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_N_CHAPTERS = 5
_SENTENCES_PER_CHAPTER = 5
_CALL_DELAY_SECONDS = 0.12

_CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""


def _xhtml(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml"><head><title>t</title></head>'
        f"<body>{body}</body></html>"
    )


def _chapter_sentence(chapter_index: int, sentence_index: int) -> str:
    # Long and distinct enough that (a) the splitter's short-fragment merge
    # leaves each one standing alone and (b) no two sentences anywhere in
    # the book collide on cache key - every one is its own resume unit.
    return (
        f"This is the unique sentence number {sentence_index} of chapter "
        f"{chapter_index}, long enough on its own to never be merged with "
        f"a neighbor by the sentence splitter."
    )


def _build_resume_epub(path: Path) -> Path:
    manifest_items = []
    spine_items = []
    nav_points = []
    docs: dict[str, str] = {}
    for c in range(_N_CHAPTERS):
        doc_id = f"ch{c}"
        href = f"{doc_id}.xhtml"
        sentences = " ".join(_chapter_sentence(c, i) for i in range(_SENTENCES_PER_CHAPTER))
        docs[href] = _xhtml(f"<h1>Chapter {c}</h1><p>{sentences}</p>")
        manifest_items.append(
            f'<item id="{doc_id}" href="{href}" media-type="application/xhtml+xml"/>'
        )
        spine_items.append(f'<itemref idref="{doc_id}"/>')
        nav_points.append(
            f'<navPoint id="n{c}" playOrder="{c + 1}">'
            f"<navLabel><text>Chapter {c}</text></navLabel>"
            f'<content src="{href}"/></navPoint>'
        )

    content_opf = f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uid" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Resume Test Book</dc:title>
    <dc:creator>Resume Author</dc:creator>
    <dc:language>en</dc:language>
    <dc:identifier id="uid">urn:uuid:resume-test</dc:identifier>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    {"".join(manifest_items)}
  </manifest>
  <spine toc="ncx">
    {"".join(spine_items)}
  </spine>
</package>
"""
    toc_ncx = f"""<?xml version="1.0" encoding="utf-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head><meta name="dtb:uid" content="urn:uuid:resume-test"/></head>
  <docTitle><text>Resume Test Book</text></docTitle>
  <navMap>{"".join(nav_points)}</navMap>
</ncx>
"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", _CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", content_opf)
        zf.writestr("OEBPS/toc.ncx", toc_ncx)
        for href, content in docs.items():
            zf.writestr(f"OEBPS/{href}", content)
    return path


def _convert_argv(epub: Path, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "epub_to_m4b",
        "convert",
        str(epub),
        "--engine",
        "silence",
        "-o",
        str(out_dir),
    ]


def _env(cache_dir: Path, call_log: Path, *, delay: float = 0.0) -> dict[str, str]:
    env = dict(os.environ)
    env["E2M_CACHE_DIR"] = str(cache_dir)
    env["E2M_SILENCE_CALL_LOG"] = str(call_log)
    env["E2M_SILENCE_DELAY_SECONDS"] = str(delay)
    return env


def _all_sentence_texts() -> list[str]:
    texts = []
    for c in range(_N_CHAPTERS):
        texts.append(f"Chapter {c}")
        texts.extend(_chapter_sentence(c, i) for i in range(_SENTENCES_PER_CHAPTER))
    return texts


def _cached_texts(cache_dir: Path, texts: list[str]) -> set[str]:
    """Ground truth for "already durably cached", read straight off the
    filesystem via the same ``synth/cache.py`` functions the orchestrator
    uses - not from the call log. The call log records when a sentence was
    *requested*, which can briefly precede the matching ``store_clip`` call;
    a kill landing in that narrow gap would make the log look like a cache
    hit that was never actually persisted. Checking the cache directly
    avoids that race entirely.
    """
    fingerprint = SilenceEngine().fingerprint()
    cached = set()
    for text in texts:
        key = clip_cache.clip_cache_key(TEXT_PIPELINE_VERSION, text)
        if clip_cache.load_clip(cache_dir, fingerprint, key) is not None:
            cached.add(text)
    return cached


def test_kill_9_mid_render_then_rerun_resumes_without_resynthesizing_cached_clips(
    tmp_path: Path,
) -> None:
    epub = _build_resume_epub(tmp_path / "resume.epub")
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    first_call_log = tmp_path / "calls-before-kill.log"
    total_sentences = _N_CHAPTERS * (_SENTENCES_PER_CHAPTER + 1)  # +1 heading per chapter

    proc = subprocess.Popen(
        _convert_argv(epub, out_dir),
        cwd=_REPO_ROOT,
        env=_env(cache_dir, first_call_log, delay=_CALL_DELAY_SECONDS),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Long enough to clear interpreter/import startup and land solidly
        # mid-run (a handful of the total sentences done), short enough to
        # guarantee the whole book isn't finished yet.
        time.sleep(_CALL_DELAY_SECONDS * (total_sentences / 2))
        proc.send_signal(signal.SIGKILL)
        remaining_output = proc.communicate(timeout=10)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)

    assert proc.returncode != 0, f"process finished before the kill could land: {remaining_output}"
    assert first_call_log.is_file(), f"engine was never invoked before the kill: {remaining_output}"
    calls_before_kill = first_call_log.read_text(encoding="utf-8").splitlines()
    assert len(calls_before_kill) == len(set(calls_before_kill)), (
        "the same sentence was requested twice in a single run - dedup within one pass is broken"
    )

    # Ground truth for "already cached before the kill", read straight off
    # the filesystem (see _cached_texts) rather than trusting the call log:
    # the log is written by the engine the instant it's asked for a text,
    # which is a hair before the orchestrator's matching store_clip() call
    # lands - a kill in that narrow gap would log a request whose clip was
    # never actually persisted. Filesystem state is what resume actually
    # depends on, so that's what this test checks.
    all_texts = _all_sentence_texts()
    assert len(all_texts) == total_sentences
    cached_before_kill = _cached_texts(cache_dir, all_texts)
    assert 0 < len(cached_before_kill) < total_sentences, (
        f"kill landed at the wrong time (got {len(cached_before_kill)} of "
        f"{total_sentences} clips durably cached) - widen the margin between "
        "the sleep and the total expected run time"
    )

    # The process never got past synthesis - no output files at all yet.
    assert not (out_dir / "resume-test-book.m4b").exists()

    second_call_log = tmp_path / "calls-after-resume.log"
    rerun = subprocess.run(
        _convert_argv(epub, out_dir),
        cwd=_REPO_ROOT,
        env=_env(cache_dir, second_call_log),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr

    m4b_path = out_dir / "resume-test-book.m4b"
    vtt_path = out_dir / "resume-test-book.vtt"
    assert m4b_path.is_file()
    assert vtt_path.is_file()

    probe = probe_chapters(m4b_path)
    assert len(probe["chapters"]) == _N_CHAPTERS

    calls_after_resume = (
        second_call_log.read_text(encoding="utf-8").splitlines() if second_call_log.exists() else []
    )

    # The whole point: nothing that was already durably cached before the
    # kill was asked of the engine again on the rerun. If resume were broken
    # (e.g. the clip cache key ignored the fingerprint directory, a
    # corrupt/partial write from the kill wedged a "hit" that wasn't really
    # readable, or the orchestrator re-synthesized whole chapters instead of
    # just the missing sentences) this would fail by finding overlap here.
    assert set(calls_after_resume).isdisjoint(cached_before_kill)
    # And after the rerun, every sentence in the book has a valid cache
    # entry - the resume picked up every bit of work the kill interrupted,
    # nothing was silently dropped.
    cached_after_resume = _cached_texts(cache_dir, all_texts)
    assert cached_after_resume == set(all_texts)
