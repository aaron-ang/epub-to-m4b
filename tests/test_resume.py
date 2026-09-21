"""Resume after ``kill -9``: a rerun must not re-synthesize any clip that was
already durably in the cache when the process died.

The CLI runs as a real subprocess (``--engine silence``: no GPU/network,
this is a pipeline test, not an engine test) so it can be sent a genuine
``SIGKILL`` - killing a thread or an in-process call would not exercise the
temp-file-then-``os.replace`` discipline in ``synth/cache.py`` that this
test is actually checking. The child process is a small ``python -c`` driver
owned by this test: it monkeypatches ``SilenceEngine.synthesize`` to sleep
briefly (so the kill lands mid-run instead of racing a fake that would
finish in milliseconds) and to append each requested text to a log file,
then calls ``epub_to_m4b.cli.main``. Nothing in ``src/`` knows this test
exists.

Ground truth for "already cached before the kill" comes from the cache
filesystem itself, validated file by file, never from that log: the engine
logs a request the instant it is asked, a hair before the orchestrator's
matching ``store_clip`` lands, and ``load_clip`` would silently delete (and
so hide) a clip that a non-atomic writer left corrupt.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
import soundfile as sf

from epub_to_m4b.audio.ffmpeg import probe_chapters
from epub_to_m4b.synth.cache import clip_cache_key
from epub_to_m4b.text import TEXT_PIPELINE_VERSION

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not found on PATH",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_N_CHAPTERS = 5
_SENTENCES_PER_CHAPTER = 5
_CALL_DELAY_SECONDS = 0.1
_KILL_AFTER_CLIPS = 3
_KILL_DEADLINE_SECONDS = 60

_CONTAINER_XML = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""

# Runs in the child. __DELAY__/__LOG__/__ARGV__ are substituted with JSON
# literals before launch.
_DRIVER = """
import sys, time
from pathlib import Path
from epub_to_m4b import cli
from epub_to_m4b.tts.fake import SilenceEngine

delay = __DELAY__
log = Path(__LOG__)
original = SilenceEngine.synthesize

def paced(self, texts):
    time.sleep(delay)
    with log.open("a", encoding="utf-8") as fh:
        for text in texts:
            fh.write(text + "\\n")
    return original(self, texts)

SilenceEngine.synthesize = paced
sys.exit(cli.main(__ARGV__))
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


def _all_sentence_texts() -> list[str]:
    texts = []
    for c in range(_N_CHAPTERS):
        texts.append(f"Chapter {c}")
        texts.extend(_chapter_sentence(c, i) for i in range(_SENTENCES_PER_CHAPTER))
    return texts


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


def _driver_command(epub: Path, out_dir: Path, *, delay: float, log: Path) -> list[str]:
    argv = ["convert", str(epub), "--engine", "silence", "-o", str(out_dir)]
    script = (
        _DRIVER.replace("__DELAY__", json.dumps(delay))
        .replace("__LOG__", json.dumps(str(log)))
        .replace("__ARGV__", json.dumps(argv))
    )
    return [sys.executable, "-c", script]


def _final_clip_files(clips_root: Path) -> list[Path]:
    """Every file under the clips tree that is *not* an in-flight temp file
    (``atomic_replace`` names those with a leading dot)."""
    if not clips_root.is_dir():
        return []
    return [p for p in clips_root.rglob("*") if p.is_file() and not p.name.startswith(".")]


def _temp_clip_files(clips_root: Path) -> list[Path]:
    if not clips_root.is_dir():
        return []
    return [p for p in clips_root.rglob(".*") if p.is_file()]


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


def test_kill_9_mid_render_then_rerun_resumes_without_resynthesizing_cached_clips(
    tmp_path: Path,
) -> None:
    epub = _build_resume_epub(tmp_path / "resume.epub")
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"
    clips_root = cache_dir / "clips"
    env = {**os.environ, "E2M_CACHE_DIR": str(cache_dir)}
    all_texts = _all_sentence_texts()
    text_for_key = {clip_cache_key(TEXT_PIPELINE_VERSION, text): text for text in all_texts}
    assert len(text_for_key) == len(all_texts)  # every sentence is its own resume unit

    first_log = tmp_path / "calls-before-kill.log"
    proc = subprocess.Popen(
        _driver_command(epub, out_dir, delay=_CALL_DELAY_SECONDS, log=first_log),
        cwd=_REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Kill as soon as a few clips have landed - no fixed sleep, so a slow
        # or busy machine just waits longer rather than missing the window.
        deadline = time.monotonic() + _KILL_DEADLINE_SECONDS
        while len(_final_clip_files(clips_root)) < _KILL_AFTER_CLIPS:
            assert proc.poll() is None, f"process finished before the kill: {proc.stdout}"
            assert time.monotonic() < deadline, "no clips appeared in time"
            time.sleep(0.02)
        proc.send_signal(signal.SIGKILL)
        output = proc.communicate(timeout=10)[0]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate(timeout=10)
    assert proc.returncode == -signal.SIGKILL, output

    # --- What the kill left behind, checked directly, file by file. -----
    # At most one temp file may exist: the single write that was in flight
    # at the instant of the kill. More would mean writes aren't sequential
    # or aren't cleaned up.
    assert len(_temp_clip_files(clips_root)) <= 1
    # Every file at a *final* path must be a complete, readable FLAC. A
    # non-atomic store_clip (writing straight to the final name) would leave
    # a truncated file here - this is where that would show up.
    finals_before_kill = _final_clip_files(clips_root)
    for path in finals_before_kill:
        assert path.suffix == ".flac", path
        assert path.stem in text_for_key, f"unexpected clip {path}"
        sf.info(path)  # raises on a truncated/garbage file
    cached_before_kill = {text_for_key[p.stem] for p in finals_before_kill}
    mtimes_before_kill = {p: p.stat().st_mtime_ns for p in finals_before_kill}
    assert _KILL_AFTER_CLIPS <= len(cached_before_kill) < len(all_texts)
    # Synthesis never finished, so nothing downstream was written either.
    assert not (out_dir / "resume-test-book.m4b").exists()

    # --- Rerun to completion. --------------------------------------------
    second_log = tmp_path / "calls-after-resume.log"
    rerun = subprocess.run(
        _driver_command(epub, out_dir, delay=0.0, log=second_log),
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr

    m4b_path = out_dir / "resume-test-book.m4b"
    assert m4b_path.is_file()
    assert (out_dir / "resume-test-book.vtt").is_file()
    assert len(probe_chapters(m4b_path)["chapters"]) == _N_CHAPTERS

    # The claim under test, from two independent angles:
    # 1. the engine was never asked again for anything already cached;
    calls_after_resume = set(_read_lines(second_log))
    assert calls_after_resume.isdisjoint(cached_before_kill)
    # 2. no pre-kill clip file was rewritten (every synthesized clip is
    #    stored, so an unchanged mtime means it was not synthesized again).
    assert {p: p.stat().st_mtime_ns for p in finals_before_kill} == mtimes_before_kill
    # And the rerun did pick up everything the kill interrupted: every
    # sentence in the book now has a readable clip, and the rerun requested
    # exactly the missing ones.
    finals_after = {p.stem: p for p in _final_clip_files(clips_root)}
    assert set(finals_after) == set(text_for_key)
    for path in finals_after.values():
        sf.info(path)
    assert calls_after_resume == set(all_texts) - cached_before_kill
