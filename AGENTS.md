# AGENTS.md

Architecture and conventions for anyone (human or agent) working on this repo.
For what the tool does and how to run it, see [README.md](README.md).

## Why this exists

A small, typed, well-tested tool for turning an EPUB into an M4B audiobook:
one job per module, no dead code, no UI dependencies dragged into headless
runs, deterministic resume (no randomness in pause lengths), clean chapter
titles (no stray markup leaking through), and TTS engines kept behind one
narrow interface so a new backend is a new file, not a rewrite.

## Layout

```
src/epub_to_m4b/
  book.py               Book, Chapter, Paragraph, Sentence, AudioClip
  epub/reader.py         ebooklib -> Book (DC metadata, cover, spine docs)
  epub/html.py            BeautifulSoup(lxml) DOM walk -> list[Paragraph]
  epub/chapters.py        TOC->spine mapping, heading fallback, running-header removal, stub merge
  text/normalize.py      normalize(text, lang) -> str, dispatch to text/lang/<lang>.py
  text/lang/english.py    decades, years, ordinals, roman (headings), clock, math, thousands, abbreviations
  text/split.py           Paragraph -> list[Sentence]; char cap; hard/soft/space/hard-cut; short merge
  tts/base.py            TTSEngine ABC, pcm16->float32 helpers
  tts/registry.py         name -> factory + importlib.metadata entry points ("epub_to_m4b.engines")
  tts/http.py             shared retry/backoff/rate-limit client for API engines
  tts/guard.py            RunawayGuard: retry/cut limits, max_new_tokens, cut+fade
  tts/sidecar.py          spawn/adopt a local HTTP TTS server, health poll, log file
  tts/breeze.py           BreezeEngine: batch endpoint, 409 wait, reference voice, guard
  tts/openai_compat.py    OpenAI-compatible /v1/audio/speech
  tts/elevenlabs.py       ElevenLabs /v1/text-to-speech/{voice}
  tts/deepgram.py         Deepgram Aura /v1/speak
  tts/fake.py             SilenceEngine, ToneEngine for tests and dry runs
  synth/cache.py         content-addressed flac store, atomic writes, invalidation
  synth/batching.py       length-sorted windows across chapters
  synth/orchestrator.py   sentences -> missing -> batches -> engine -> cache; gap policy
  audio/assemble.py      clips + gaps -> chapter flac; records offsets
  audio/ffmpeg.py         pure command builders + subprocess: concat, ffmetadata, aac encode, ffprobe
  audio/metadata.py       ffmetadata text (title/artist/album/chapters), mutagen cover
  audio/vtt.py            (text, start, end) cues -> WEBVTT
```

Rules:
- Engines never see files, SML tags, or silence. The orchestrator owns gaps between clips.
- `audio/` owns files; everything upstream works with in-memory dataclasses.
- No inline `[break]`/`[pause]` markers anywhere — gaps are a `Sentence.gap_after` float, computed
  deterministically from punctuation, so resume never depends on randomness.

## Data model

```python
class ParagraphKind(StrEnum): HEADING = "heading"; BODY = "body"; TABLE_ROW = "row"
@dataclass(frozen=True, slots=True)
class Paragraph: text: str; kind: ParagraphKind
@dataclass(frozen=True, slots=True)
class Chapter: title: str; paragraphs: tuple[Paragraph, ...]; source_ids: tuple[str, ...]
@dataclass(frozen=True, slots=True)
class Book: title: str; author: str | None; cover: bytes | None; cover_mime: str | None
            chapters: tuple[Chapter, ...]; source_sha256: str
@dataclass(frozen=True, slots=True)
class Sentence: text: str; gap_after: float; chapter_index: int
@dataclass(frozen=True, slots=True)
class AudioClip: samples: np.ndarray; sample_rate: int  # float32 mono
```

## TTS engine interface

```python
class TTSEngine(ABC):
    name: ClassVar[str]
    sample_rate: int
    max_batch: int = 1          # texts per synthesize() call
    max_concurrency: int = 1    # parallel synthesize() calls allowed
    max_chars: int = 4096       # provider limit; splitter cap stays below
    @abstractmethod
    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]: ...
    @abstractmethod
    def fingerprint(self) -> str: ...  # digest of engine+model+voice+params; part of cache key
    def close(self) -> None: ...
```

Adding an engine: create `tts/<name>.py` with a frozen config dataclass and a
`TTSEngine` subclass, register it in `tts/registry.py` (or via an
`epub_to_m4b.engines` entry point from another package), add
`[engine.<name>]` handling in `config.py`. Nothing else changes.

- API engines (`openai_compat`, `elevenlabs`, `deepgram`) share `tts/http.py`:
  retry with backoff on connect errors and 408/429/5xx, honour `Retry-After`,
  raise `TTSError` after exhausting retries. `api_key_env` names an env var —
  keys never live in config files.
- Breeze-specific guards (runaway reseed/cut, 409 busy-wait, reference voice)
  live inside `BreezeEngine`, not the ABC. See `tts/guard.py` for the pure,
  testable budget math.

## Resume

Content-addressed flat files, `os.replace` for atomic writes, no database:

```
~/.cache/epub-to-m4b/clips/<engine_fingerprint[:16]>/<sha256(text_pipeline_version, text)[:32]>.flac
<out_dir>/.work/<book_sha256[:16]>/chapters/<idx>.flac (+ .json manifest of clip keys)
```

Changing engine/voice/model starts a fresh cache directory; the old one stays
usable if you switch back. Changing the gap policy alone never re-synthesizes
— gaps are added at assembly time, not baked into cached clips.

## Tooling

```bash
make check      # ruff check + ruff format --check + mypy --strict + pytest
make format     # ruff format + ruff check --fix
uv run pytest -m gpu        # tests that need a running Breeze sidecar
uv run pytest -m network    # tests that hit a paid API
```

Both are excluded by default (`addopts` in `pyproject.toml`).

## Reference material

The full design rationale and chapter-detection algorithm live in
`epub-to-m4b.md` in the repo root — **not tracked in git**, kept for local
reference only. If you need the "why" behind a decision that isn't captured
above, check there first.
