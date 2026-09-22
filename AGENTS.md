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
  cli.py                 argparse: chapters / dump-text / convert; m4b freshness check, atomic encode
  config.py              --config > E2M_CONFIG > ~/.config/epub-to-m4b/config.toml; [engine.*] tables -> AppConfig
  book.py               Book, Chapter, Paragraph, Sentence, AudioClip
  epub/reader.py         ebooklib -> Book (DC metadata, cover, spine docs)
  epub/html.py            BeautifulSoup(lxml) DOM walk -> list[Paragraph]
  epub/chapters.py        TOC->spine mapping, heading fallback, running-header removal, stub merge
  text/__init__.py       TEXT_PIPELINE_VERSION: sha256 of the text pipeline's own sources
  text/normalize.py      normalize(text, lang) -> str, dispatch to text/lang/<lang>.py
  text/lang/english.py    decades, years, ordinals, roman (headings), clock, math, thousands, abbreviations
  text/lang/tables_en.py  lookup tables for english.py
  text/split.py           Paragraph -> list[str]; char cap; hard/soft/space/hard-cut; short merge
  tts/base.py            TTSEngine ABC, pcm16<->float32 helpers, fingerprint_digest
  tts/registry.py         name -> factory(AppConfig); reads API keys from env, nothing else does
  tts/http.py             retrying POST helper for API engines (backoff, Retry-After); rate limiting left to the server
  tts/guard.py            pure runaway-clip budget math: retry/cut limits, max_new_tokens, cut_and_fade, apply_guard
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
- Engines never see files, SSML tags, or silence. The orchestrator owns gaps between clips.
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
class AudioClip: samples: npt.NDArray[np.float32]; sample_rate: int  # mono, shape (n,); .seconds property
```

## TTS engine interface

```python
class TTSEngine(ABC):
    name: ClassVar[str]
    sample_rate: int
    max_batch: int = 1          # texts per synthesize() call
    max_concurrency: int = 1    # parallel synthesize() calls allowed
    max_chars: int = 4096       # provider limit; the splitter's 125-char cap stays far below
    @abstractmethod
    def synthesize(self, texts: Sequence[str]) -> list[AudioClip]: ...
    @abstractmethod
    def fingerprint(self) -> str: ...  # digest of engine+model+voice+params; cache directory partition
    def close(self) -> None: ...
    # __enter__/__exit__ call close(); the CLI uses `with engine:`
```

Engines: `silence`, `tone` (no config), `breeze`, `openai`, `elevenlabs`,
`deepgram`. The orchestrator sizes each `synthesize()` call by `max_batch`
and fans batches over a thread pool when `max_concurrency > 1` (the hosted
APIs); everything else runs sequentially.

Adding an engine: create `tts/<name>.py` with a frozen config dataclass and a
`TTSEngine` subclass; add an `AppConfig` field plus a `_build_engine_config`
call in `config.py` (it derives accepted keys from the dataclass fields and
rejects unknown/missing ones); add a factory to `_ENGINES` in
`tts/registry.py`. There is no plugin/entry-point mechanism.

- API engines (`openai_compat`, `elevenlabs`, `deepgram`) share `tts/http.py`:
  retry with backoff on transport errors and 408/429/5xx, honour `Retry-After`,
  raise `TTSError` after exhausting retries. `api_key_env` names an env var —
  keys never live in config files; the registry factory reads the variable
  and passes the key in.
- Breeze-specific guards (runaway reseed/cut, 409 busy-wait, reference voice)
  live inside `BreezeEngine`, not the ABC. See `tts/guard.py` for the pure,
  testable budget math. `tts/sidecar.py` spawns the configured `command`
  (plus `--host/--port`) or adopts a server already answering `/health` on
  the port; adopted servers are never killed on `close()`. Breeze keeps its
  server log and `breeze/reference_voice.{wav,txt}` under the cache dir; the
  reference wav's hash is part of the fingerprint.

## Resume

Content-addressed flat files, `os.replace` for atomic writes, no database:

```
<cache_dir>/clips/<engine_fingerprint[:16]>/<sha256(text_pipeline_version, text)[:32]>.flac
<out_dir>/.work/<book_sha256[:16]>/chapters/<idx:04d>.flac (+ .json manifest: fingerprint, sample rate, clip keys, gaps, offsets, duration)
```

`cache_dir` is `~/.cache/epub-to-m4b`, overridable with `E2M_CACHE_DIR`
(tests set it to a tmp dir). Clips are shared across books; the chapter
work dir is per book and per `out_dir`.

Changing engine/voice/model starts a fresh cache directory; the old one stays
usable if you switch back. Changing the gap policy alone never re-synthesizes
— gaps are added at assembly time, not baked into cached clips — but it does
re-assemble chapters, since the manifest records gaps. `TEXT_PIPELINE_VERSION`
hashes the text pipeline's own source files, so editing `text/normalize.py`,
`text/split.py`, or `text/lang/*` invalidates every cached clip without a
manual bump. A zero-length or unreadable cache file is deleted and treated as
a miss. The final `.m4b` is only re-encoded when a chapter flac is newer than
it or ffprobe cannot read it with the expected chapter count.

## Tooling

Python 3.14 (`requires-python`, `.python-version`, ruff `py314`, mypy 3.14).

```bash
make check      # ruff check + ruff format --check + mypy --strict + pytest
make format     # ruff format + ruff check --fix
```

`pyproject.toml` registers `gpu` (needs a running Breeze sidecar and CUDA)
and `network` (hits a paid API) pytest markers and excludes both by default
via `addopts`; run them with `uv run pytest -m gpu` / `-m network`. No test
currently carries either marker — the suite runs entirely on `silence`/`tone`
and `httpx.MockTransport`.

## Reference material

The full design rationale and chapter-detection algorithm live in
`epub-to-m4b.md` in the repo root — **not tracked in git**, kept for local
reference only. If you need the "why" behind a decision that isn't captured
above, check there first.
