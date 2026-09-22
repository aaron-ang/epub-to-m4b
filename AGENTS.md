# AGENTS.md

Module layout, interfaces, and conventions for contributors. Usage: [README.md](README.md).

## Pipeline

```
epub/reader ─> epub/chapters ─> text/normalize + split ─> synth/orchestrator ─> audio/assemble ─> audio/ffmpeg ─> <out_dir>/<slug>.m4b
  (Book)        (Chapter[])       (Sentence[])              │        ↕            (chapter FLAC)  └> audio/vtt ─> <out_dir>/<slug>.vtt
                                                            ▼        │
                                                          tts/*  synth/cache ◄─► <cache_dir>/clips/<fingerprint>/<key>.flac        (clips, shared across books)
                                                       (AudioClip)           ◄─► <out_dir>/.work/<book>/chapters/<idx>.{flac,json}  (assembled chapters, per book)
```

## Layout

| Path                     | Responsibility                                                                      |
|--------------------------|-------------------------------------------------------------------------------------|
| `cli.py`                 | argparse: `chapters` / `dump-text` / `convert`; m4b freshness check; atomic encode  |
| `config.py`              | Config path resolution; `[engine.*]` tables -> `AppConfig`; `E2M_CACHE_DIR`         |
| `book.py`                | `Book`, `Chapter`, `Paragraph`, `Sentence`, `AudioClip`                             |
| `errors.py`              | `EpubToM4bError`: base for user-facing errors; CLI prints `error: <message>`, exit 1 |
| `epub/reader.py`         | ebooklib -> `Book` (DC metadata, cover, spine docs)                                 |
| `epub/html.py`           | BeautifulSoup(lxml) DOM walk -> `list[Paragraph]`                                   |
| `epub/chapters.py`       | TOC -> spine mapping, heading fallback, running-header removal, stub merge          |
| `text/__init__.py`       | `TEXT_PIPELINE_VERSION`: sha256 of the docstring-stripped AST of the text pipeline  |
| `text/normalize.py`      | `normalize(text, lang="en") -> str`; dispatches to `text/lang/<lang>.py`            |
| `text/lang/english.py`   | Decades, years, ordinals, roman numerals (headings), clock, math, thousands, abbreviations |
| `text/lang/tables_en.py` | Lookup tables for `english.py`                                                      |
| `text/split.py`          | `Paragraph -> list[str]`; char cap; hard/soft/space/hard-cut; short merge           |
| `tts/base.py`            | `TTSEngine` ABC, pcm16 <-> float32 helpers, `fingerprint_digest`                    |
| `tts/registry.py`        | name -> factory(`AppConfig`); the only place API keys are read from env             |
| `tts/http.py`            | Retrying POST for API engines (backoff, `Retry-After`)                              |
| `tts/guard.py`           | Runaway-clip budget math: retry/cut limits, `max_new_tokens`, `cut_and_fade`, `apply_guard` |
| `tts/sidecar.py`         | Spawn or adopt a local HTTP TTS server; health poll; log file                       |
| `tts/breeze.py`          | `BreezeEngine`: batch endpoint, 409 wait, reference voice, guard                    |
| `tts/openai_compat.py`   | OpenAI-compatible `/v1/audio/speech`                                                |
| `tts/elevenlabs.py`      | ElevenLabs `/v1/text-to-speech/{voice}`                                             |
| `tts/deepgram.py`        | Deepgram Aura `/v1/speak`                                                           |
| `tts/fake.py`            | `SilenceEngine`, `ToneEngine` for tests and dry runs                                |
| `synth/cache.py`         | Content-addressed FLAC store, chapter manifests, atomic writes, invalidation        |
| `synth/batching.py`      | Length-sorted windows across chapters                                               |
| `synth/orchestrator.py`  | sentences -> missing -> batches -> engine -> cache; `GapPolicy`                     |
| `audio/assemble.py`      | clips + gaps -> chapter FLAC; records offsets                                       |
| `audio/ffmpeg.py`        | Command builders + subprocess: concat, ffmetadata, AAC encode, ffprobe              |
| `audio/metadata.py`      | ffmetadata text (title/artist/album/chapters); mutagen cover                        |
| `audio/vtt.py`           | `(text, start, end)` cues -> WEBVTT                                                 |

## Conventions

- Engines receive plain text only: no files, SSML, or silence. The orchestrator owns gaps.
- `audio/` owns files; everything upstream works with in-memory dataclasses.
- No inline `[break]`/`[pause]` markers. Gaps are `Sentence.gap_after`, computed deterministically from punctuation.
- API keys come from the env var named by `api_key_env`, read in `tts/registry.py` only. Never in TOML, never in engines.
- Engine-specific behaviour (Breeze guard, 409 wait, reference voice) lives in that engine's module, not the ABC.
- Adopted sidecar servers are never killed on `close()`.
- Clips, chapter FLACs, manifests, and the `.m4b` land via `synth/cache.py:atomic_replace` (temp file + `os.replace`).
- No plugin/entry-point mechanism; engines are registered in `_ENGINES`.

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

`TTSEngine(ABC)` in `tts/base.py`:

| Member                                              | Kind             | Meaning                                                        |
|-----------------------------------------------------|------------------|----------------------------------------------------------------|
| `name: ClassVar[str]`                               | attribute        | Registry key and `--engine` value                              |
| `sample_rate: int`                                  | attribute        | Output sample rate of every clip                               |
| `max_batch: int = 1`                                | attribute        | Texts per `synthesize()` call                                  |
| `max_concurrency: int = 1`                          | attribute        | Parallel `synthesize()` calls; `> 1` fans out over a thread pool |
| `synthesize(texts: Sequence[str]) -> list[AudioClip]` | abstract method | One clip per input text, same order                            |
| `fingerprint() -> str`                              | abstract method  | Digest of engine + model + voice + params; clip cache partition |
| `close() -> None`                                   | method           | Release resources; `__exit__` calls it                         |

| Engine       | Config table          | `max_batch`   | `max_concurrency` |
|--------------|-----------------------|---------------|-------------------|
| `silence`    | none                  | 1             | 1                 |
| `tone`       | none                  | 1             | 1                 |
| `breeze`     | required              | `batch_size`  | 1                 |
| `openai`     | required              | 1             | 4                 |
| `elevenlabs` | required              | 1             | 2                 |
| `deepgram`   | optional (all defaults) | 1           | 4                 |

API engines (`openai_compat`, `elevenlabs`, `deepgram`) share `tts/http.py`: retry with backoff on transport errors and 408/429/5xx, honour `Retry-After`, raise `TTSError` after exhausting retries.

Breeze keeps `breeze-server-<port>.log` and `breeze/reference_voice.{wav,txt}` under `cache_dir`. The reference wav's hash is part of the fingerprint.

## Adding an engine

1. Create `tts/<name>.py` with a frozen `<Name>Config` dataclass and a `TTSEngine` subclass.
2. Add an `AppConfig` field in `config.py` and a `_build_engine_config` call in `load_config` (accepted keys are derived from the dataclass fields; pass `required=`).
3. Add a factory to `_ENGINES` in `tts/registry.py`. Read the API key there via `_api_key`.
4. Add tests using `httpx.MockTransport`.

## Cache layout

```
<cache_dir>/clips/<engine_fingerprint[:16]>/<sha256(TEXT_PIPELINE_VERSION, text)[:32]>.flac
<out_dir>/.work/<book_sha256[:16]>/chapters/<idx:04d>.flac
<out_dir>/.work/<book_sha256[:16]>/chapters/<idx:04d>.json
```

- `cache_dir` defaults to `~/.cache/epub-to-m4b`; `E2M_CACHE_DIR` overrides it. Tests must set it to a tmp dir.
- Clips are shared across books. The chapter work dir is per book and per `out_dir`.
- The clip key excludes engine fingerprint (directory partition) and gap policy (applied at assembly).
- The chapter manifest records engine fingerprint, sample rate, clip keys, gaps, offsets, duration. Any mismatch re-assembles the chapter.
- `TEXT_PIPELINE_VERSION` hashes the docstring-stripped AST of `text/normalize.py`, `text/split.py`, `text/lang/*`. Code changes there invalidate every clip; comment/docstring/format edits do not.
- Zero-length or unreadable cache files are deleted and treated as misses.
- The `.m4b` is re-encoded when any chapter FLAC is newer than it or ffprobe cannot read it with the expected chapter count.

## Tooling

Python 3.14 (`requires-python`, `.python-version`, ruff `py314`, mypy `python_version`).

```bash
make check      # ruff check + ruff format --check + mypy --strict + pytest
make format     # ruff format + ruff check --fix
make coverage   # pytest --cov --cov-report=term-missing
make ci         # alias of make check
```

| pytest marker | Meaning                                  | Run with                   |
|---------------|------------------------------------------|----------------------------|
| `gpu`         | Needs a running Breeze sidecar and CUDA  | `uv run pytest -m gpu`     |
| `network`     | Hits a paid API                          | `uv run pytest -m network` |

Both markers are excluded by default via `addopts`. No test currently carries either; the suite runs on `silence`/`tone` and `httpx.MockTransport`.

## Tunables

Every threshold or default lives as a named module constant next to a comment explaining the mechanism. Change the constant, not a literal at the call site.

| Constant | Module | Meaning |
|----------|--------|---------|
| `DEFAULT_MAX_CHARS` | `text/split.py` | Longest clip text handed to an engine |
| `DEFAULT_TOC_DEPTH` | `epub/chapters.py` | TOC nesting level that starts a chapter |
| `DEFAULT_MIN_CHARS` | `epub/chapters.py` | Body chars below which a chapter is a stub and merges |
| `MIN_TOC_COVERAGE` | `epub/chapters.py` | Fraction of text docs the TOC must cover before it is trusted over headings |
| `MAX_TITLE_BYTES` | `epub/chapters.py` | Longest heading text accepted as a chapter title |
| `RUNNING_HEADER_MIN_DOCS` | `epub/chapters.py` | Docs a repeated first line must appear in to count as a running header |
| `MIN_HEADING_KEY_CHARS` | `epub/chapters.py` | Shortest normalised heading key that can match a TOC label |
| `AAC_BITRATE` | `audio/ffmpeg.py` | AAC bitrate for the `.m4b` |
| `_MIN_CHAPTER_SECONDS` | `audio/assemble.py` | Floor on assembled chapter length |
| `GapPolicy` defaults | `synth/orchestrator.py` | Silence after sentence / clause cut / paragraph / heading |
| `RetryPolicy` defaults | `tts/http.py` | Retry count, doubling backoff, Retry-After cap |
| `BreezeConfig` defaults | `tts/breeze.py` | Sidecar port, cfg scale, seed, batch size |
| `_REFERENCE_TIMEOUT_SECONDS`, `_BATCH_TIMEOUT_SECONDS` | `tts/breeze.py` | HTTP timeouts for reference-voice and batch POSTs |
| `_BUSY_STATUS`, `_BUSY_RETRIES`, `_BUSY_WAIT_SECONDS` | `tts/breeze.py` | 409 busy handling: status, attempts, wait between attempts |
| `CLIP_BASE_SECONDS`, `CLIP_SECONDS_PER_CHAR` | `tts/guard.py` | Duration budget that triggers a reseed retry |
| `CUT_SECONDS_PER_CHAR` | `tts/guard.py` | Duration budget beyond which a clip is truncated |
| `TOKENS_PER_SECOND` | `tts/guard.py` | Codec audio tokens per second, for the server-side token cap |
| `TOKEN_CAP_SLACK` | `tts/guard.py` | Multiplier loosening the server-side token cap |
| `FADE_SECONDS` | `tts/guard.py` | Fade-out applied to a truncated clip |
| `RUNAWAY_RETRIES` | `tts/guard.py` | Reseed attempts before cutting |
| `_HEALTH_TIMEOUT_SECONDS`, `_POLL_INTERVAL_SECONDS` | `tts/sidecar.py` | `/health` request timeout and poll spacing |
| `_STARTUP_TIMEOUT_SECONDS`, `_TERMINATE_TIMEOUT_SECONDS` | `tts/sidecar.py` | Wait for server ready; wait for graceful exit before kill |

## Reference material

`epub-to-m4b.md` in the repo root is untracked local reference. Never stage it.
