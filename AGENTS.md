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

| Path                 | Responsibility                                                 |
|----------------------|----------------------------------------------------------------|
| `cli.py`             | argparse: `chapters` / `dump-text` / `convert`; atomic encode  |
| `config.py`          | Config path resolution; `[engine.*]` tables -> `AppConfig`     |
| `book.py`            | `Book`, `Chapter`, `Paragraph`, `Sentence`, `AudioClip`        |
| `errors.py`          | `EpubToM4bError`: base for user-facing errors; CLI exit 1      |
| `epub/`              | ebooklib + BeautifulSoup -> `Book`; TOC/heading chaptering     |
| `text/`              | `normalize.py`: NeMo (per language, output used as is); `split.py`: sentence split |
| `tts/`               | `TTSEngine` ABC, registry, engines, HTTP retry, sidecar, guard |
| `synth/`             | Clip cache, batching, orchestrator                             |
| `audio/`             | Chapter assembly, ffmpeg, metadata, VTT                        |
| `tests/`             | Mirrors `src/`; `tests/helpers.py` shared builders             |
| `.github/workflows/` | CI + release                                                   |
| `pyproject.toml`     | Deps, pixi workspace + tasks, ruff, mypy, pytest markers + `addopts` |
| `pixi.lock`          | Locked conda-forge + PyPI environment for every pixi platform  |

## Conventions

- Engines receive plain text only: no files, SSML, or silence. The orchestrator owns gaps.
- `audio/` owns files; everything upstream works with in-memory dataclasses.
- No inline `[break]`/`[pause]` markers. Gaps are `Sentence.gap_after`, computed deterministically from punctuation.
- API keys come from the env var named by `api_key_env`, read in `tts/registry.py` only. Never in TOML, never in engines.
- Engine-specific behaviour lives in that engine's module, not the ABC. Sidecar lifecycle (spawn/adopt, health poll, busy-wait, child env) lives in `tts/sidecar.py` behind `SidecarPolicy` and is shared by every sidecar engine.
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
| `breeze`     | required              | `min(batch_size, max_batch_texts)` | 1            |
| `openai`     | required              | 1             | 4                 |
| `elevenlabs` | required              | 1             | 2                 |
| `deepgram`   | optional (all defaults) | 1           | 4                 |

Sidecar engines call `start_or_adopt(command, port, log_path=..., policy=SidecarPolicy(...), env=...)` and wrap busy-prone POSTs in `post_until_free(send, policy=..., on_busy=...)`.

API engines (`openai_compat`, `elevenlabs`, `deepgram`) share `tts/http.py`: retry with backoff on transport errors and 408/429/5xx, honour `Retry-After`, raise `TTSError` after exhausting retries.

Sidecar engines write `<name>-server-<port>.log` under `cache_dir`.

Engine notes:

| Engine                             | Notes                                                                                                                                          |
|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `breeze`                           | Model is the last `command` arg (local dir or HF repo id, resolved by breeze-tts); server started with `breeze-tts-server`; `GET /v1/model` reports `frame_rate`, `model_digest`, `max_new_tokens`, `max_batch_texts` (`ServerInfo`; 404 or bad field = wrong server); fingerprint uses `model_digest` + reference wav hash; token cap from `frame_rate`, clamped to `max_new_tokens`; runaway guard from `tts/guard.py` via `RunawayPolicy`; 409 busy-wait via `SidecarPolicy` |
| `openai`, `elevenlabs`, `deepgram` | `tts/http.py` retry                                                                                                                            |
| `silence`, `tone`                  | none                                                                                                                                           |

## Adding an engine

1. Create `tts/<name>.py` with a frozen `<Name>Config` dataclass and a `TTSEngine` subclass.
2. Sidecar engine? Use `start_or_adopt` + `post_until_free` from `tts/sidecar.py`; pass engine-specific env via `env=`.
3. Add an `AppConfig` field in `config.py` and a `_build_engine_config` call in `load_config` (accepted keys are derived from the dataclass fields; pass `required=`).
4. Add a factory to `_ENGINES` in `tts/registry.py`. Read the API key there via `_api_key`.
5. Add tests using `httpx.MockTransport`.

## Cache layout

```
<cache_dir>/clips/<engine_fingerprint[:16]>/<sha256(text)[:32]>.flac
<cache_dir>/nemo/<nemo-text-processing version>/<lang>/*.far
<out_dir>/.work/<book_sha256[:16]>/chapters/<idx:04d>.flac
<out_dir>/.work/<book_sha256[:16]>/chapters/<idx:04d>.json
<out_dir>/.work/<book_sha256[:16]>/encode.json
```

- `cache_dir` defaults to `~/.cache/epub-to-m4b`; `E2M_CACHE_DIR` overrides it. Tests must set it to a tmp dir.
- `nemo/` holds NeMo's compiled grammars, one dir per language, built by the first normalization in that language that finds none. `E2M_NEMO_CACHE_DIR` replaces `<cache_dir>/nemo`; the test session points it at pytest's cache dir and builds the English normalizer once.
- Clips are shared across books. The chapter work dir is per book and per `out_dir`.
- `text` is the exact string passed to `engine.synthesize`. Gaps are added at assembly, never baked into a clip.
- The chapter manifest records engine fingerprint, sample rate, clip keys, gaps, offsets, duration. Any mismatch re-assembles the chapter.
- Zero-length or unreadable cache files are deleted and treated as misses.
- The `.m4b` is re-encoded when any chapter FLAC is newer than it, ffprobe cannot read it with the expected chapter count, or `encode.json` is missing or its digest differs from the current encode arguments.

## Tooling

Python 3.14 (`requires-python`, pixi `python`, ruff `py314`, mypy `python_version`). [pixi](https://pixi.sh) manages the environment: Python, `pynini`/OpenFst and `editdistance` from conda-forge, everything else from PyPI, the project itself editable; dev tools come from the `dev` feature, part of the default environment.

```bash
pixi install        # create .pixi/envs/default from pixi.lock
pixi run check      # ruff check + ruff format --check + mypy --strict + pytest
pixi run format     # ruff format + ruff check --fix
pixi run coverage   # pytest --cov --cov-report=term-missing
```

| pytest marker | Meaning                                  | Run with                   |
|---------------|------------------------------------------|----------------------------|
| `sidecar`     | Needs a running local TTS sidecar server | `pixi run pytest -m sidecar` |
| `network`     | Hits a paid API                          | `pixi run pytest -m network` |

Both markers are excluded via `addopts`; no test carries either. The suite runs on `silence`/`tone` and `httpx.MockTransport`.

| Workflow                                | Trigger                     | Does                                                                                   |
|-----------------------------------------|-----------------------------|----------------------------------------------------------------------------------------|
| `.github/workflows/ci.yml`              | push to `main`, PR          | `pixi run check`, `pixi exec hatch build`, installs the built wheel with pip over conda-forge `pynini` and runs `--version` |
| `.github/workflows/release.yml`         | push to `main`; `workflow_dispatch` with `tag` | `release-please` job opens/updates the release PR from Conventional Commits and, on merge, tags `vX.Y.Z` + creates a GitHub Release; `publish` job then builds with `pixi exec hatch build` at that tag and uploads to PyPI via Trusted Publishing (OIDC, environment `pypi`). Versions in `.release-please-manifest.json`, config in `release-please-config.json` |

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
| `LOUDNESS_TARGET_LUFS` | `audio/ffmpeg.py` | Integrated loudness `loudnorm` normalizes the `.m4b` to |
| `_MIN_CHAPTER_SECONDS` | `audio/assemble.py` | Floor on assembled chapter length |
| `GapPolicy` defaults | `synth/orchestrator.py` | Silence after sentence / clause cut / paragraph / heading |
| `RetryPolicy` defaults | `tts/http.py` | Retry count, backoff base (doubles per retry), Retry-After cap |
| `BreezeConfig` defaults | `tts/breeze.py` | Sidecar port, cfg scale, seed, batch size (clamped to server `max_batch_texts`) |
| `_SIDECAR_ENV` | `tts/breeze.py` | Env vars the Breeze server child gets when spawned (`TRITON_PTXAS_PATH`) |
| `RunawayPolicy` defaults | `tts/guard.py` | Retry limit (base + per-char seconds), token cap slack over it (also the cut limit), reseed attempts |
| `FADE_SECONDS` | `tts/guard.py` | Fade-out applied to a truncated clip |
| `SidecarPolicy` defaults | `tts/sidecar.py` | Health/poll/startup/terminate timeouts, single-request and batch timeouts, busy status + total busy wait + retry spacing |

## Reference material

`epub-to-m4b.md` in the repo root is untracked local reference. Never stage it.
