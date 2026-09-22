# epub-to-m4b

Turn an EPUB into an M4B audiobook with chapter markers, cover art, and a WebVTT transcript.

[![CI](https://github.com/aaron-ang/epub-to-m4b/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-ang/epub-to-m4b/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](pyproject.toml)

## Quick start

Requires Python 3.14, [uv](https://docs.astral.sh/uv/), and `ffmpeg` + `ffprobe` on `PATH`.

```bash
git clone https://github.com/aaron-ang/epub-to-m4b && cd epub-to-m4b && uv sync
uv run epub-to-m4b chapters book.epub
uv run epub-to-m4b convert book.epub --engine silence -o out/
```

`silence` renders a silent `.m4b`. Speech needs an `[engine.<name>]` table in `config.toml`; see [Configuration](#configuration).

## Example

```
$ uv run epub-to-m4b chapters outliers.epub
Outliers the story of success — Gladwell Malcolm (14 chapters)
  #  title                                               paras    chars  docs
  1  INTRODUCTION — The Roseto Mystery                      27    12584     3
  2  CHAPTER ONE                                            65    26263     8
  3  CHAPTER TWO                                            76    33493     7
  4  THE 10,000-HOUR RULE                                   26     9883     1
  5  CHAPTER THREE                                          79    32187     7
  6  CHAPTER FOUR                                           97    38108     8
  7  CHAPTER FIVE                                           44    20923     7
  8  Lesson Number Two:Demographic Luck                     97    45529    10
  …
```

Chapter markers in the rendered file (`ffprobe -show_chapters out/outliers-the-story-of-success.m4b`), first 5 of 14:

```
00:00:00 → INTRODUCTION — The Roseto Mystery
00:14:12 → CHAPTER ONE
00:43:47 → CHAPTER TWO
01:22:00 → THE 10,000-HOUR RULE
01:34:37 → CHAPTER THREE
```

## Engines

| Engine       | Runs where                                        | Needs                                                            | Cost                                 | Notes                                          |
|--------------|---------------------------------------------------|------------------------------------------------------------------|--------------------------------------|------------------------------------------------|
| `breeze`     | Local GPU sidecar (spawned or adopted on `port`)  | Model weights + `breeze-infer-api` server `command` in config    | Free                                 | Batched (`batch_size`); resume-friendly        |
| `openai`     | Any OpenAI-compatible `/v1/audio/speech` endpoint | API key in the env var named by `api_key_env`                    | Per character, provider pricing      | `base_url`, `model`, `voice` required          |
| `elevenlabs` | Cloud                                             | API key in the env var named by `api_key_env`                    | Per character, provider pricing      | `voice_id` required                            |
| `deepgram`   | Cloud                                             | API key in the env var named by `api_key_env`                    | Per character, provider pricing      | Config table optional                          |
| `silence`    | Local                                             | Nothing                                                          | Free                                 | Pipeline dry runs; silent clips                |
| `tone`       | Local                                             | Nothing                                                          | Free                                 | Pipeline dry runs; sine-tone clips             |

## Usage

| Subcommand  | Purpose                                                        |
|-------------|----------------------------------------------------------------|
| `chapters`  | List detected chapters (title, paragraph/char counts, docs)    |
| `dump-text` | Print chapter titles and paragraphs as they will be read       |
| `convert`   | Render `<slug>.m4b` and `<slug>.vtt`                           |

`convert` flags:

| Flag                 | Required | Meaning                                                                          |
|----------------------|----------|----------------------------------------------------------------------------------|
| `--engine NAME`      | yes      | `breeze`, `deepgram`, `elevenlabs`, `openai`, `silence`, `tone`                  |
| `-o, --out-dir DIR`  | yes      | Output directory                                                                 |
| `--config PATH`      | no       | TOML config file (default: `$E2M_CONFIG`, then `~/.config/epub-to-m4b/config.toml`) |

`dump-text` flags:

| Flag           | Meaning                                                  |
|----------------|----------------------------------------------------------|
| `--chapter N`  | Only chapter `N` (1-based)                               |
| `--normalized` | Run each paragraph through text normalization            |
| `--split`      | Also show sentence boundaries (implies `--normalized`)   |

Flags shared by all subcommands:

| Flag            | Default | Meaning                                    |
|-----------------|---------|--------------------------------------------|
| `--toc-depth N` | `1`     | Deepest TOC level whose entries start chapters |
| `--min-chars N` | `200`   | Chapters with fewer body chars merge into the next one (a trailing stub into the previous) |

`--version` prints the package version.

```bash
uv run epub-to-m4b chapters book.epub
uv run epub-to-m4b dump-text book.epub --chapter 3 --split
uv run epub-to-m4b convert book.epub --engine silence -o /tmp/dry-run
OPENAI_API_KEY=... uv run epub-to-m4b convert book.epub --engine openai -o ~/audiobooks
```

## Configuration

Config file resolution, first match wins:

1. `--config PATH` (must exist)
2. `E2M_CONFIG` (must exist)
3. `~/.config/epub-to-m4b/config.toml` (optional)

Only `[engine.<name>]` tables are read. Unknown keys, missing required keys, and values of the wrong TOML type are errors.
`silence` and `tone` take no configuration.

| Env var              | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `E2M_CONFIG`         | Config file path                                         |
| `E2M_CACHE_DIR`      | Cache root (default `~/.cache/epub-to-m4b`)              |
| `OPENAI_API_KEY`     | API key for `openai` (name set by `api_key_env`)         |
| `ELEVENLABS_API_KEY` | API key for `elevenlabs` (name set by `api_key_env`)     |
| `DEEPGRAM_API_KEY`   | API key for `deepgram` (name set by `api_key_env`)       |

`[engine.breeze]`

| Key           | Type     | Default                                                                 | Required |
|---------------|----------|-------------------------------------------------------------------------|----------|
| `weights_dir` | string   |                                                                         | ✓        |
| `command`     | string[] |                                                                         | ✓        |
| `port`        | int      | `7861`                                                                  |          |
| `batch_size`  | int      | `64`                                                                    |          |
| `instruction` | string   | `"A clear, neutral adult narrator voice with a calm, steady reading pace."` |      |
| `cfg_scale`   | float    | `4.0`                                                                   |          |
| `seed`        | int      | `42`                                                                    |          |

`command` is the argv that starts the server; `weights_dir` and `--host`/`--port` are appended. A server already listening on `port` is adopted instead of spawned.

`[engine.openai]`

| Key           | Type   | Default              | Required |
|---------------|--------|----------------------|----------|
| `base_url`    | string |                      | ✓        |
| `model`       | string |                      | ✓        |
| `voice`       | string |                      | ✓        |
| `speed`       | float  | `1.0`                |          |
| `api_key_env` | string | `"OPENAI_API_KEY"`   |          |

`[engine.elevenlabs]`

| Key           | Type   | Default                       | Required |
|---------------|--------|-------------------------------|----------|
| `voice_id`    | string |                               | ✓        |
| `model_id`    | string | `"eleven_multilingual_v2"`    |          |
| `base_url`    | string | `"https://api.elevenlabs.io"` |          |
| `api_key_env` | string | `"ELEVENLABS_API_KEY"`        |          |

`[engine.deepgram]` (table optional)

| Key           | Type   | Default                      | Required |
|---------------|--------|------------------------------|----------|
| `model`       | string | `"aura-2-thalia-en"`         |          |
| `base_url`    | string | `"https://api.deepgram.com"` |          |
| `api_key_env` | string | `"DEEPGRAM_API_KEY"`         |          |

```toml
[engine.breeze]
weights_dir = "/path/to/breeze-tts-2"
command = ["uv", "run", "--directory", "/path/to/breeze-tts", "breeze-infer-api"]

[engine.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini-tts"
voice = "alloy"

[engine.elevenlabs]
voice_id = "..."

[engine.deepgram]
model = "aura-2-thalia-en"
```

## Output

- `<out_dir>/<slug>.m4b` — chapter markers, cover, title/author tags
- `<out_dir>/<slug>.vtt` — sentence-level transcript

`<slug>` is derived from the book title.

## Resume

- Rerun the same command to resume an interrupted or partial render.
- Sentence clips are cached as FLAC under `<cache_dir>/clips/<engine-fingerprint>/`, shared across books. Assembled chapters are cached under `<out_dir>/.work/<book-id>/chapters/`.
- Changing engine, voice, model, or other audio settings, or editing the text pipeline source (`text/normalize.py`, `text/split.py`, `text/lang/*`), stops old clips being reused. Old clips stay on disk.
- A missing or damaged `.m4b` is rebuilt; an up-to-date one is kept and only the `.vtt` is rewritten.

## Troubleshooting

| Symptom                                                                                      | Fix                                                                                                                          |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `error: required on PATH but not found: ffmpeg, ffprobe`                                     | Install ffmpeg; both `ffmpeg` and `ffprobe` must be on `PATH`                                                                |
| `error: engine 'breeze' selected but no [engine.breeze] table was found - ...`               | Add the `[engine.breeze]` table to the config file, or pass `--config PATH` to a file that has it                            |
| `error: environment variable OPENAI_API_KEY is not set (needed for engine 'openai')`         | `export` the variable named by that engine's `api_key_env`                                                                   |
| `error: server on port 7861 did not become healthy within 180s; see log at ...`               | Read `<cache_dir>/breeze-server-<port>.log`; check `command`, `weights_dir`, and whether another process holds `port`         |
| `Breeze server busy, waiting for the running inference to finish` (stderr, once per batch)    | Another client holds the server's single inference slot; the run waits (up to 60 retries, 5 s apart) and continues on its own |
| Resume re-synthesizes every sentence                                                         | Engine settings changed (new fingerprint) or code in `text/normalize.py`, `text/split.py`, `text/lang/*` changed (new `TEXT_PIPELINE_VERSION`) |

## Development

```bash
make check   # ruff check, ruff format --check, mypy --strict, pytest
```

## Contributing

Workflow and checks: [CONTRIBUTING.md](CONTRIBUTING.md). Module layout and conventions: [AGENTS.md](AGENTS.md).

## License

[MIT](LICENSE).
