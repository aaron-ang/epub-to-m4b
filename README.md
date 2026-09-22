# epub-to-m4b

Turn an EPUB into an M4B audiobook with chapter markers, cover art, and a WebVTT transcript.

## Install

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- `ffmpeg` and `ffprobe` on `PATH`

```bash
uv sync
```

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

Only `[engine.<name>]` tables are read. Unknown keys and missing required keys are errors.
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

`command` is the argv that starts the server; `--host`/`--port` are appended. A server already listening on `port` is adopted instead of spawned.

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

## Development

```bash
make check   # ruff check, ruff format --check, mypy --strict, pytest
```

See [AGENTS.md](AGENTS.md) for module layout and conventions.
