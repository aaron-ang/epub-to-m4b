# epub-to-m4b

Turn an EPUB into an M4B audiobook with chapter markers, cover art, and a WebVTT transcript.

[![CI](https://github.com/aaron-ang/epub-to-m4b/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-ang/epub-to-m4b/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/epub-to-m4b.svg)](https://pypi.org/project/epub-to-m4b/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/aaron-ang/epub-to-m4b/blob/main/LICENSE)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://github.com/aaron-ang/epub-to-m4b/blob/main/pyproject.toml)

## Sample

*Crime and Punishment*, Part 1, Chapter 1:

https://github.com/user-attachments/assets/5154b4cd-e66a-4316-8ab6-6c798d55593c

## Quick start

Requires [pixi](https://pixi.sh) and `ffmpeg` + `ffprobe` on `PATH`. pixi installs Python 3.14 and the rest. To install from PyPI instead, see the [FAQ](https://github.com/aaron-ang/epub-to-m4b/blob/main/README.md#faq).

```bash
git clone https://github.com/aaron-ang/epub-to-m4b && cd epub-to-m4b && pixi install
pixi run epub-to-m4b chapters book.epub
pixi run epub-to-m4b convert book.epub --engine silence -o out/
```

`silence` renders a silent `.m4b`. Speech needs an `[engine.<name>]` table in the config file:

```toml
# ~/.config/epub-to-m4b/config.toml
[engine.openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4o-mini-tts"
voice = "alloy"
```

```bash
export OPENAI_API_KEY=...
pixi run epub-to-m4b convert book.epub --engine openai -o out/
```

## Engines

| Engine            | Runs where                                                   | Needs                                         |
|-------------------|--------------------------------------------------------------|-----------------------------------------------|
| `openai`          | Any OpenAI-compatible `/v1/audio/speech` endpoint            | API key; billed per character                 |
| `deepgram`        | Cloud                                                        | API key; billed per character                 |
| `elevenlabs`      | Cloud                                                        | API key; billed per character                 |
| `breeze`          | Self-hosted GPU: `breeze-tts-server`                         | Model weights                                 |
| `drama`           | Self-hosted GPU: Fish Audio Drama 3 on vLLM-Omni             | ~48 GB GPU memory, a reference voice to clone |
| `silence`, `tone` | Local                                                        | Nothing; pipeline dry runs                    |

## Usage

| Subcommand  | Purpose                                                     |
|-------------|-------------------------------------------------------------|
| `chapters`  | List detected chapters (title, paragraph/char counts, docs) |
| `dump-text` | Print chapter titles and paragraphs as they will be read    |
| `convert`   | Render `<slug>.m4b` and `<slug>.vtt`                        |

| `convert` flag      | Required | Meaning                                                                             |
|---------------------|----------|-------------------------------------------------------------------------------------|
| `--engine NAME`     | yes      | `breeze`, `deepgram`, `drama`, `elevenlabs`, `openai`, `silence`, `tone`            |
| `-o, --out-dir DIR` | yes      | Output directory                                                                    |
| `--config PATH`     | no       | TOML config file (default: `$E2M_CONFIG`, then `~/.config/epub-to-m4b/config.toml`) |

`<subcommand> --help` lists every flag. Rerun the same `convert` to resume an interrupted render.

## Configuration

Config file resolution, first match wins: `--config PATH`, then `E2M_CONFIG` (both must exist), then `~/.config/epub-to-m4b/config.toml` (optional). Only `[engine.<name>]` tables are read; unknown keys, missing required keys, and wrong TOML types are errors.

| Env var              | Purpose                                              |
|----------------------|------------------------------------------------------|
| `E2M_CONFIG`         | Config file path                                     |
| `E2M_CACHE_DIR`      | Cache root (default `~/.cache/epub-to-m4b`)          |
| `OPENAI_API_KEY`     | API key for `openai` (name set by `api_key_env`)     |
| `ELEVENLABS_API_KEY` | API key for `elevenlabs` (name set by `api_key_env`) |
| `DEEPGRAM_API_KEY`   | API key for `deepgram` (name set by `api_key_env`)   |

`[engine.openai]`

| Key           | Type   | Default            | Required |
|---------------|--------|--------------------|----------|
| `base_url`    | string |                    | ✓        |
| `model`       | string |                    | ✓        |
| `voice`       | string |                    | ✓        |
| `speed`       | float  | `1.0`              |          |
| `api_key_env` | string | `"OPENAI_API_KEY"` |          |

`[engine.deepgram]` (table optional)

| Key           | Type   | Default                      | Required |
|---------------|--------|------------------------------|----------|
| `model`       | string | `"aura-2-thalia-en"`         |          |
| `base_url`    | string | `"https://api.deepgram.com"` |          |
| `api_key_env` | string | `"DEEPGRAM_API_KEY"`         |          |

`[engine.elevenlabs]`

| Key           | Type   | Default                       | Required |
|---------------|--------|-------------------------------|----------|
| `voice_id`    | string |                               | ✓        |
| `model_id`    | string | `"eleven_multilingual_v2"`    |          |
| `base_url`    | string | `"https://api.elevenlabs.io"` |          |
| `api_key_env` | string | `"ELEVENLABS_API_KEY"`        |          |

`[engine.breeze]`

| Key           | Type     | Default                                                                     | Required |
|---------------|----------|-----------------------------------------------------------------------------|----------|
| `command`     | string[] |                                                                             | ✓        |
| `port`        | int      | `7861`                                                                      |          |
| `batch_size`  | int      | `64`                                                                        |          |
| `instruction` | string   | `"A clear, neutral adult narrator voice with a calm, steady reading pace."` |          |
| `cfg_scale`   | float    | `4.0`                                                                       |          |
| `seed`        | int      | `42`                                                                        |          |

`command` starts `breeze-tts-server` (from breeze-tts) and ends with the model: a local dir or a Hugging Face repo id (plus `--revision`).

`[engine.drama]`

| Key               | Type     | Default               | Required |
|-------------------|----------|-----------------------|----------|
| `command`         | string[] |                       | ✓        |
| `reference_audio` | string   |                       | ✓        |
| `reference_text`  | string   |                       | ✓        |
| `model`           | string   | `"fishaudio/drama-3"` |          |
| `port`            | int      | `8091`                |          |
| `batch_size`      | int      | `64`                  |          |
| `seed`            | int      | `42`                  |          |

`command` starts `vllm serve <model> --omni` from [vLLM-Omni](https://github.com/vllm-project/vllm-omni). The narrator voice is cloned from `reference_audio` (WAV, FLAC or MP3, up to 30 s) and `reference_text`, its exact transcript. Drama 3 weights are under the Fish Audio Community License.

For both self-hosted engines, `--host`/`--port` are appended to `command`, and a server already listening on `port` is adopted instead of spawned.

```toml
[engine.breeze]
command = [
  "uv", "run", "--directory", "/path/to/breeze-tts", "breeze-tts-server",
  "BreezeBlue/Breeze-TTS-2",
]

[engine.drama]
command = [
  "/path/to/vllm-omni/.venv/bin/vllm", "serve", "fishaudio/drama-3", "--omni",
  "--init-timeout", "1800", "--stage-init-timeout", "1800",
]
reference_audio = "~/voices/narrator.wav"
reference_text = "The exact words spoken in narrator.wav."
```

## Output

- `<out_dir>/<slug>.m4b`: AAC at 96 kbps, loudness-normalized to -16 LUFS; chapter markers, cover, title/author tags
- `<out_dir>/<slug>.vtt`: sentence-level transcript

`<slug>` is derived from the book title.

## FAQ

<details>
<summary>How do I install from PyPI?</summary>

Text normalization uses [NeMo text processing](https://github.com/NVIDIA/NeMo-text-processing), which needs `pynini` 2.1.6.post1. PyPI has no Python 3.14 wheel of it, and no Linux aarch64 or macOS wheel of any pynini version, so take `pynini` from conda-forge and the package from PyPI:

```bash
pixi init e2m && cd e2m
pixi add python=3.14 pynini=2.1.6.post1 editdistance
pixi add --pypi epub-to-m4b
pixi run epub-to-m4b chapters book.epub
```

</details>

<details>
<summary>What does <code>chapters</code> show?</summary>

```
$ pixi run epub-to-m4b chapters outliers.epub
Outliers the story of success — Gladwell Malcolm (14 chapters)
  #  title                                               paras    chars  docs
  1  INTRODUCTION — The Roseto Mystery                      27    12584     3
  2  CHAPTER ONE                                            65    26263     8
  3  CHAPTER TWO                                            76    33493     7
  4  THE 10,000-HOUR RULE                                   26     9883     1
  …
```

Each detected chapter becomes a chapter marker in the `.m4b`.

</details>

<details>
<summary>Chapters are split wrong. What can I change?</summary>

- `--toc-depth N` (default `1`): deepest TOC level whose entries start chapters.
- `--min-chars N` (default `200`): chapters with fewer body chars merge into the next one (a trailing stub into the previous).
- `dump-text --chapter N --split` shows chapter `N` as the narrator will read it, normalized and split into sentences. `--max-chars` sets where long sentences are cut (default 125); set it to the engine's limit to see its clips.

</details>

<details>
<summary>What does resume reuse?</summary>

- Sentence clips are cached as FLAC under `<cache_dir>/clips/<engine-fingerprint>/`, shared across books. Assembled chapters are cached under `<out_dir>/.work/<book-id>/chapters/`.
- Changing engine, voice, model, or other audio settings stops old clips being reused. Old clips stay on disk.
- Clips are keyed by the exact sentence text. A text pipeline change (`text/normalize.py`, `text/split.py`, a new `nemo-text-processing`) re-synthesizes only sentences whose text it changed.
- NeMo compiles a language's grammars on first use into `<cache_dir>/nemo/<version>/<lang>/` (`E2M_NEMO_CACHE_DIR` replaces `<cache_dir>/nemo`).
- A missing or damaged `.m4b`, or one encoded with different settings, is rebuilt; an up-to-date one is kept and only the `.vtt` is rewritten.

</details>

<details>
<summary>Why is a sentence cut in two?</summary>

A clip is one sentence, or several short ones packed together, so it always ends where a sentence ends. Only a sentence longer than the engine accepts is cut, at a comma where possible: about 125 characters for `silence`/`tone`, up to 300 for `breeze`, 1000 for the API engines, and about 1200 for `drama` at its default 8192-token context (what fits after the reference voice, in any script).

</details>

<details>
<summary>How do the self-hosted servers behave?</summary>

- Breeze: `batch_size` is clamped to the server's reported limit. The server runs one inference at a time; a second client waits for it.
- Drama: a cold start takes about ten minutes. The reference voice is uploaded once, named after its audio and transcript, so a changed file or transcript uploads a new voice. `batch_size` sentences are sent at once and two batches run together; the server decodes them in one GPU batch.

</details>

<details>
<summary>Common errors</summary>

| Symptom                                                                                      | Fix                                                                                                                          |
|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `error: required on PATH but not found: ffmpeg, ffprobe`                                     | Install ffmpeg; both `ffmpeg` and `ffprobe` must be on `PATH`                                                                |
| `error: environment variable OPENAI_API_KEY is not set (needed for engine 'openai')`         | `export` the variable named by that engine's `api_key_env`                                                                   |
| `error: engine 'breeze' selected but no [engine.breeze] table was found - ...`               | Add the `[engine.breeze]` table to the config file, or pass `--config PATH` to a file that has it                            |
| Resume re-synthesizes every sentence                                                         | Engine settings changed (new fingerprint). A text pipeline change re-synthesizes only sentences whose text changed |
| `error: server on port 7861 did not become healthy within 180s; see log at ...`               | Read `<cache_dir>/breeze-server-<port>.log`; check `command` (including its model argument) and whether another process holds `port`         |
| `Breeze server at ...: GET /v1/model answered 404 ...` / `missing or invalid /v1/model field(s)` | Server is not `breeze-tts-server`; stop it and start it with `breeze-tts-server` from the current breeze-tts |
| `Breeze server busy, waiting for the running inference to finish` (stderr, once per batch)    | Another client holds the server's single inference slot; the run waits (up to 300 s, retrying every 5 s) and continues on its own |
| `Drama server at ... serves [...], not 'fishaudio/drama-3'`                                   | Another model holds `port`; stop it, or set `[engine.drama].model` to the model it serves |
| Drama server log shows `did not become ready within 300s (stage_init_timeout)` or `Orchestrator startup timed out after 600s` | Weight loading is slow on a cold page cache; add `--init-timeout 1800 --stage-init-timeout 1800` to `command` |

</details>

## Contributing

`pixi run check` runs ruff, mypy --strict and pytest. Workflow: [CONTRIBUTING.md](https://github.com/aaron-ang/epub-to-m4b/blob/main/CONTRIBUTING.md). Module layout and conventions: [AGENTS.md](https://github.com/aaron-ang/epub-to-m4b/blob/main/AGENTS.md).

## License

[MIT](https://github.com/aaron-ang/epub-to-m4b/blob/main/LICENSE).
