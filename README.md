# epub-to-m4b

Turn an EPUB into an M4B audiobook with chapter markers, cover art, and a WebVTT transcript.

TTS engines are pluggable behind one interface: any OpenAI-compatible speech
API, ElevenLabs, Deepgram Aura, or a self-hosted sidecar such as Breeze-TTS.
Self-hosted engines run as a separate local HTTP server, so this project
never needs heavy ML dependencies like `torch` itself.

## Install

Requires Python 3.14, [uv](https://docs.astral.sh/uv/), and `ffmpeg`/`ffprobe` on `PATH`.

```bash
uv sync
```

## Usage

Three subcommands: `chapters`, `dump-text`, `convert`.

```bash
# See how a book's chapters will be detected before rendering it
uv run epub-to-m4b chapters book.epub

# Print the text as it will be read (--normalized / --split show the TTS input)
uv run epub-to-m4b dump-text book.epub --chapter 3 --split

# Dry-run the pipeline with silence instead of real speech (fast, no GPU/API needed)
uv run epub-to-m4b convert book.epub --engine silence -o /tmp/dry-run

# Render with a self-hosted engine (e.g. Breeze-TTS) configured in config.toml
uv run epub-to-m4b convert book.epub --engine breeze -o ~/Documents/audiobooks

# Render with an OpenAI-compatible speech API
OPENAI_API_KEY=... uv run epub-to-m4b convert book.epub --engine openai -o ~/Documents/audiobooks
```

`--engine` accepts `breeze`, `deepgram`, `elevenlabs`, `openai`, `silence`, `tone`.
All subcommands take `--toc-depth N` (default 1) and `--min-chars N` (default 200)
to tune chapter detection; check the result with `chapters` first.

Output lands in the `-o` directory as `<slug>.m4b` (chapter markers, cover,
title/author tags) and `<slug>.vtt`, where `<slug>` is derived from the book title.

## Configuration

`convert` reads an optional TOML file. Resolution order, first match wins:

1. `--config PATH`
2. `E2M_CONFIG` environment variable
3. `~/.config/epub-to-m4b/config.toml` (may be absent)

A path given via `--config` or `E2M_CONFIG` must exist; the default path is
optional. Only `[engine.<name>]` tables are read. Unknown keys and missing
required keys are reported as errors. API keys are never put in the TOML;
each API engine names an environment variable via `api_key_env` and reads the
key from there at run time.

```toml
# Self-hosted Breeze-TTS sidecar. `command` is the argv that starts the
# server; the tool appends the weights dir and --host/--port itself. If a
# server is already listening on `port`, it is adopted instead of spawned.
[engine.breeze]
weights_dir = "/path/to/breeze-tts-2"                                      # required
command = ["uv", "run", "--directory", "/path/to/breeze-tts", "breeze-infer-api"]  # required
port = 7861
batch_size = 64
instruction = "A clear, neutral adult narrator voice with a calm, steady reading pace."
cfg_scale = 4.0
seed = 42

# Any OpenAI-compatible /v1/audio/speech endpoint.
[engine.openai]
base_url = "https://api.openai.com/v1"   # required
model = "gpt-4o-mini-tts"                # required
voice = "alloy"                          # required
speed = 1.0
api_key_env = "OPENAI_API_KEY"

[engine.elevenlabs]
voice_id = "..."                         # required
model_id = "eleven_multilingual_v2"
base_url = "https://api.elevenlabs.io"
api_key_env = "ELEVENLABS_API_KEY"

# Every key has a default; the table itself is optional.
[engine.deepgram]
model = "aura-2-thalia-en"
base_url = "https://api.deepgram.com"
api_key_env = "DEEPGRAM_API_KEY"
```

`silence` and `tone` take no configuration; they exist for dry runs and tests.

## Resume

Rerun the same command to resume an interrupted or partial render.

- Synthesized sentences are cached as FLAC under `~/.cache/epub-to-m4b/clips/<engine-fingerprint>/`
  (override the root with `E2M_CACHE_DIR`). The cache is shared across books:
  a sentence already rendered by the same engine settings is never sent to
  the engine again.
- Assembled chapters live under `<out_dir>/.work/<book-id>/chapters/` and are
  reused when nothing in the chapter changed.
- A missing, damaged, or partially written `.m4b` is rebuilt; an up-to-date
  one is left alone and only the `.vtt` is rewritten.
- Changing engine, voice, model, or other audio-affecting settings changes the
  fingerprint, so old clips are not reused (and stay on disk if you switch back).

## Development

See [AGENTS.md](AGENTS.md) for the architecture, module layout, and conventions.

```bash
make check   # ruff check, ruff format --check, mypy --strict, pytest
```
