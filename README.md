# epub-to-m4b

Turn an EPUB into an M4B audiobook with chapter markers, cover art, and a WebVTT transcript.

TTS engines are pluggable behind one interface: any OpenAI-compatible speech
API, ElevenLabs, Deepgram Aura, or a self-hosted sidecar such as Breeze-TTS.
Add another by implementing the same small interface — no changes to the rest
of the pipeline.

Self-hosted engines run as a separate local HTTP server so this project never
needs heavy ML dependencies like `torch` itself. Point `config.toml` at
whatever command starts that server on your machine — see `epub-to-m4b engines --help`
for the config shape of each engine. Nothing assumes a particular repo layout
or sibling directory.

## Install

Requires Python 3.12, [uv](https://docs.astral.sh/uv/), and `ffmpeg`/`ffprobe` on `PATH`.

```bash
uv sync
```

## Usage

```bash
# See how a book's chapters will be detected before rendering it
uv run epub-to-m4b chapters book.epub

# Dry-run the pipeline with silence instead of real speech (fast, no GPU/API needed)
uv run epub-to-m4b convert book.epub --engine silence -o /tmp/dry-run

# Render with a self-hosted engine (e.g. Breeze-TTS), pointed at by config.toml
uv run epub-to-m4b convert book.epub --engine breeze -o ~/Documents/audiobooks

# Render with an OpenAI-compatible speech API
uv run epub-to-m4b convert book.epub --engine openai -o ~/Documents/audiobooks
```

Interrupted runs resume automatically: already-synthesized sentences are cached
and skipped on the next run.

## Development

See [AGENTS.md](AGENTS.md) for the architecture, module layout, and conventions.

```bash
make check   # ruff check, ruff format --check, mypy --strict, pytest
```
