# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-09-21

### Fixed

- Cached clips and the output `.m4b` use umask-derived permissions instead of `0600`.
- `TEXT_PIPELINE_VERSION` is derived from the docstring-stripped AST, so comment-only edits no longer invalidate the clip cache.

## [0.1.0] - 2026-09-21

### Added

- EPUB reader with chapter detection and XHTML paragraph parser.
- English text normalizer and sentence splitter.
- `chapters`, `dump-text`, and `convert` subcommands.
- M4B assembly with chapter markers and WebVTT transcript output.
- Sentence orchestrator with gap policy, length-sorted batching, and resume cache.
- Concurrent batch fan-out over a thread pool.
- TTS engines: Breeze (local HTTP sidecar), OpenAI-compatible, ElevenLabs, Deepgram Aura, and fake silence/tone engines.
- Retrying POST helper and PCM decoder shared by HTTP API engines.
- TOML engine config wired through `--engine`.

### Fixed

- Missing ffmpeg reports a clean error instead of a traceback.
- Progress log lines flush when output is redirected.
- Chapter and M4B reuse is engine-aware and crash-safe; memory is bounded to one chapter.
- Breeze sidecar startup no longer spawns a competing server on a mid-startup port.
- FLAC clips are re-encoded across the concat boundary instead of stream-copied.

[0.1.1]: https://github.com/aaron-ang/epub-to-m4b/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/aaron-ang/epub-to-m4b/releases/tag/v0.1.0
