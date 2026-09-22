# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CLI prints engine guard notes on stderr.
- MIT `LICENSE` file.
- CI workflow, Dependabot config, `.editorconfig`, `CONTRIBUTING.md`, this changelog.
- `make coverage` and `make ci` targets; coverage floor of 90%.
- Package metadata: SPDX license, keywords, classifiers, project URLs.

### Changed

- Runaway clips are reseeded as one batch per attempt instead of one request per clip.
- README and AGENTS.md condensed into tables and lists.

### Fixed

- Engine config rejects TOML values whose type does not match the target field.
- TOML integers are accepted for float config fields.

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

[Unreleased]: https://github.com/aaron-ang/epub-to-m4b/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/aaron-ang/epub-to-m4b/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/aaron-ang/epub-to-m4b/releases/tag/v0.1.0
