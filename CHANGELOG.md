# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0](https://github.com/aaron-ang/epub-to-m4b/compare/v0.2.0...v0.3.0) (2026-09-24)


### ⚠ BREAKING CHANGES

* **text:** the environment is managed by pixi (uv.lock, Makefile and .python-version are removed), and normalized text changes for most sentences, so most clips are re-synthesized.
* **cache:** clip cache keys change, so existing clip caches are not reused.
* **breeze:** the `weights_dir` key in [engine.breeze] is removed. Put the model dir or repo id at the end of `command` and start the server with `breeze-tts-server`. The fingerprint changes, so existing Breeze clips are not reused.

### Features

* **audio:** normalize loudness to -16 LUFS and rebuild on encode changes ([9b2b4d4](https://github.com/aaron-ang/epub-to-m4b/commit/9b2b4d4a299627a6e23db84d3e8a68ef2c6c3f27))
* **breeze:** take the model from command and read model facts from the server ([dc2abb0](https://github.com/aaron-ang/epub-to-m4b/commit/dc2abb0168efefecdcae63b95f478f5a866a130f))
* **cache:** key clips by their exact text ([7a15553](https://github.com/aaron-ang/epub-to-m4b/commit/7a15553776d805e50d328a464e2c7bc71c92e4a9))
* **epub:** skip notes and source lists by their semantics ([89ffa8e](https://github.com/aaron-ang/epub-to-m4b/commit/89ffa8e66212b7e068757f85251a235a861db556))
* **text:** normalize with NeMo text processing and manage the env with pixi ([ca269b6](https://github.com/aaron-ang/epub-to-m4b/commit/ca269b68075a7a8850ee2e5ab53507b0504d3752))


### Bug Fixes

* **split:** keep name initials in one sentence ([a1f6c79](https://github.com/aaron-ang/epub-to-m4b/commit/a1f6c791c061e3feca65d9879ecc995e0dbfe10b))
* **synth:** pause for the heading gap after a chapter's last sentence ([b8a38ac](https://github.com/aaron-ang/epub-to-m4b/commit/b8a38aceefba17791dfe4ba19c7e3c88e97a9914))


### Documentation

* **agents:** top-level layout, generic sidecar notes, sidecar marker ([ac3c31d](https://github.com/aaron-ang/epub-to-m4b/commit/ac3c31d444d6f4a8bc221bd1539d5b89631129dd))
* **readme:** add narrated sample from Crime and Punishment ([f9addb7](https://github.com/aaron-ang/epub-to-m4b/commit/f9addb7f0133a839269c291fd1f74caa526dcedd))

## [0.2.0](https://github.com/aaron-ang/epub-to-m4b/compare/v0.1.1...v0.2.0) (2026-09-22)


### Features

* **cli:** surface engine guard notes on stderr ([9a79ff9](https://github.com/aaron-ang/epub-to-m4b/commit/9a79ff94f904ea85125ea590c78a1d9ec02cb2fd))


### Bug Fixes

* **cli:** report any EpubToM4bError from convert as a clean exit 1 ([2cab482](https://github.com/aaron-ang/epub-to-m4b/commit/2cab48271d5586ab322927f2ccce77b5e4161e03))
* **config:** convert TOML integers to float for float fields ([c7e5788](https://github.com/aaron-ang/epub-to-m4b/commit/c7e578895880a598b1800c2a51374b65d06abdfa))
* **config:** reject TOML values whose type does not match the engine config field ([a53087e](https://github.com/aaron-ang/epub-to-m4b/commit/a53087e483eacda5e48de30672a7ed7764c21ebd))
* **errors:** route sidecar timeout and malformed batch responses through EpubToM4bError ([78a4e33](https://github.com/aaron-ang/epub-to-m4b/commit/78a4e335e2675d973370716a94c160f8e1aee4c3))


### Performance Improvements

* **tts:** reseed runaway clips as one batch per attempt ([9aea1e9](https://github.com/aaron-ang/epub-to-m4b/commit/9aea1e93048fa5f5070fb6cc3f5217e6bd68e77d))


### Documentation

* add CONTRIBUTING.md and CHANGELOG.md ([9402975](https://github.com/aaron-ang/epub-to-m4b/commit/9402975cf55b72da245b10e77391d310d629827c))
* add quick start, example output, engine and troubleshooting tables, pipeline diagram ([c08c609](https://github.com/aaron-ang/epub-to-m4b/commit/c08c6091cd6bef76c0c52fb29e8b6f290253ef09))
* **cli:** add help text for every argument and subcommand ([fde7970](https://github.com/aaron-ang/epub-to-m4b/commit/fde797073cdbc69574b62628bf615e8060d586a8))
* **cli:** note that the log handler binds stderr at call time ([0cd1798](https://github.com/aaron-ang/epub-to-m4b/commit/0cd1798d3a5641454af18f968aeb759de7394e26))
* **readme:** lead with API engines, present Breeze as the self-hosted sidecar ([77dc527](https://github.com/aaron-ang/epub-to-m4b/commit/77dc527af72eafdfc215994082ca1b6fe27e75bb))
* tighten README and AGENTS.md into tables and lists ([a357bfd](https://github.com/aaron-ang/epub-to-m4b/commit/a357bfdbac032a057f8b0528c46d6aad4b31e84c))

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
