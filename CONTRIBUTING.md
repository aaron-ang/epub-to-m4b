# Contributing

## Setup

| Step | Command |
|------|---------|
| Install Python 3.14 and dependencies | `uv sync` |
| Install ffmpeg | `sudo apt-get install -y ffmpeg` (Debian/Ubuntu) or `brew install ffmpeg` (macOS) |

## Run checks

| Target | What it runs |
|--------|--------------|
| `make check` | `ruff check`, `ruff format --check`, `mypy --strict`, `pytest` |
| `make coverage` | `pytest --cov --cov-report=term-missing`; fails under 90% |
| `make format` | `ruff format` + `ruff check --fix` |
| `make ci` | Alias of `make check`; the CI workflow runs this |

GPU and paid-API tests are excluded by default (`-m 'not gpu and not network'`).

## Conventions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/): `feat(scope): ...`, `fix(scope): ...`, `docs: ...`, `build: ...`, `ci: ...`.
- Tests live in `tests/` and mirror the `src/epub_to_m4b/` layout.
- `mypy --strict` must pass; no `# type: ignore` without a reason.
- `ruff` lint and format must pass with the config in `pyproject.toml`.
- Comments explain mechanism, not history. Measured numbers belong in commit messages or test assertions, not in comments.
- Config uses `.editorconfig`: UTF-8, LF, final newline, 4-space Python, 2-space YAML/TOML/Markdown.

## Pull requests

- CI (`.github/workflows/ci.yml`) must be green.
- Behaviour changes ship with tests.
- User-facing changes get a line under `## [Unreleased]` in `CHANGELOG.md`.
- One logical change per PR; squash noise commits before opening.

## Architecture

See `AGENTS.md` for module layout, engine registry, cache layout, and the synthesis pipeline.
