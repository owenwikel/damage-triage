# damage-triage

Uncertainty-aware building damage assessment on the xBD dataset.

## Status

Data exploration complete (see [docs/observations.md](docs/observations.md)). Model in development.

## Layout

- `src/` — library code
- `notebooks/` — exploratory notebooks
- `scripts/` — standalone scripts (training, preprocessing, etc.)
- `outputs/` — generated artifacts (not tracked in git)
- `tests/` — test suite
- `docs/` — design notes: [observations](docs/observations.md), [split proposal](docs/split_proposal.md)
- Data: the xBD Challenge set from [xview2.org](https://xview2.org), stored outside the repo
  (scripts take `--data-root`)

## Setup

See `SETUP_NOTES.md` for environment setup details.
