# damage-triage

Uncertainty-aware building damage assessment on the xBD dataset: classify each building
as no-damage / minor / major / destroyed from a before and an after satellite image, and
say how sure the model is, so that confident predictions can be accepted and doubtful
ones sent to a person.

## Status

Data exploration, split and data decisions are done. A first model is trained and
evaluated **on validation data only**. The test set (hurricane-michael,
santa-rosa-wildfire) has not been loaded; code refuses to load it without `--final`
(`src/split.py`).

Still to run (see [BLOCKERS.md](BLOCKERS.md)): the second architecture ("stack"), the
3-model ensemble and MC dropout.

## Preliminary results (validation only)

One model so far: a siamese ResNet-18 (`siamese_s0`, 10 epochs, about 3 minutes per epoch
on an 8 GB laptop GPU). Full tables and figures: [docs/results.md](docs/results.md),
[docs/error_analysis.md](docs/error_analysis.md).

| | buildings | accuracy | macro-F1 | majority-class baseline (accuracy / macro-F1) |
|---|---|---|---|---|
| **val-tiles**: held-out tiles of the 6 training disasters | 14,718 | 0.879 | 0.708 | 0.809 / 0.224 |
| **val**: 2 disasters never trained on (both floods) | 19,099 | 0.451 | 0.243 | 0.878 / 0.234 |

What this says, plainly:

- **The model does not yet transfer to a new disaster.** On the two unseen flood disasters
  it is less accurate than always answering "no-damage", and its macro-F1 is at the level
  of guessing. On midwest-flooding it calls about 70% of undamaged buildings damaged.
- On tiles from disasters it trained on, it is well above the baselines, but unevenly:
  near-perfect where almost everything is undamaged (mexico-earthquake), weak on
  hurricane-matthew and hurricane-harvey.
- **Confidence is useful where the model is competent, and not where it is not.** On
  val-tiles, keeping the most confident 80% of predictions raises accuracy from 0.879 to
  0.965; on val the same rule only moves it from 0.451 to 0.496.
- Temperature scaling (T = 1.60) fixes calibration on val-tiles (ECE 0.045 to 0.011) but
  the model stays over-confident on val (ECE 0.272 to 0.170).

A held-out tile of a training disaster (palu-tsunami) and a tile of an unseen disaster
(hurricane-florence). REVIEW marks buildings below the confidence threshold. More in
[docs/demo.md](docs/demo.md).

![palu-tsunami demo tile](outputs/demo/palu-tsunami_00000166.jpg)

![hurricane-florence demo tile](outputs/demo/hurricane-florence_00000459.jpg)

## Layout

- `src/`: library code. `labels.py` (label loader), `split.py` (split + test-set guard),
  `crops.py`, `dataset.py`, `model.py`, `train.py`, `metrics.py`, `calibrate.py`,
  `uncertainty.py`, `triage.py`
- `scripts/`: `build_crops.py` (crop cache), `make_results.py`, `error_analysis.py`,
  `demo_figures.py`, plus the data-exploration scripts
- `configs/`: training configs
- `splits/split_v1.csv`: the split (source of truth)
- `docs/`: start with [WALKTHROUGH.md](docs/WALKTHROUGH.md) (study guide to the modelling
  code). Data: [observations](docs/observations.md), [data quality](docs/data_quality.md),
  [data decisions](docs/data_decisions.md), [split](docs/split_final.md) and
  [why](docs/split_decision.md). Model: [results](docs/results.md),
  [error analysis](docs/error_analysis.md), [demo](docs/demo.md)
- `runs/`: checkpoints, logs and predictions (not in git)
- `outputs/`: generated artifacts (not in git, except `outputs/demo/`)
- `tests/`: test suite (synthetic data; no dataset or GPU needed)
- Data: the xBD Challenge set from [xview2.org](https://xview2.org), stored outside the
  repo, as is the crop cache built from it

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\build_crops.py --data-dir C:\Users\wikel\dev\xbd-data
.\.venv\Scripts\python.exe -m src.train --config configs\siamese.json
.\.venv\Scripts\python.exe scripts\make_results.py
.\.venv\Scripts\python.exe scripts\error_analysis.py --run runs\siamese_s0
.\.venv\Scripts\python.exe scripts\demo_figures.py --run runs\siamese_s0 --data-dir C:\Users\wikel\dev\xbd-data
```

## Setup

See `SETUP_NOTES.md` for environment setup details.
