# damage-triage

Uncertainty-aware building damage assessment on the xBD dataset: classify each building
as no-damage / minor / major / destroyed from a before and an after satellite image, and
say how sure the model is, so that confident predictions can be accepted and doubtful
ones sent to a person.

## Status

Data exploration, split and data decisions are done. Two architectures, an augmentation
experiment and three uncertainty methods have been trained and compared **on validation
data only**. The test set (hurricane-michael, santa-rosa-wildfire) has not been loaded;
code refuses to load it without `--final` (`src/split.py`).

## Preliminary results (validation only)

Full tables and figures: [docs/results.md](docs/results.md),
[docs/error_analysis.md](docs/error_analysis.md).

| | val-tiles accuracy | val-tiles macro-F1 | val accuracy | val macro-F1 |
|---|---|---|---|---|
| majority-class baseline | 0.809 | 0.224 | 0.878 | 0.234 |
| siamese ResNet-18, first run | 0.879 | 0.708 | 0.451 | 0.243 |
| stack ResNet-18 | 0.871 | 0.685 | 0.676 | 0.297 |
| stack + strong augmentation (base model) | 0.870 | 0.693 | 0.613 | 0.308 |
| ensemble of 3 base models | 0.875 | 0.700 | 0.709 | 0.331 |

- **val-tiles**: 14,718 buildings on held-out tiles of the 6 training disasters.
- **val**: 19,099 buildings from 2 flood disasters never trained on. **val is used for
  model selection** (the kept checkpoint maximises the mean of the val-tiles and val
  macro-F1, and the architecture and augmentation were chosen on it), so its numbers are
  somewhat optimistic. The first siamese row predates that rule.

What this says, plainly:

- **The model still does not transfer well to a new disaster.** On val every model is less
  accurate than always answering "no-damage" (0.878). Macro-F1 is above the baseline
  (0.33 against 0.23) but far below val-tiles (0.70).
- Stack edged out siamese, and stronger augmentation changed little for a single model
  (val macro-F1 +0.01, val accuracy -0.06). Single runs differ by more than that from
  seed to seed, so neither comparison is settled.
- **On val, the seed matters more than anything else that was varied.** Three runs of the
  same model differ only in their random seed and score 0.61, 0.66 and 0.76 accuracy on
  val (0.86-0.87 on val-tiles). The ensemble of the three lands at 0.71: better than the
  typical member, not better than the best one. What it buys is that the result no
  longer depends on which seed was trained. Keeping its most confident 50% of val
  predictions gives 0.90 accuracy.
- **MC dropout adds nothing here.** Dropout sits only in front of the last layer, and its
  confidence is practically the same as the plain top probability.
- On val-tiles, confidence works: keeping the most confident 80% raises accuracy from
  0.87 to 0.96. With an inspection budget of 20% of buildings, ranking by "most likely
  damaged" reaches 79% of the damaged buildings on val-tiles and 47% on val.

A held-out tile of a training disaster (palu-tsunami) and a tile of an unseen disaster
(hurricane-florence), from the single base model. REVIEW marks buildings below the
confidence threshold. More in [docs/demo.md](docs/demo.md).

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
$py = ".\.venv\Scripts\python.exe"
& $py scripts\build_crops.py --data-dir C:\Users\wikel\dev\xbd-data
& $py -m src.train --config configs\siamese.json
& $py -m src.train --config configs\stack.json
& $py -m src.train --config configs\stack_aug.json            # base model, seed 0
& $py -m src.train --config configs\stack_aug.json --seed 1
& $py -m src.train --config configs\stack_aug.json --seed 2
& $py -m src.uncertainty --run runs\stack_aug_s0 --passes 20  # MC dropout
& $py scripts\make_results.py
& $py scripts\error_analysis.py --run runs\stack_aug_s0
& $py scripts\demo_figures.py --run runs\stack_aug_s0 --data-dir C:\Users\wikel\dev\xbd-data
```

A training run takes 15-30 minutes on an 8 GB laptop GPU and about 10 GB of RAM. None of
these commands touches the test set.

## Setup

See `SETUP_NOTES.md` for environment setup details.
