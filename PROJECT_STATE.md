# Project state (2026-10-09)

The one place that says where the project stands. `SESSION_REPORT.md` is the log of what
happened in each session; `docs/results.md` holds the full tables.

**Validation only. The test set (hurricane-michael, santa-rosa-wildfire) has never been
loaded, evaluated or looked at.** The crop cache holds no test crops, and every loader
goes through `src/split.py`, which refuses the test part without `final=True`.

## Where things stand

- Data exploration, the split and the crop cache are done: 134,660 train, 14,718
  val-tiles and 19,099 val buildings, un-classified buildings dropped.
- **Base model:** stack ResNet-18, strong augmentation (`configs/stack_aug.json`), three
  seeds. **Best overall: the ensemble of those three seeds.**
- The model does not transfer to the unseen disasters: on val every model is less
  accurate than always answering "no-damage" (0.878).
- Per-building logits of the three base seeds are exported to `outputs/predictions/`
  for train, val-tiles and val.
- Uncertainty signal chosen: top probability of the 3-seed ensemble, temperature-scaled
  (`docs/uncertainty_plan.md`).
- Calibration of the ensemble is done (`docs/calibration_results.md`,
  `scripts/calibrate_ensemble.py`): T = 1.215 fitted on val-tiles; ECE 0.025 to 0.010 on
  val-tiles and 0.033 to 0.030 on val. A temperature fitted on val itself is nearly the
  same (1.232). Calibration is not the val problem; accuracy per class is.
- Tests: 136 pass, 1 skipped (the optional real-data test, which reads test tiles' label
  files and is deliberately not run).

## Models and checkpoints

Selection rule: the mean of the val-tiles and val macro-F1, averaged over 3 epochs; the
weights kept are those of the last epoch of the best 3-epoch stretch (`smoothed_score`
in `src/train.py`). val therefore influences which weights are kept, so val numbers are
somewhat optimistic.

### Trained and kept under this rule (`runs/`)

| run | what | epoch kept | smoothed score | val-tiles acc / macro-F1 | val acc / macro-F1 |
|---|---|---|---|---|---|
| `stack_aug_s0` | base, seed 0 | 10 of 10 | 0.496 | 0.871 / 0.693 | 0.604 / 0.304 |
| `stack_aug_s1` | base, seed 1 | 10 of 10 | 0.499 | 0.872 / 0.697 | 0.641 / 0.310 |
| `stack_aug_s2` | base, seed 2 | 9 of 10 | 0.498 | 0.869 / 0.693 | 0.607 / 0.299 |
| ensemble of the 3 | probabilities averaged | | | 0.876 / 0.705 | 0.640 / 0.313 |
| `stack_aug_pernorm_s0` | base + per-image normalisation | 10 of 10 | 0.461 | 0.866 / 0.679 | 0.483 / 0.247 |
| `stack_aug_pernorm_s1` | same, seed 1 | 9 of 10 | 0.458 | 0.866 / 0.681 | 0.477 / 0.237 |
| `stack_aug_pernorm_s2` | same, seed 2 | **unfinished** | | | |

`stack_aug_pernorm_s2` was stopped during epoch 10. It has a `best.pt` (epoch 9) and a
log but no `metrics.json` and no predictions; nothing uses it. Retrain from the start
rather than resume.

### Trained under the earlier rule only (`runs/old_rule/`)

No siamese model and no plain stack model (without strong augmentation) has been trained
under the current rule. What exists is one seed of each from before:

| run | weights saved | what the current rule would have kept, from the per-epoch log |
|---|---|---|
| `siamese_s0` | epoch 8 (chosen on val-tiles alone) | epoch 10: val-tiles 0.880 / 0.704, val 0.483 / 0.256, smoothed score 0.477 |
| `stack_s0` | epoch 4 | epoch 6 of the 7 that ran: val-tiles 0.867 / 0.689, val 0.541 / 0.255, smoothed score 0.476 |
| `stack_aug_s0..2` | epochs 9, 9, 4 | superseded by the runs above |

The right-hand column is read from the logs; **those weights do not exist on disk.**
`stack_s0` was also stopped early at epoch 7 by the old rule, so its row is not a full
run. These two rows are not a fair comparison with the base model.

`runs/sanity_overfit/` is the 256-crop overfitting check, not a model.

### Why the base model is the best

- It is the only setting with three finished seeds under the current rule, and the seeds
  agree (smoothed score 0.496-0.499).
- Per-image normalisation is clearly worse on val (accuracy 0.480 against 0.617, the
  seeds' ranges do not overlap) and slightly worse on val-tiles.
- The ensemble is preferred over any single seed because single seeds range from 0.604
  to 0.641 on val and nothing but val labels says which one is good; the ensemble
  reaches the best seed's level and is the best on val-tiles.
- **What is not established:** that stack beats siamese, or that strong augmentation
  beats base augmentation. Each was compared on one seed under the old rule, and the
  gaps were smaller than the seed-to-seed spread. Histogram matching
  (`configs/stack_aug_histmatch.json`) has never been run.

## Saved predictions

| where | what | parts |
|---|---|---|
| `outputs/predictions/<run>/<part>.csv` | uid, disaster, tile_id, label, logit_0..3, one row per building; written by `scripts/export_predictions.py` for `stack_aug_s0..2` | train, val-tiles, val |
| `runs/<run>/preds_<part>.npz` | logits and labels only, in cache order (no ids); written by training | val-tiles, val |
| `runs/old_rule/stack_aug_s0/mc_dropout_<part>.npz` | MC-dropout summary of the earlier seed-0 model | val-tiles, val |

Neither `runs/` nor `outputs/predictions/` is in git. Row counts of the export: 134,660 /
14,718 / 19,099, equal to the crop cache's label table; the exported val-tiles and val
logits give the same predicted class as the ones training saved for every building.
The train rows are predictions on data the model learned from (accuracy 0.914-0.915):
for diagnostics only, never for scores or for fitting a temperature.

## Stale or broken

Nothing is broken: the whole test suite passes, and the test suite that
`SESSION_REPORT.md` says was not re-run after the last edits of session 3 has now been
run. Stale:

- **`README.md`.** Its results table and bullet points are the old-rule numbers (seeds
  at 0.61 / 0.66 / 0.76 on val, ensemble 0.709). It says "three uncertainty methods"
  and describes selection without the 3-epoch window. Its "Reproduce" block trains
  siamese and plain stack and runs MC dropout, none of which is behind the current
  results, and omits the per-image runs, `scripts/gap_diagnosis.py` and
  `scripts/export_predictions.py`. `docs/gap_diagnosis.md` and
  `docs/uncertainty_plan.md` are not linked.
- **Usage examples that point at a moved run.** The docstrings of `src/calibrate.py`,
  `scripts/error_analysis.py` and `scripts/demo_figures.py` show
  `--run runs\siamese_s0`; that run now lives in `runs\old_rule\siamese_s0`.
- **`SESSION_REPORT.md`** is correct as a log but its lower half describes the old-rule
  runs; read only the "Session 3" part for current numbers.
- **`docs/results.md` section 6** calls MC dropout "20 times the prediction cost". It is
  20 passes; measured wall-clock is about 6 times (65 s against 11 s on val), because
  most of a prediction run is loading crops.
- **`PROJECT_STATE.md` did not exist** before this file.

## Open decisions

1. Finish the cheap-fix batch? Histogram matching (3 seeds) and the third per-image
   seed, about 1.5 hours. Commands are in `SESSION_REPORT.md`.
2. The model mostly ignores the pre image on the training disasters
   (`docs/gap_diagnosis.md`). Push it towards comparing before more preprocessing?
3. Retrain siamese and plain stack with 3 seeds under the current rule, so that the
   architecture and augmentation choices rest on a fair comparison?
4. Update `README.md` to the current numbers.
5. When to open the test set: not yet.
6. Small ones still open: the batch cap at 512; the 0.5 factor on the stack model's
   copied first-layer weights; loader workers (6 was used in session 3, 17.5 GB RAM per
   run).

## Commands

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest -q
& $py scripts\export_predictions.py          # train, val-tiles, val; about 5.5 minutes
& $py scripts\make_results.py                # rebuilds docs/results.md from runs/
```
