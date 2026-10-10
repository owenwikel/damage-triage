# Project state (2026-10-10)

The one place that says where the project stands. `SESSION_REPORT.md` is the log of what
happened in each session; `docs/results.md` holds the full tables.

**The test set has been evaluated, once, on 2026-10-10.** Protocol
`docs/test_protocol.md` (committed first, `14cfb0e`), results `docs/test_results.md`,
lock `TEST_EVALUATION.lock`. The project's modelling work is closed: nothing may be
changed and re-scored on test. Everything else in this file is validation.

The development crop cache still holds no test crops (they are in a separate folder,
`C:\Users\wikel\dev\xbd-data\crops_test_final`), test predictions are in
`outputs/predictions_test/` apart from the development predictions, and every
development loader still refuses the test part (`src/split.py`).

## Final held-out result (test)

Ensemble of `stack_aug_s0..2`, mean softmax, no temperature; queue "doubtful no-damage
first"; 20% review; 95% intervals from 1000 tile-bootstrap resamples.

| test | hurricane-michael | santa-rosa-wildfire | both |
|---|---|---|---|
| buildings / tiles | 28,056 / 426 | 17,111 / 291 | 45,167 / 717 |
| missed-damage rate, nobody reviewed | 0.614 | 0.074 | 0.441 |
| missed-damage rate, 20% random review | 0.491 | 0.059 | 0.353 |
| missed-damage rate, 20%, queue in random order | 0.449 | 0.050 | 0.315 |
| **missed-damage rate, 20%, doubtful no-damage first** | **0.390 (0.368-0.411)** | **0.026 (0.018-0.038)** | **0.272 (0.241-0.303)** |
| accuracy / macro-F1 on accepted, 20% | 0.599 / 0.392 | 0.828 / 0.456 | 0.686 / 0.512 |
| share of predicted damage that is real | 0.526 | 0.670 | 0.593 |
| accuracy, nobody reviewed (always "no-damage") | 0.601 (0.650) | 0.849 (0.731) | 0.695 (0.681) |
| recall no-damage / minor / major / destroyed | 0.81 / 0.21 / 0.14 / 0.32 | 0.83 / 0.03 / 0.06 / 0.93 | 0.82 / 0.21 / 0.14 / 0.82 |

- The queue beat both baselines in both disasters, outside the intervals. It still
  leaves 27% of damaged buildings accepted as undamaged (39% in michael).
- santa-rosa-wildfire behaves like the training disasters; hurricane-michael is worse
  than a constant answer and misses 61% of damage before review.
- With test's class mix, val's class-by-class miss rates give 0.444 (no review) and
  0.265 (20%), against 0.441 and 0.272 on test: about what val predicted, though
  major-damage is missed more and destroyed less than on val (exploratory section of
  the results document).
- The run: test crops built (45,167 buildings, 717 tiles), export ran once
  (2026-10-10 08:53-08:54), no crash, no rerun. A second start of the export is refused
  by the lock (checked).

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
- Risk-coverage analysis of the ensemble is done (`docs/risk_coverage.md`,
  `scripts/risk_coverage.py`; tile-level bootstrap intervals). Reviewing the 20% least
  confident buildings cuts the miss rate from 0.163 to 0.059 on val-tiles, but only
  from 0.374 to 0.283 on val, where random review would give 0.299. Ranking by
  P(damaged) does not reduce misses until everything predicted damaged is reviewed.
  A third queue, only buildings predicted no-damage and the most doubtful first, does
  far better on misses: 0.022 on val-tiles and 0.185 on val at 20% review. Its cost:
  buildings predicted damaged are never reviewed, and on val only 21% of them are
  truly damaged.
- **Pre-image experiments (overnight session, 2026-10-10): both DROPPED.** Two changes
  meant to make the model compare before and after were trained with 3 seeds each and
  judged by a stop rule committed before training (`docs/pre_image_plan.md`,
  `scripts/stop_rule.py`). Neither met it; both are worse than the base on val. The
  base model stays the model of record. Details in the next section.
- `README.md` is rewritten to the current state; val-only demo figures are in
  `docs/figures/demo_val_*.jpg` (`docs/demo_val.md`).
- **Error analysis extended** (`docs/error_analysis.md`, base model seed 0, val-tiles and
  val): confusion per disaster and class, which damaged buildings are missed, six val
  tiles of missed damage (`docs/figures/missed_damage_val_*.jpg`), and a label-noise
  check. On val most errors are false alarms (36% of undamaged buildings are called
  damaged); misses concentrate in hurricane-florence major-damage (482 buildings) and
  minor-damage has the highest miss rate (59%). Label-noise candidates are 5-9 times
  over-represented among the most confident val errors but are still under 10% of
  them (43 of the top 500). No label was changed.
- Tests: 170 pass, 1 skipped (the optional real-data test, which reads test tiles' label
  files and is deliberately not run).

## The overnight session of 2026-10-10

What ran, in order (all validation only; training from 00:15 to 02:09):

1. **Plan and stop rule**, committed before any training (`7b96b05`). KEEP only if the
   3-seed mean val macro-F1 rises by at least 0.03 (to 0.334), mean val accuracy does
   not fall (0.617), and mean val-tiles macro-F1 falls by at most 0.02 (to 0.674).
2. **Pre-image-use test** (`scripts/pre_image_use.py`): each val building gets the pre
   image of another building of the same disaster; the drop in macro-F1 is reported.
   Reference, base model: val 0.304 to 0.254 (drop 0.050), val-tiles drop 0.073.
3. **Primary, same-image training pairs** (`stack_aug_samepair`, option `same_pair_prob`
   = 0.25): 3 seeds, about 14 minutes each.
4. **Backup, siamese with a difference-only head** (`siamese_diff_aug`,
   `src/model_diff.py`): 3 seeds, about 22 minutes each.
5. Nothing was kept, so no new exports and no new risk-coverage run;
   `docs/results.md` was rebuilt with all five settings.

| setting (mean of 3 seeds) | val macro-F1 | val accuracy | val-tiles macro-F1 | pre-image-use drop on val | decision |
|---|---|---|---|---|---|
| base `stack_aug` | 0.304 | 0.617 | 0.694 | 0.050 | reference |
| `stack_aug_samepair` | 0.274 | 0.554 | 0.688 | 0.060 | **DROP**: fails the val macro-F1 and val accuracy conditions |
| `siamese_diff_aug` | 0.268 | 0.537 | 0.695 | 0.009 | **DROP**: fails the same two |

Why DROP, in words: the rule asked for a clear gain on the unseen disasters, and both
changes lost about 0.03 macro-F1 and 0.06-0.08 accuracy there while staying level on
val-tiles. The same-image model does lean on the pre image a little more than the base
and transfers worse. The difference-only model cannot ignore the pre image, yet on val
it scores almost the same with another building's pre image (0.259 against 0.268): on
the unseen disasters the building's own "before" gives it next to nothing.

One finding changes an earlier statement in this file: the base model does **not**
simply ignore the pre image. Swapping in another building's pre image changes 38% of
its val answers and costs 0.073 macro-F1 on val-tiles. What `docs/gap_diagnosis.md`
showed is narrower: replacing the pre image by the post image barely moves val-tiles
accuracy.

Cost: stack runs 4.1 GB GPU peak and 17.0-17.1 GB RAM per run; siamese runs 6.8 GB GPU
and 20.5 GB RAM per run (above the 17.5 GB of a base run; the machine kept 20 GB free).
No run failed; nothing went into `BLOCKERS.md`.

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
| `stack_aug_pernorm_s2` | same, seed 2 | 9 of 10 | 0.471 | 0.866 / 0.683 | 0.531 / 0.260 |
| ensemble of the 3 | per-image normalisation | | | 0.871 / 0.690 | 0.512 / 0.246 |
| `stack_aug_histmatch_s0` | base + histogram matching | 10 of 10 | 0.465 | 0.863 / 0.673 | 0.549 / 0.261 |
| `stack_aug_histmatch_s1` | same, seed 1 | 10 of 10 | 0.471 | 0.860 / 0.672 | 0.578 / 0.279 |
| `stack_aug_histmatch_s2` | same, seed 2 | 9 of 10 | 0.468 | 0.863 / 0.672 | 0.586 / 0.273 |
| ensemble of the 3 | histogram matching | | | 0.867 / 0.683 | 0.590 / 0.276 |
| `stack_aug_samepair_s0` | base + same-image training pairs | 10 of 10 | 0.487 | 0.870 / 0.690 | 0.561 / 0.286 |
| `stack_aug_samepair_s1` | same, seed 1 | 10 of 10 | 0.478 | 0.868 / 0.687 | 0.568 / 0.276 |
| `stack_aug_samepair_s2` | same, seed 2 | 10 of 10 | 0.474 | 0.867 / 0.689 | 0.532 / 0.260 |
| ensemble of the 3 | same-image pairs | | | 0.872 / 0.694 | 0.573 / 0.281 |
| `siamese_diff_aug_s0` | siamese, difference-only head | 10 of 10 | 0.477 | 0.875 / 0.697 | 0.522 / 0.259 |
| `siamese_diff_aug_s1` | same, seed 1 | 10 of 10 | 0.475 | 0.873 / 0.696 | 0.478 / 0.254 |
| `siamese_diff_aug_s2` | same, seed 2 | 10 of 10 | 0.490 | 0.872 / 0.692 | 0.611 / 0.291 |
| ensemble of the 3 | difference-only siamese | | | 0.878 / 0.703 | 0.566 / 0.283 |

The cheap-fix batch is complete (2026-10-09): `stack_aug_pernorm_s2` was retrained from
the start and the three histogram-matching seeds were trained, all under the rule above.
**Neither fix helps; both are worse than the base on val and on val-tiles**
(`docs/results.md` section 2).

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
- Per-image normalisation is clearly worse on val (mean accuracy 0.497 against 0.617,
  macro-F1 0.248 against 0.304; the seeds' ranges do not overlap) and slightly worse on
  val-tiles.
- Histogram matching is also worse on val (mean accuracy 0.571, macro-F1 0.271; ranges
  do not overlap with the base) and on val-tiles (macro-F1 0.672 against 0.694).
- The ensemble is preferred over any single seed because single seeds range from 0.604
  to 0.641 on val and nothing but val labels says which one is good; the ensemble
  reaches the best seed's level and is the best on val-tiles.
- The two pre-image changes (same-image pairs, difference-only siamese) are worse on
  val and were dropped by the pre-committed stop rule.
- **What is not established:** that strong augmentation beats base augmentation, or
  that stack beats the *ordinary* siamese model (`[pre, post, post - pre]` head). Each
  was compared on one seed under the old rule. The difference-only siamese is a
  different model and does not settle it, though it is level with the base on
  val-tiles and worse on val.

## Saved predictions

| where | what | parts |
|---|---|---|
| `outputs/predictions/<run>/<part>.csv` | uid, disaster, tile_id, label, logit_0..3, one row per building; written by `scripts/export_predictions.py` for `stack_aug_s0..2`, `stack_aug_histmatch_s0..2` and `stack_aug_pernorm_s2` (not for `stack_aug_pernorm_s0..1`, and not for the two dropped pre-image settings) | train, val-tiles, val |
| `runs/<run>/preds_<part>.npz` | logits and labels only, in cache order (no ids); written by training | val-tiles, val |
| `runs/old_rule/stack_aug_s0/mc_dropout_<part>.npz` | MC-dropout summary of the earlier seed-0 model | val-tiles, val |
| `outputs/pre_image_use/<setting>.json` | pre-image-use test, per seed | val, val-tiles |

Neither `runs/` nor `outputs/predictions/` is in git. Row counts of the export: 134,660 /
14,718 / 19,099, equal to the crop cache's label table; the exported val-tiles and val
logits give the same predicted class as the ones training saved for every building.
The train rows are predictions on data the model learned from (accuracy 0.914-0.915):
for diagnostics only, never for scores or for fitting a temperature.

## Stale or broken

Nothing is broken: the whole test suite passes. Stale or to be aware of:

- **`SESSION_REPORT.md`** is the log of 2026-10-01 only. It was not extended for the later
  sessions; this file and the docs are current.
- **`docs/gap_diagnosis.md`** says the model "is mostly reading the post image and hardly
  comparing". The pre-image-use test qualifies that (see above). The document itself
  was not changed.
- **`docs/demo.md` and `outputs/demo/`** show the six-tile demo of 2026-10-01 (val and
  val-tiles tiles of seed 0). Still valid for that model; the README now uses the
  val-only figures in `docs/figures/`.
- **`outputs/label_noise/candidates.csv` and `docs/label_noise.md` include the two test
  disasters.** They come from the first data exploration (2026-10-01), before the split
  existed, and hold label statistics only (no model output). Code that reads the file
  now keeps val rows only (`scripts/error_analysis.py`). Worth deciding whether to
  regenerate both without the test disasters.
- **`docs/results.md` section 6** calls MC dropout "20 times the prediction cost". It is
  20 passes; measured wall-clock is about 6 times (65 s against 11 s on val), because
  most of a prediction run is loading crops.

## Open decisions

1. **What next for the val gap?** (Closed for this project by the test evaluation; kept
   as the record of where things stood.) Four changes are now rejected with 3 seeds each (two
   colour fixes, two pre-image fixes). Options not yet tried: more context around each
   building; training on more or different disasters (for example moving a flood
   disaster from val into training, which costs a val disaster); or accepting the gap
   and presenting the project as "works within disaster types, with a measured failure
   on unseen ones". My recommendation is the last, together with item 2.
2. **False alarms on the damage map.** The review queue that works leaves predicted
   damage unchecked, and on val only 21% of it is real. Is that acceptable for the use
   you have in mind, or should part of the review budget go to the map?
3. ~~When to open the test set.~~ Done on 2026-10-10. **From here on, any change to the
   model or the queue cannot be scored on these two disasters** without spoiling them;
   new modelling work would need new held-out disasters.
4. Remove the test disasters from `outputs/label_noise/candidates.csv` and
   `docs/label_noise.md`?
5. Retrain the ordinary siamese and plain stack with 3 seeds under the current rule, so
   that the architecture and augmentation choices rest on a fair comparison? Low value
   now: nothing suggests either would change the val picture.
6. Small ones still open: the batch cap at 512; the 0.5 factor on the stack model's
   copied first-layer weights; loader workers (6 was used in session 3, 17.5 GB RAM per
   run).

## Commands

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest -q
& $py scripts\export_predictions.py          # train, val-tiles, val; about 5.5 minutes
& $py scripts\make_results.py                # rebuilds docs/results.md from runs/
& $py scripts\risk_coverage.py               # the headline table and plots
& $py scripts\stop_rule.py --candidate stack_aug_samepair    # prints the DROP decision again
& $py scripts\stop_rule.py --candidate siamese_diff_aug
& $py scripts\pre_image_use.py --settings stack_aug stack_aug_samepair siamese_diff_aug
```

The final evaluation (run once on 2026-10-10; the export cannot be run again while
`TEST_EVALUATION.lock` exists):

```powershell
# rebuilds the tables of docs/test_results.md from the saved test predictions; no new evaluation
& $py scripts\test_results.py --final
& $py scripts\test_class_mix.py --final      # the exploratory class-mix section
```

Next steps are writing, not modelling: the resume bullet and interview notes in
`docs/risk_coverage.md` still quote val numbers and could now quote the test numbers
(missed damage 44% to 27% at 20% review on two held-out disasters, random review 35%).
