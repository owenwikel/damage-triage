# Test protocol: the one evaluation on the held-out disasters

Written and committed on 2026-10-10 **before the test set was touched**: before any test
crop was cut, before any model saw a test image, before any test number existed.
Nothing in this file is changed afterwards. Results go to `docs/test_results.md`.

Up to this commit, the test part (hurricane-michael, santa-rosa-wildfire) has not been
loaded by any model or evaluation code. One caveat, stated for honesty: the first data
exploration on 2026-10-01, before the split existed, counted labels over all disasters,
so label statistics of these two disasters (class shares, label-noise candidates) are in
`docs/observations.md`, `docs/disasters.md` and `docs/label_noise.md`. No model output on
them exists.

## 1. What is evaluated (fixed)

- **Model:** the three base runs `runs/stack_aug_s0`, `runs/stack_aug_s1`,
  `runs/stack_aug_s2` as they are on disk (stack ResNet-18, strong augmentation, each at
  the epoch kept by the 3-epoch selection rule). No retraining, no fine-tuning.
- **Ensemble:** for each building, the mean of the three models' softmax probabilities.
  **No temperature scaling.** Predicted class = the largest mean probability.
- **Review queue ("doubtful no-damage first"):** only buildings predicted no-damage can
  be reviewed, in order of highest 1 - P(no-damage). Buildings predicted damaged go to
  the damage map unreviewed. Exactly as in `scripts/risk_coverage.py` (ranking
  `no_damage_queue`).
- **Headline review rate: 20%** of all buildings of the group being scored.

## 2. Data (fixed)

- **Parts:** the test part of `splits/split_v1.csv`: all tiles of hurricane-michael and
  of santa-rosa-wildfire. Reported for each disaster **separately** and for both
  **combined** (all buildings pooled).
- Buildings are prepared exactly as for train and val: un-classified buildings dropped,
  invalid polygons repaired, 128 px crops with 16 px padding (`scripts/build_crops.py`
  with `--final --parts test`), ImageNet normalisation, no augmentation.
- The test crops are written to a **separate folder**
  (`C:\Users\wikel\dev\xbd-data\crops_test_final`), not into the development crop cache,
  so that no development script can come across them. Test predictions go to
  `outputs/predictions_test/`, likewise apart from the development predictions.

## 3. Metrics (fixed)

All at review rates 0%, 10%, **20% (headline)** and 30%; the full curve from 0% to 50%
is saved and plotted.

1. **Missed-damage rate:** of all truly damaged buildings (minor, major or destroyed) in
   the group, the share that is not reviewed and is predicted no-damage.
2. **Accuracy on the accepted buildings** (everything not reviewed, the damage map
   included).
3. **Macro-F1 on the accepted buildings.**
4. **Share of predicted damage that is real:** of the buildings predicted damaged, the
   share that is truly damaged.
5. **Per-class recall** of the ensemble's predictions before any review, with the
   number of buildings per class.

Also reported, because they are needed to read the above: the number of buildings and
tiles, the class mix, and accuracy and macro-F1 with nobody reviewed.

## 4. Baselines (fixed)

- **Random review:** the same share of buildings reviewed at random
  (missed-damage rate = rate with nobody reviewed x share not reviewed).
- **The same queue in random order:** only buildings predicted no-damage are reviewed,
  in random order.

For context, and reported in full whatever they show: the two other rankings analysed on
val ("least confident first", "most likely damaged first").

## 5. Uncertainty (fixed)

95% intervals from **1000 bootstrap resamples of whole tiles** (percentile method,
random seed 0), computed within the group being scored: michael's tiles for michael,
santa-rosa's for santa-rosa, all test tiles for the combined numbers. Differences
between a ranking and a baseline are computed inside each resample.

## 6. Code (fixed)

The risk-coverage numbers are computed by the existing functions of
`scripts/risk_coverage.py` (`analyse_part`, `Curves`, `tile_bootstrap_weights`,
`review_priority`), **unchanged**. That file refuses the test part by design, so two
small new scripts sit around it: one exports the test predictions (it needs `--final`,
checks that this protocol is committed, and writes a lock file), one feeds the exported
tables to the unchanged functions and writes the document. Both are committed before
the export is run, and the second is tried out on the val predictions first, where the
answers are already known.

## 7. Rules

1. **Run once.** The export writes `TEST_EVALUATION.lock` before it loads anything and
   refuses to run while that file exists.
2. **No changes after seeing results.** Not to the models, the ensemble, the queue, the
   review rate, the metrics or the baselines. Whatever the numbers are, they are
   reported as the final held-out result.
3. **No selective reporting.** Every metric of section 3, for both disasters and
   combined, goes into `docs/test_results.md`, good or bad.
4. **Technical failure.** If the export crashes before any metric has been computed or
   looked at (a missing file, a memory error), the cause is fixed, the lock is removed
   by hand, the run is repeated, and all of that is written into the results document.
   Once any test metric has been seen, there is no second run.
5. Anything examined on test beyond section 3 is labelled as exploratory in the results
   document and does not change the headline.

## 8. What to expect, written down beforehand

The same numbers on the validation sets, for comparison afterwards (20% review,
"doubtful no-damage first"; brackets: 95% tile-bootstrap intervals):

| | val-tiles (training disasters) | val (two unseen floods) |
|---|---|---|
| missed-damage rate, nobody reviewed | 0.163 (0.120-0.204) | 0.374 (0.342-0.405) |
| missed-damage rate, 20% random review | 0.130 | 0.299 |
| missed-damage rate, 20%, the queue in random order | 0.121 | 0.258 |
| **missed-damage rate, 20%, doubtful no-damage first** | **0.022 (0.010-0.042)** | **0.185 (0.158-0.214)** |
| accuracy on accepted, 20% | 0.878 | 0.579 |
| macro-F1 on accepted, 20% | 0.735 | 0.308 |
| share of predicted damage that is real | 0.73 | 0.21 |
| accuracy / macro-F1, nobody reviewed | 0.876 / 0.705 | 0.640 / 0.313 |

What I expect, so that it can be held against the result: test is two unseen disasters
of other types than val (a wind hurricane, a wildfire), so it should look like val
rather than like val-tiles, that is, clearly worse than val-tiles. The training set
contains a wildfire (socal-fire) and wind/flood hurricanes, so santa-rosa-wildfire may
do better than hurricane-michael. val took part in choosing the checkpoints and test
did not, which by itself argues for test being somewhat worse than val. Test also has
another class mix (a larger share of minor-damage and destroyed buildings), which moves
every one of these metrics for reasons unrelated to model quality; the results document
has to separate the two as far as it can.
