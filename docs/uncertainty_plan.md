# Uncertainty plan: which signal to use

Written 2026-10-09. Validation data only (val-tiles and val); the test set has not been
loaded. Base model throughout: stack ResNet-18 with strong augmentation, seeds 0-2
(`runs/stack_aug_s0..2`), kept under the 3-epoch selection rule.

The question: for each building the model gives a class. Which number should say how
far to trust it, so that doubtful buildings go to a person? Three candidates.

## Recommendation

**Use the top probability of the 3-seed ensemble, temperature-scaled on val-tiles. Keep
the single model's temperature-scaled top probability next to it as the baseline. Do not
spend more time on MC dropout.**

Why, in three lines:

1. The ensemble does not rank mistakes better than a single model. What it removes is
   the luck of the seed: single seeds score 0.604-0.641 accuracy on val, and without
   labels there is no way to know which seed is the good one. The ensemble lands at the
   level of the best seed (0.640) every time.
2. It is already paid for. The three seeds are trained and their logits are exported
   (`outputs/predictions/`), so the ensemble now costs no GPU time at all.
3. MC dropout, as this model is built, gives the same answer as plain softmax at 6 times
   the prediction time.

One thing the recommendation does **not** fix: on val, every signal is much weaker than
on val-tiles (AUROC about 0.74 against 0.90). Choosing between these three methods moves
that number by about 0.01. The val gap is a model problem (`docs/gap_diagnosis.md`), not
a choice-of-uncertainty-method problem.

## The three candidates

### a) Plain softmax confidence (single model)

Confidence = the largest of the four class probabilities, after dividing the logits by
one temperature fitted on val-tiles (`src/calibrate.py`; T = 1.29-1.30 for the three
seeds).

- For: free. One forward pass, nothing to train, nothing to change.
- Against: one model's opinion. Which seed you happened to train changes val accuracy by
  almost 4 points and AUROC by 0.016.

### b) MC dropout

Leave dropout on at prediction time, run each building through the model 20 times,
average (`src/uncertainty.py`).

- Measured once, on the earlier seed-0 model (`docs/results.md` section 6): val AURC
  0.2017 against 0.2022 for plain softmax, val accuracy at 80% coverage 0.677 against
  0.676. No difference.
- Why: dropout sits only in front of the last layer. The backbone's 512 features are
  identical in all 20 passes, so the passes barely differ.
- To make it a real method, dropout would have to go inside the backbone. That changes
  the model, so all three seeds would have to be retrained and every base number in
  `docs/results.md` redone, with no evidence yet that it would beat the ensemble.

### c) Ensemble over the existing seeds

Average the probabilities of the three seeds, then temperature-scale the average.

- For: the result no longer depends on the seed; accuracy is at the level of the best
  member; it is the best calibrated on val before any scaling (ECE 0.033 against 0.090
  for a single model).
- Against: 3 trainings and 3 forward passes. And its ranking of mistakes is inside the
  range of its members, not above it.

## The numbers

From `docs/results.md` sections 3 and 5 (single = mean and range over the 3 seeds):

| | val-tiles accuracy | val-tiles AUROC | val accuracy | val AUROC | val AURC | val accuracy at 50% coverage |
|---|---|---|---|---|---|---|
| single model, softmax | 0.870 (0.869-0.872) | 0.901 (0.897-0.904) | 0.617 (0.604-0.641) | 0.739 (0.732-0.748) | 0.190-0.222 | 0.773-0.815 |
| ensemble of 3, softmax | 0.876 | 0.899 | 0.640 | 0.746 | 0.190 | 0.813 |
| MC dropout (earlier model) | same as that model's softmax | | | | 0.2017 vs 0.2022 | 0.796 vs 0.795 |

AUROC = chance that a correct prediction gets a higher confidence than a wrong one
(0.5 = useless). AURC = area under the risk-coverage curve, lower is better.

**Does disagreement between the seeds help?** An ensemble offers a second kind of
signal: not "how flat is the average" but "how much do the members disagree". Checked
quickly from the exported logits (CPU only, each member with its own temperature; not
yet part of `scripts/make_results.py`):

| signal (ensemble of 3) | val-tiles AUROC | val AUROC | val: share of mistakes in the least-confident 20% |
|---|---|---|---|
| top probability of the average | 0.900 | 0.748 | 34.7% |
| entropy of the average | 0.896 | 0.736 | 31.5% |
| margin between the top two classes | 0.899 | 0.732 | 33.1% |
| mutual information (disagreement only) | 0.750 | 0.564 | 22.7% |
| share of members voting for the ensemble's class | 0.604 | 0.628 | 28.6% |

Disagreement alone is a poor flag: 0.564 on val is close to a coin. The three seeds
give the same class for 94% of val-tiles buildings and 64% of val buildings, and on val
they are often wrong *together*. Three seeds of the same architecture, trained on the
same six disasters, share their blind spots. So the plain top probability stays the
signal to use; the fancier ensemble signals are not worth carrying.

## What each costs on the 8 GB GPU

Measured on this machine (RTX 2000 Ada, 8 GB), batch 512, mixed precision. Prediction
times are for val, 19,099 buildings, loading in the main process.

| | training | prediction time (val) | GPU memory, prediction | notes |
|---|---|---|---|---|
| softmax, single model | 1 run: about 15 min with 6 loader workers (17.5 GB RAM), about 28 min with 2 (9.6 GB RAM); GPU peak 4.1 GB | 11 s | 1.1 GB | 9 of the 11 s are reading and preparing crops, not the model |
| MC dropout, 20 passes | same 1 run | 65 s | 1.1 GB | each batch is loaded once and passed 20 times |
| MC dropout with dropout in the backbone | 3 new runs, 45-85 min, plus redoing the base results | 65 s per model | 1.1 GB | not built; needs a change to `src/model.py` |
| ensemble of 3 | 3 runs, 45-85 min. **Already done.** | about 34 s (3 x 11 s, one model at a time) | 1.1 GB | from the exported logits: 0 s, no GPU |

Memory is not the constraint for any of these. Training peaks at 4.1 GB of the 8 GB
and prediction at 1.1 GB; the models are loaded one after another, never together. The
real limits on this machine were RAM during training (`BLOCKERS.md`) and wall-clock
time. Exporting train, val-tiles and val for all three seeds took 5.5 minutes in total
(`scripts/export_predictions.py`).

## Next steps

1. Build the triage reports on the ensemble from `outputs/predictions/` rather than from
   `runs/*/preds_*.npz`: the exported tables carry the building id, tile and disaster, so
   per-disaster and per-tile breakdowns need no extra bookkeeping.
2. Report AUROC and coverage numbers per disaster on val (hurricane-florence and
   midwest-flooding fail for different reasons, `docs/gap_diagnosis.md`).
3. Leave MC dropout where it is (code kept and tested, out of the reports).
4. More seeds would not change this picture much: going from 1 to 3 gave no ranking
   gain. A member that differs in kind (the siamese model, or a model pushed to compare
   pre and post) is the more promising way to get disagreement that means something.
   Not planned until the val gap itself has been worked on.
