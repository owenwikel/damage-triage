# Session report (2026-10-01)

## Session 3 (11:42-13:25): stopped early at your request

You asked me to stop all background training at 13:24 because you were unplugging. All
training was stopped then: no Python process is left running and the GPU is idle. The
batch was about half done. **Validation only; the test set was not touched.**

### Which runs finished

All runs are the stack model with strong augmentation, under the new selection rule
(mean of val-tiles and val macro-F1, averaged over a 3-epoch window).

| setting | seed | status | val-tiles acc / macro-F1 | val acc / macro-F1 |
|---|---|---|---|---|
| base (`stack_aug`) | 0 | finished | 0.871 / 0.693 | 0.604 / 0.304 |
| base | 1 | finished | 0.872 / 0.697 | 0.641 / 0.310 |
| base | 2 | finished | 0.869 / 0.693 | 0.607 / 0.299 |
| per-image normalisation (`stack_aug_pernorm`) | 0 | finished | 0.866 / 0.679 | 0.483 / 0.247 |
| per-image normalisation | 1 | finished | 0.866 / 0.681 | 0.477 / 0.237 |
| per-image normalisation | 2 | **stopped during epoch 10 of 10**; no results saved, not used anywhere | | |
| histogram matching (`stack_aug_histmatch`) | 0, 1, 2 | **not started** | | |

`runs/stack_aug_pernorm_s2/` holds a partial checkpoint and log from the stopped run;
it has no `metrics.json`, so the report scripts ignore it. It must be retrained from the
start, not resumed.

### The one table: every setting, mean ± spread over seeds

| setting | seeds | val-tiles accuracy | val-tiles macro-F1 | val accuracy | val macro-F1 |
|---|---|---|---|---|---|
| majority baseline | | 0.809 | 0.224 | 0.878 | 0.234 |
| base | 3 | 0.870 ± 0.001 | 0.694 ± 0.002 | 0.617 ± 0.021 | 0.304 ± 0.005 |
| base, ensemble of 3 | | 0.876 | 0.705 | 0.640 | 0.313 |
| per-image normalisation | 2 of 3 | 0.866 ± 0.001 | 0.680 ± 0.001 | 0.480 ± 0.004 | 0.242 ± 0.007 |
| per-image normalisation, ensemble of 2 | | 0.871 | 0.688 | 0.488 | 0.238 |
| histogram matching | 0 of 3 | not run | | | |

± is the standard deviation over seeds; ranges are in `docs/results.md` section 2.

### What each step of the batch reached

1. **Selection rule (3-epoch window): done.** `smoothed_score` in `src/train.py`. With it
   the three base seeds agree on val (accuracy 0.604-0.641); under the old single-epoch
   rule the same setting gave 0.613-0.761.
2. **MC dropout dropped: done.** Out of the reports; `docs/results.md` section 6 records it
   as tried and set aside, with its numbers. The code stays in `src/uncertainty.py`.
3. **Does uncertainty flag failure on val? Done for the base** (`docs/results.md` section 3).
   Single model, mean of 3 seeds, val against val-tiles: AUROC 0.739 against 0.901; the
   least-confident 20% hold 33% of the mistakes (73% on val-tiles) and the
   least-confident 50% hold 72% (97%); mean confidence 0.644 against 0.874, falling about
   as much as accuracy does. The ensemble is inside the range of its members (AUROC
   0.746). Short answer: partly. Confidence carries real information on val, far less
   than on val-tiles.
4. **Gap diagnosis: done** (`docs/gap_diagnosis.md`), on the three base seeds.
   - midwest-flooding: the *pre* image carries the problem. Given the post image twice,
     the model calls 96% of buildings undamaged (the truth is 96%) and accuracy is 0.936;
     given the pre image twice it calls only 55% undamaged. Normal accuracy is 0.627.
   - hurricane-florence: either image twice gives 88% undamaged; the real pair gives 60%
     (truth 77%). There it is the difference between the two captures that reads as
     damage. Florence also has the largest pre-to-post brightness change of any disaster
     (-11.7 on a 0-255 scale).
   - On val-tiles the model barely uses the pre image: accuracy is 0.870 normally and
     0.869 with the post image in both slots.
   - Average colours of the val disasters are mostly inside the range of the training
     disasters, so overall colour alone does not obviously explain the gap.
5. **Cheap fixes: half done.** Per-image normalisation, on 2 seeds, is clearly *worse* on
   val (accuracy 0.480 against 0.617; the seeds' ranges do not overlap) and slightly
   worse on val-tiles. Histogram matching was not run, so the batch's main comparison is
   incomplete.
6. **This report, commit, push: done.** `docs/error_analysis.md` and the demo figures
   were rebuilt for the new base seed 0 before the stop.

### Not verified before stopping

The full test suite was not re-run after the last two small edits (a summary paragraph
in `scripts/gap_diagnosis.py`, a wording fix in `scripts/make_results.py`). Both scripts
were run on the real data afterwards and produced the documents committed here. The
suite passed (the affected smoke tests included) just before those edits.

### To pick up from here

```powershell
$py = ".\.venv\Scripts\python.exe"
Remove-Item -Recurse runs\stack_aug_pernorm_s2          # the partial run
& $py -m src.train --config configs\stack_aug_pernorm.json --seed 2 --num-workers 6
foreach ($s in 0, 1, 2) { & $py -m src.train --config configs\stack_aug_histmatch.json --seed $s --num-workers 6 }
& $py scripts\make_results.py
```

About 15 minutes per per-image run and an estimated 20-25 per histogram-matching run
(not measured), with 6 loader workers: each run then owns about 17.5 GB of RAM, and the
machine had 17-21 GB free during these runs.

### Decisions for you

1. **Finish histogram matching (and the third per-image seed)?** About 1.5 hours. Given
   the diagnosis, I would still run histogram matching: Florence's problem looks like a
   pre/post difference, which is what matching targets. It will not help midwest, where
   the pre image itself is the issue.
2. **Per-image normalisation looks like a dead end** on 2 seeds. Drop it, or spend 15
   minutes on the third seed for completeness?
3. **The model mostly ignores the pre image on the training disasters.** That may be the
   root of the val gap: it classifies what the post image looks like instead of what
   changed. Worth deciding whether to push the model towards comparing (for example by
   training with some pairs where post is replaced by pre and the label set to
   no-damage) before trying more preprocessing.
4. **Loader workers.** I used 6 instead of 2 to fit the batch into the time; memory was
   logged every epoch and stayed safe, but with less headroom than at 2 workers.
5. Still open from before: when to open the test set (not yet, in my view); the batch cap
   at 512; the 0.5 factor on the stack model's copied first-layer weights.

---

# Earlier: sessions 1 and 2 (2026-10-01)

The runs described below used the earlier selection rule and are archived locally in
`runs/old_rule/` (not in git).

Session 1 (00:15-01:15) built the pipeline and trained the first model, then stopped on a
memory problem. Session 2 (09:07-11:00) fixed that and ran everything that was missing,
in the order you gave. This report covers both.

**The test set was never loaded, evaluated or looked at.** Every loader goes through
`src/split.py`, which raises unless `--final` / `final=True` is given, and the crop cache
contains no test crops. The optional test that re-derives the split from raw data
(`XBD_DATA_DIR`) was not run, because it reads test tiles' label files.

**val is now used for model selection**, as you decided: the kept checkpoint maximises
the mean of the val-tiles and val macro-F1, and the architecture and augmentation were
chosen with val in view. val numbers below are therefore somewhat optimistic. This is
noted in `docs/results.md`, `docs/split_decision.md`, `README.md` and
`docs/WALKTHROUGH.md`.

## What ran in session 2

| step | what | result |
|---|---|---|
| prep | selection on the mean of val-tiles and val macro-F1; 2 loader workers; RAM and GPU memory logged every epoch; per-item loading made about 4x cheaper | `9b5646d` |
| 1 | `stack_s0`, same settings as siamese | stack is ahead on the selection score (0.491 vs 0.480) |
| 2 | `stack_aug_s0`: strong colour jitter, independent for pre and post, plus +-4 px pre/post shifts | mixed on val: macro-F1 +0.011, accuracy -0.063. Taken as base model (higher val macro-F1 and selection score) |
| 3 | MC dropout (20 passes), seeds 1 and 2, 3-model ensemble | all done |
| 3 | calibration, triage, inspection budget, error analysis for every method | `docs/results.md`, `docs/error_analysis.md` |
| 4 | results, error analysis, demo figures, walkthrough section 11, README | rebuilt |

No run died, so nothing was retried. The last training run started at 10:26, well inside
the 4-hour limit. Tests: **120 pass, 1 skipped** (the real-data test above).

## Results across all runs (validation only)

Each row is the epoch the selection rule picks.

| run | what it is | epoch kept | val-tiles acc | val-tiles macro-F1 | val acc | val macro-F1 |
|---|---|---|---|---|---|---|
| majority baseline | always "no-damage" | | 0.809 | 0.224 | 0.878 | 0.234 |
| weighted random | guess by class frequency | | 0.615 | 0.248 | 0.660 | 0.236 |
| `siamese_s0` | siamese, base augmentation | 10 (weights saved: 8) | 0.880 | 0.704 | 0.483 | 0.256 |
| `stack_s0` | stack, base augmentation | 4 of 7 | 0.871 | 0.685 | 0.676 | 0.297 |
| `stack_aug_s0` | stack, strong augmentation | 9 of 10 | 0.870 | 0.693 | 0.613 | 0.308 |
| `stack_aug_s1` | same, seed 1 | 9 of 10 | 0.870 | 0.693 | 0.657 | 0.310 |
| `stack_aug_s2` | same, seed 2 | 4 of 7 | 0.862 | 0.669 | 0.761 | 0.337 |
| ensemble of 3 | `stack_aug` seeds 0-2 averaged | | 0.875 | 0.700 | 0.709 | 0.331 |

`siamese_s0` was trained before the selection rule changed: the row is epoch 10 from its
log, but its saved weights are epoch 8 (val 0.451 / 0.243). It is not the base model, so
it was not retrained.

Uncertainty methods on the base model (`stack_aug`):

| method | val-tiles ECE | val ECE | val-tiles AURC | val AURC | val accuracy at 80% / 50% coverage |
|---|---|---|---|---|---|
| max prob, raw | 0.033 | 0.094 | 0.024 | 0.204 | 0.677 / 0.793 |
| max prob, temperature-scaled (T = 1.27) | 0.015 | 0.043 | 0.024 | 0.202 | 0.676 / 0.795 |
| MC dropout, mean max prob | 0.016 | 0.042 | 0.024 | 0.202 | 0.677 / 0.796 |
| MC dropout, entropy | 0.016 | 0.041 | 0.024 | 0.202 | 0.670 / 0.793 |
| ensemble of 3 | 0.015 | 0.042 | 0.024 | 0.125 | 0.785 / 0.897 |
| ensemble of 3, temperature-scaled (T = 1.10) | 0.010 | 0.055 | 0.024 | 0.125 | 0.785 / 0.897 |

On val-tiles all methods are about the same: accuracy 0.87 overall, 0.96 at 80% coverage,
0.99 at 50%. Inspection budget of 20% of buildings, "most likely damaged first": about
78-79% of damaged buildings reached on val-tiles and 47% on val, for every method.

Time and memory: stack runs took 102-152 s of training per epoch (strong augmentation is
the slower end, because 2 workers then limit the speed), each run owned at most 9.6 GB of
RAM, the machine had 25-27 GB free throughout, and GPU memory peaked at 4.1 GB.

## What the results say

1. **The val gap is still there.** Every model is less accurate on val than always
   answering "no-damage" (0.878). Macro-F1 on val (0.30-0.34) is above the baselines
   (0.23) and far below val-tiles (0.69-0.70).
2. **On val, the seed matters more than the architecture or the augmentation.** Three
   identical runs differing only in seed score 0.613, 0.657 and 0.761 accuracy on val;
   on val-tiles they are within a point of each other. Within a run, val macro-F1 also
   jumps by several points between epochs. The stack-vs-siamese and augmentation
   comparisons, one run each, are smaller than this spread, so neither is settled.
3. **The ensemble is not better than its best member on val** (0.709 against 0.761
   accuracy; AURC 0.125 against 0.105). It beats seed 0, which happens to be the weakest
   member. What it gives is a result that does not depend on the luck of one seed.
4. **MC dropout adds nothing.** Its confidence is almost identical to the plain top
   probability, because dropout sits only in front of the last layer.
5. **Temperature scaling works on val-tiles and carries over to val for the single
   model** (ECE 0.094 to 0.043). For the ensemble it made val slightly worse (0.042 to
   0.055): an ensemble is already less over-confident.
6. **Stack with strong augmentation is far better calibrated on val than the first
   siamese model** (ECE 0.043 against 0.170 after scaling), and midwest-flooding went from
   0.31 to 0.63 accuracy. With one run each I cannot say how much of that is the
   architecture, the augmentation, the selection rule or the seed.
7. **Boundary hypothesis:** still holds on val-tiles (rank correlation between distance to
   a differently-labelled building and confidence +0.42) and still not on val (about 0),
   for every uncertainty method.
8. **Label-noise candidates** are over-represented among the least confident 10% on
   val-tiles (about 22% of them, where 10% would be chance) and barely on val (10-15%).
9. **Per disaster, val-tiles remains uneven:** mexico-earthquake 0.995 accuracy,
   hurricane-matthew 0.513 (no-damage recall 0.02), hurricane-harvey 0.646.

## Decisions waiting for you

1. **How to compare anything on val from here.** With this much seed-to-seed spread, a
   single run cannot show whether a change helps. Either every comparison gets 3 seeds
   (about 1.5 hours per setting), or comparisons are made on val-tiles plus a more
   stable val measure. My recommendation: 3 seeds per setting, and report mean and
   spread.
2. **Was choosing `stack_aug` as the base right?** The rule I applied was "higher val
   macro-F1" (0.308 against 0.297), which is inside the noise; val accuracy went the
   other way (0.613 against 0.676). A plain-stack ensemble was not trained, so the two
   were never compared on equal terms.
3. **Selection rule.** The mean of val-tiles and val macro-F1 kept epoch 4 for two of the
   four runs trained under it, at a spike in val. It does what you asked, but it partly selects on noise.
   Averaging the score over neighbouring epochs, or keeping the last epoch of a full
   cosine schedule, would be steadier. Your call.
4. **MC dropout.** To make it a real comparison, dropout would have to go inside the
   backbone, which changes training. Drop the method, or do that?
5. **The val gap itself.** Augmentation did not close it. Next candidates: per-image
   normalisation, matching post-image colours to the pre image, more context around
   buildings, or accepting that two flood disasters are too different from the training
   set and saying so. I have not tried any of these.
6. **When to open the test set.** Not yet, in my view: the comparisons above are not
   settled, and each look at test is final.
7. Still open from session 1: the batch cap at 512; the 0.5 factor on the stack model's
   copied first-layer weights; `outputs/demo/` tracked in git.

## Changes to what you specified

- The shift in the strong augmentation is +-4 px in the 128 px crop, which is about +-2 px
  in the original tile for a typical building.
- "Strong" colour jitter is: brightness, contrast and saturation +-30%, and a +-10% colour
  cast per channel. You gave no numbers; these are mine.
- MC-dropout passes use the single model's temperature, and the ensemble gets its own, so
  the methods are compared on equal terms.
- `scripts/finish_runs.ps1` from session 1 was removed; it was never run and is replaced
  by the commands in the README.

## Problems hit

- Session 1: batch-size probe stall, then the memory shortage (`BLOCKERS.md`, both
  resolved).
- Session 2: none during training. One generated sentence in `docs/results.md` first
  read "the stronger augmentation helps on val" for a +0.011 change; I rewrote the
  wording before committing.
