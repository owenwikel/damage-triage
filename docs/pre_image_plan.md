# Plan: make the model use the pre image

Written 2026-10-10 shortly after midnight (commit 7b96b05), **before any of this was trained**. The stop rule in section 4
and the test in section 5 are fixed by this commit. Results are added below the line at
the end of the file, later, without touching what is above it.

Validation data only. The test set is not used anywhere in this plan.

## 1. The problem

From `docs/gap_diagnosis.md` (base model, mean of 3 seeds):

- On val-tiles the model scores 0.870 accuracy normally and 0.869 when the pre image is
  thrown away and the post image is fed in twice. It is not comparing before with
  after; it is reading the post image.
- On val that habit fails in two different ways. midwest-flooding: with the post image
  twice the model is 0.936 accurate, with the real pair 0.627. hurricane-florence:
  either image twice gives 88% "no-damage", the real pair gives 60% (truth 77%).
- Two colour fixes (per-image normalisation, histogram matching) made val worse
  (`docs/results.md` section 2).

The bet of this plan: a model that has to compare the two images, because the post
image alone cannot give the answer, will carry over to a new disaster better than one
that has learned what damaged roofs look like in six particular disasters. It is a bet.
Rubble is visible in the post image alone, so "what the post image looks like" is real
signal too, and taking it away may cost more than it gives.

## 2. Candidates

### A. Same-image pairs during training (PRIMARY)

With probability 0.25, a training example is replaced by a "nothing happened" pair: one
of the building's two images (pre or post, half each) is used for BOTH inputs, and the
label is set to no-damage. Each copy then goes through the usual strong augmentation
separately (own colour jitter, own colour cast, the second copy shifted by up to 4
pixels), so the two inputs are not pixel-identical.

Why it forces use of the pre image: take a destroyed building. Its real pair
(pre, post) must be called destroyed and the pair (post, post) must be called no-damage.
The second input is the same image in both. The only way to get both right is to look
at the first input.

- **Code:** one new option in `CropDataset` (`same_pair`, default 0 = off) and one in the
  training `Config` (`same_pair_prob`, default 0). About 15 lines. Model, loss, schedule
  and evaluation are untouched. Run name: `stack_aug_samepair_s<seed>`.
- **Cost:** the same as a base run (same model, same batch, same number of steps): about
  15 minutes per seed with 6 loader workers, 4.1 GB GPU peak, about 17.5 GB RAM.
- **Risks:**
  1. A shortcut: "the two inputs are the same image" may be detectable from fine
     texture, without understanding change. Then the model learns the trick and
     nothing else improves. The separate jitter and the shift make it harder, not
     impossible.
  2. A quarter of the training examples now say no-damage regardless of the building,
     so the model sees fewer real damaged pairs and may call damage less often:
     val-tiles recall of the damage classes could fall.
  3. It already answers no-damage for same-image pairs on florence (88%). What hurts
     there is the difference between two real captures, which this does not imitate
     beyond colour jitter.
- **Fixed settings:** probability 0.25; source image pre or post with equal chance; class
  weights computed from the true training labels as before (not from the altered ones).

### B. Siamese model with a difference-only head (BACKUP)

The siamese model of `src/model.py` runs both images through the same ResNet-18 and
hands the classifier `[pre features, post features, post - pre]`. The backup keeps only
the last part: the classifier sees `post features - pre features` (512 numbers) and
nothing else. The post image alone then carries no usable information: if the pre
image is replaced, the input to the classifier changes completely.

- **Code:** a new architecture name `siamese_diff` in `src/model.py` (a subclass that
  changes the fusion and the head's input size), selected with `arch` in the config.
  The existing `siamese` and `stack` classes are not edited. Trained with the same
  strong augmentation as the base. Run name: `siamese_diff_aug_s<seed>`.
- **Cost:** two backbone passes per building. The earlier siamese run took 149 s per
  training epoch (against about 70 for stack with 6 workers) and its GPU memory peaked
  at the card's 8.0 GB with an evaluation batch of 1024. Planned: training batch 512 as
  for the base, evaluation batch 256, expected 25-30 minutes per seed. If the first
  epoch's training time exceeds 300 s (a sign that the GPU is spilling into system
  RAM), the run is stopped and restarted with a training batch of 256; that would be
  recorded as a deviation, since it doubles the number of weight updates.
- **Risks:**
  1. It changes the architecture and the fusion at once, so a difference from the base
     cannot be pinned on "uses the pre image" alone.
  2. It throws away the absolute appearance of the post image. val-tiles macro-F1 may
     fall by more than the 0.02 the rule allows.
  3. A difference of pooled features is blind to where in the crop the change is.
  4. Two captures of an undamaged building in different light also give a non-zero
     difference; nothing here teaches the model to ignore that except the jitter.

### C. Difference channels for the stack model (not chosen)

Feed the stack model 9 channels: pre, post and post - pre. Cheap, but it does not force
anything: the network can go on reading the post channels and ignore the rest, which is
what it does today with 6. It adds information the first convolution could already
compute. Not run.

### Why A first

A changes only the training data, so the comparison with the base is clean: same
architecture, same cost, same everything else. It attacks the measured symptom directly
(val-tiles: post image twice scores the same as the real pair). B is the stronger
intervention and the more expensive one, and is tried only if A is dropped and time
allows. No third idea will be tried in this session.

## 3. Procedure (the same for primary and backup)

1. Implement behind new options; the base code path stays as it is, checked by a test
   that compares the base path's outputs with values recorded before the change.
2. Train seeds 0, 1 and 2 with the base's settings (`configs/stack_aug.json` plus the one
   change), 6 loader workers, one run at a time. Same selection rule as the base: mean of
   val-tiles and val macro-F1, averaged over 3 epochs (`select_on = "mean"`,
   `select_window = 3`).
3. RAM and GPU peaks are logged per epoch by `src/train.py` as usual.
4. Report accuracy, macro-F1 and per-class recall on val-tiles and val, per seed and
   as the mean, and the pre-image-use test of section 5.
5. Apply the stop rule with `scripts/stop_rule.py`. Its output is the decision.

All training stops at 06:07 on 2026-10-10 whatever state it is in. A setting without
three finished seeds is DROPPED.

## 4. Stop rule (fixed before training)

Baseline: `stack_aug`, seeds 0, 1, 2, each at its kept epoch.

| | mean over the 3 seeds |
|---|---|
| val macro-F1 | 0.304234 |
| val accuracy | 0.617327 |
| val-tiles macro-F1 | 0.694440 |

A candidate is **KEPT** only if all three hold for its 3-seed means:

1. val macro-F1 >= 0.304234 + 0.03 = **0.334234**
2. val accuracy >= **0.617327** (it does not fall)
3. val-tiles macro-F1 >= 0.694440 - 0.02 = **0.674440**

Otherwise it is **DROPPED**. Comparisons use unrounded numbers; `scripts/stop_rule.py`
(committed with this plan) applies them and prints the decision. Same seeds, same
selection rule as the base. The pre-image-use test below is reported next to the
decision but is not part of it: a model can pass that test and still be dropped.

Note on what the rule can and cannot show. val takes part in choosing each run's kept
epoch, for the base and for the candidate alike, so a KEEP is evidence on val, not on a
new disaster. And the base's seeds differ by 0.011 in val macro-F1 between best and
worst; a gain of 0.03 in the mean is well outside that, which is why the bar is there.

## 5. The pre-image-use test (fixed before training)

Question: does the model's answer depend on the pre image of *this* building?

For every building in val, the pre image is replaced by the pre image of a different
building of the same disaster; the post image and the label stay. "Different building"
is fixed once: within each disaster the buildings are put in a random order (numpy
generator, seed 0) and each building receives the pre image of the next one in that
order (the last receives the first's), so no building keeps its own and every model
sees exactly the same swapped pairs. Images are prepared as each model was trained
(normalisation), without augmentation.

Reported per model: macro-F1 on val with the real pairs, macro-F1 with the swapped pre
images, and the **drop** = real minus swapped; mean and range over the 3 seeds. Also
the share of buildings whose predicted class changes. The same on val-tiles, for
context.

How to read it: a model that ignores the pre image gives the same answers and the drop
is near 0. A model that compares gets a wrong "before" and should do worse, so its drop
should be larger than the base's. The base is measured first (`scripts/pre_image_use.py`)
and is the reference. A swapped pair is an artificial input, like the probes in
`docs/gap_diagnosis.md`: this is a test of behaviour, not a performance number.

## 6. What happens after the decision

- KEPT: export its predictions (train, val-tiles, val), rerun the risk-coverage analysis
  on its 3-seed ensemble, compare with the base in `docs/risk_coverage.md`.
- DROPPED: its numbers go into `docs/results.md` and the README as a rejected fix.
  Nothing else is built on it.

---

Everything below this line was added after training.

<!-- section:pre-image-use:start -->
## Pre-image-use test: results

Generated by `scripts/pre_image_use.py`. Each building's pre image is replaced by the pre image of another building of the same disaster (section 5). drop = macro-F1 with the real pairs minus macro-F1 with the swapped pre images; a larger drop means the model relies more on the pre image. Means over the seeds.

| setting | part | seeds | macro-F1, real pairs | macro-F1, swapped pre | drop | drop, range over seeds | accuracy, real pairs | accuracy, swapped pre | answers that change |
|---|---|---|---|---|---|---|---|---|---|
| stack_aug | val | 3 | 0.304 | 0.254 | 0.050 | +0.049 to +0.053 | 0.617 | 0.490 | 0.377 |
| stack_aug | val-tiles | 3 | 0.694 | 0.621 | 0.073 | +0.069 to +0.079 | 0.870 | 0.824 | 0.108 |
| stack_aug_samepair | val | 3 | 0.274 | 0.214 | 0.060 | +0.051 to +0.070 | 0.554 | 0.383 | 0.442 |
| stack_aug_samepair | val-tiles | 3 | 0.688 | 0.608 | 0.081 | +0.074 to +0.088 | 0.868 | 0.814 | 0.118 |
| siamese_diff_aug | val | 3 | 0.268 | 0.259 | 0.009 | -0.000 to +0.018 | 0.537 | 0.537 | 0.440 |
| siamese_diff_aug | val-tiles | 3 | 0.695 | 0.649 | 0.045 | +0.038 to +0.050 | 0.873 | 0.848 | 0.099 |

Per seed:

| setting | seed | part | macro_f1_real | macro_f1_swapped | drop_macro_f1 | changed_share |
|---|---|---|---|---|---|---|
| stack_aug | 0 | val | 0.304 | 0.255 | 0.049 | 0.383 |
| stack_aug | 0 | val-tiles | 0.693 | 0.620 | 0.073 | 0.110 |
| stack_aug | 1 | val | 0.310 | 0.257 | 0.053 | 0.359 |
| stack_aug | 1 | val-tiles | 0.697 | 0.629 | 0.069 | 0.101 |
| stack_aug | 2 | val | 0.299 | 0.249 | 0.050 | 0.389 |
| stack_aug | 2 | val-tiles | 0.693 | 0.614 | 0.079 | 0.112 |
| stack_aug_samepair | 0 | val | 0.286 | 0.228 | 0.058 | 0.441 |
| stack_aug_samepair | 0 | val-tiles | 0.690 | 0.602 | 0.088 | 0.122 |
| stack_aug_samepair | 1 | val | 0.276 | 0.206 | 0.070 | 0.445 |
| stack_aug_samepair | 1 | val-tiles | 0.687 | 0.612 | 0.074 | 0.115 |
| stack_aug_samepair | 2 | val | 0.260 | 0.209 | 0.051 | 0.439 |
| stack_aug_samepair | 2 | val-tiles | 0.689 | 0.609 | 0.080 | 0.117 |
| siamese_diff_aug | 0 | val | 0.259 | 0.250 | 0.009 | 0.447 |
| siamese_diff_aug | 0 | val-tiles | 0.697 | 0.647 | 0.050 | 0.100 |
| siamese_diff_aug | 1 | val | 0.254 | 0.254 | -0.000 | 0.463 |
| siamese_diff_aug | 1 | val-tiles | 0.696 | 0.647 | 0.048 | 0.098 |
| siamese_diff_aug | 2 | val | 0.291 | 0.273 | 0.018 | 0.410 |
| siamese_diff_aug | 2 | val-tiles | 0.692 | 0.654 | 0.038 | 0.099 |
<!-- section:pre-image-use:end -->

<!-- section:result-stack_aug_samepair:start -->
## Result: `stack_aug_samepair`

Generated by `scripts/candidate_report.py`. Baseline `stack_aug`; both with seeds 0, 1, 2, each run at the epoch kept by the 3-epoch selection rule. Validation only.

### Stop rule

- FAIL: val macro-F1 0.2742 >= 0.3042 + 0.03 = 0.3342
- FAIL: val accuracy 0.5536 >= 0.6173
- PASS: val-tiles macro-F1 0.6885 >= 0.6944 - 0.02 = 0.6744

**Decision: DROP.**

### Metrics per seed and mean

| setting | seed | part | accuracy | macro-F1 | recall no-damage | recall minor-damage | recall major-damage | recall destroyed |
|---|---|---|---|---|---|---|---|---|
| stack_aug | 0 | val-tiles | 0.871 | 0.693 | 0.923 | 0.485 | 0.731 | 0.722 |
| stack_aug | 1 | val-tiles | 0.872 | 0.697 | 0.922 | 0.499 | 0.728 | 0.734 |
| stack_aug | 2 | val-tiles | 0.869 | 0.693 | 0.918 | 0.498 | 0.740 | 0.719 |
| stack_aug | mean | val-tiles | 0.870 | 0.694 | 0.921 | 0.494 | 0.733 | 0.725 |
| stack_aug_samepair | 0 | val-tiles | 0.870 | 0.690 | 0.921 | 0.477 | 0.745 | 0.719 |
| stack_aug_samepair | 1 | val-tiles | 0.868 | 0.687 | 0.919 | 0.479 | 0.731 | 0.727 |
| stack_aug_samepair | 2 | val-tiles | 0.867 | 0.689 | 0.918 | 0.489 | 0.729 | 0.731 |
| stack_aug_samepair | mean | val-tiles | 0.868 | 0.688 | 0.919 | 0.482 | 0.735 | 0.726 |
| stack_aug | 0 | val | 0.604 | 0.304 | 0.636 | 0.231 | 0.415 | 0.219 |
| stack_aug | 1 | val | 0.641 | 0.310 | 0.681 | 0.204 | 0.399 | 0.214 |
| stack_aug | 2 | val | 0.607 | 0.299 | 0.646 | 0.234 | 0.368 | 0.171 |
| stack_aug | mean | val | 0.617 | 0.304 | 0.654 | 0.223 | 0.394 | 0.201 |
| stack_aug_samepair | 0 | val | 0.561 | 0.286 | 0.589 | 0.245 | 0.395 | 0.214 |
| stack_aug_samepair | 1 | val | 0.568 | 0.276 | 0.606 | 0.204 | 0.323 | 0.198 |
| stack_aug_samepair | 2 | val | 0.532 | 0.260 | 0.563 | 0.226 | 0.340 | 0.187 |
| stack_aug_samepair | mean | val | 0.554 | 0.274 | 0.586 | 0.225 | 0.352 | 0.200 |

### Training cost

| run | epoch kept | batch | loader workers | train s/epoch | RAM of this run, peak GB | RAM free, minimum GB | GPU peak GB (allocated) | GPU peak GB (reserved) |
|---|---|---|---|---|---|---|---|---|
| stack_aug_s0 | 10 of 10 | 512 | 6 | 71.48 | 17.51 | 17.45 | 2.50 | 4.08 |
| stack_aug_s1 | 10 of 10 | 512 | 6 | 67.93 | 17.26 | 19.22 | 2.50 | 4.08 |
| stack_aug_s2 | 9 of 10 | 512 | 6 | 67.52 | 17.26 | 20.58 | 2.50 | 4.08 |
| stack_aug_samepair_s0 | 10 of 10 | 512 | 6 | 64.91 | 17.01 | 20.68 | 2.50 | 4.08 |
| stack_aug_samepair_s1 | 10 of 10 | 512 | 6 | 64.16 | 17.01 | 20.85 | 2.50 | 4.08 |
| stack_aug_samepair_s2 | 10 of 10 | 512 | 6 | 64.27 | 17.13 | 20.57 | 2.50 | 4.08 |
<!-- section:result-stack_aug_samepair:end -->

<!-- section:result-siamese_diff_aug:start -->
## Result: `siamese_diff_aug`

Generated by `scripts/candidate_report.py`. Baseline `stack_aug`; both with seeds 0, 1, 2, each run at the epoch kept by the 3-epoch selection rule. Validation only.

### Stop rule

- FAIL: val macro-F1 0.2678 >= 0.3042 + 0.03 = 0.3342
- FAIL: val accuracy 0.5372 >= 0.6173
- PASS: val-tiles macro-F1 0.6950 >= 0.6944 - 0.02 = 0.6744

**Decision: DROP.**

### Metrics per seed and mean

| setting | seed | part | accuracy | macro-F1 | recall no-damage | recall minor-damage | recall major-damage | recall destroyed |
|---|---|---|---|---|---|---|---|---|
| stack_aug | 0 | val-tiles | 0.871 | 0.693 | 0.923 | 0.485 | 0.731 | 0.722 |
| stack_aug | 1 | val-tiles | 0.872 | 0.697 | 0.922 | 0.499 | 0.728 | 0.734 |
| stack_aug | 2 | val-tiles | 0.869 | 0.693 | 0.918 | 0.498 | 0.740 | 0.719 |
| stack_aug | mean | val-tiles | 0.870 | 0.694 | 0.921 | 0.494 | 0.733 | 0.725 |
| siamese_diff_aug | 0 | val-tiles | 0.875 | 0.697 | 0.929 | 0.464 | 0.728 | 0.727 |
| siamese_diff_aug | 1 | val-tiles | 0.873 | 0.696 | 0.926 | 0.464 | 0.728 | 0.737 |
| siamese_diff_aug | 2 | val-tiles | 0.872 | 0.692 | 0.927 | 0.459 | 0.716 | 0.729 |
| siamese_diff_aug | mean | val-tiles | 0.873 | 0.695 | 0.928 | 0.462 | 0.724 | 0.731 |
| stack_aug | 0 | val | 0.604 | 0.304 | 0.636 | 0.231 | 0.415 | 0.219 |
| stack_aug | 1 | val | 0.641 | 0.310 | 0.681 | 0.204 | 0.399 | 0.214 |
| stack_aug | 2 | val | 0.607 | 0.299 | 0.646 | 0.234 | 0.368 | 0.171 |
| stack_aug | mean | val | 0.617 | 0.304 | 0.654 | 0.223 | 0.394 | 0.201 |
| siamese_diff_aug | 0 | val | 0.522 | 0.259 | 0.551 | 0.167 | 0.350 | 0.294 |
| siamese_diff_aug | 1 | val | 0.478 | 0.254 | 0.491 | 0.223 | 0.440 | 0.193 |
| siamese_diff_aug | 2 | val | 0.611 | 0.291 | 0.656 | 0.164 | 0.307 | 0.326 |
| siamese_diff_aug | mean | val | 0.537 | 0.268 | 0.566 | 0.185 | 0.366 | 0.271 |

### Training cost

| run | epoch kept | batch | loader workers | train s/epoch | RAM of this run, peak GB | RAM free, minimum GB | GPU peak GB (allocated) | GPU peak GB (reserved) |
|---|---|---|---|---|---|---|---|---|
| stack_aug_s0 | 10 of 10 | 512 | 6 | 71.48 | 17.51 | 17.45 | 2.50 | 4.08 |
| stack_aug_s1 | 10 of 10 | 512 | 6 | 67.93 | 17.26 | 19.22 | 2.50 | 4.08 |
| stack_aug_s2 | 9 of 10 | 512 | 6 | 67.52 | 17.26 | 20.58 | 2.50 | 4.08 |
| siamese_diff_aug_s0 | 10 of 10 | 512 | 6 | 108.67 | 20.49 | 20.27 | 4.31 | 6.77 |
| siamese_diff_aug_s1 | 10 of 10 | 512 | 6 | 108.56 | 20.49 | 20.15 | 4.31 | 6.77 |
| siamese_diff_aug_s2 | 10 of 10 | 512 | 6 | 108.31 | 20.49 | 20.24 | 4.31 | 6.77 |
<!-- section:result-siamese_diff_aug:end -->

## Outcome (written by hand after both experiments, 2026-10-10 02:15)

**Both candidates were DROPPED by the stop rule of section 4, applied as written.**
Nothing was kept, so the base model (`stack_aug`) stays the model of record.

| | val macro-F1 | val accuracy | val-tiles macro-F1 | decision |
|---|---|---|---|---|
| needed to KEEP | >= 0.334 | >= 0.617 | >= 0.674 | |
| base (`stack_aug`) | 0.304 | 0.617 | 0.694 | reference |
| A. same-image pairs (`stack_aug_samepair`) | 0.274 | 0.554 | 0.688 | **DROP** (fails 1 and 2) |
| B. difference-only siamese (`siamese_diff_aug`) | 0.268 | 0.537 | 0.695 | **DROP** (fails 1 and 2) |

Means over seeds 0, 1, 2; per-seed numbers, per-class recall and training cost are in
the generated sections above.

What the two experiments say:

- **Neither helped on val; both made it worse**, by about 0.03 in macro-F1 and 0.06 to
  0.08 in accuracy. On val-tiles both are level with the base (0.688 and 0.695 against
  0.694). So the val-tiles condition was met and the two val conditions were not.
- **A did what it was built to do, a little.** With another building's pre image, its
  val macro-F1 drops by 0.060 where the base drops by 0.050, and 44% of its val answers
  change where 38% of the base's do. It leans on the pre image somewhat more, and is
  worse on val. Using the pre image more did not bring better transfer.
- **B cannot ignore the pre image, by construction, and on val that bought nothing.**
  Its val macro-F1 is 0.268 with the real pre image and 0.259 with another building's:
  a drop of 0.009, although 44% of its answers change. On the unseen disasters the
  building's own "before" is worth almost no more to this model than a random
  neighbour's. On val-tiles the same swap costs it 0.046, so there the pre image does
  carry information. B's seeds also disagree widely on val (accuracy 0.478 to 0.611).
- **The reference result was itself a surprise.** The base loses 0.050 on val and 0.073 on
  val-tiles when the pre image is swapped, and changes 38% of its val answers. That
  does not fit "the base mostly ignores the pre image" as `docs/gap_diagnosis.md` put
  it. Both observations are true: replacing the pre image by the *post* image barely
  changes val-tiles accuracy (0.870 to 0.869), while replacing it by *another
  building's* pre image does (0.870 to 0.824). The base uses the pre image at least as
  a picture of the building and its surroundings; whether it compares is a separate
  question that neither probe settles.

What this does not show: that comparing before and after cannot help. Two ways of
pushing the model towards it were tried, once each, with settings chosen in advance
(same-image probability 0.25; one difference-only head). Both are now rejected fixes
under the rule. The honest summary for the val gap is unchanged: four changes tried
with three seeds each (two colour fixes, two pre-image fixes), none of them better than
the base on the unseen disasters.

Deviations from the plan: none in the rule or the seeds. Backup runs used a training
batch of 512 as planned (first epoch 146 s of training, under the 300 s limit) and
peaked at 6.8 GB GPU and 20.5 GB RAM per run, more than the 17.5 GB of a base run; the
machine kept 20 GB free. All training ended at 02:09, inside the 06:07 limit.
