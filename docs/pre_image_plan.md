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

Per seed:

| setting | seed | part | macro_f1_real | macro_f1_swapped | drop_macro_f1 | changed_share |
|---|---|---|---|---|---|---|
| stack_aug | 0 | val | 0.304 | 0.255 | 0.049 | 0.383 |
| stack_aug | 0 | val-tiles | 0.693 | 0.620 | 0.073 | 0.110 |
| stack_aug | 1 | val | 0.310 | 0.257 | 0.053 | 0.359 |
| stack_aug | 1 | val-tiles | 0.697 | 0.629 | 0.069 | 0.101 |
| stack_aug | 2 | val | 0.299 | 0.249 | 0.050 | 0.389 |
| stack_aug | 2 | val-tiles | 0.693 | 0.614 | 0.079 | 0.112 |
<!-- section:pre-image-use:end -->
