# Walkthrough: the modelling code, file by file

A study guide for the code written in the modelling session. For each file: what it does,
the lines worth understanding first, the design choices and what else could have been
done, and three questions an interviewer might ask, with answers.

Read the files in this order; each builds on the one before.

| # | file | one-line job |
|---|---|---|
| 1 | `src/split.py` | read the split; refuse to load the test set |
| 2 | `src/crops.py`, `scripts/build_crops.py` | cut each building out once and cache it |
| 3 | `src/dataset.py` | hand (pre, post, label) triples to PyTorch |
| 4 | `src/model.py` | two architectures: siamese and stack |
| 5 | `src/metrics.py` | accuracy, recall, macro-F1, ECE |
| 6 | `src/train.py` | the training loop |
| 7 | `src/calibrate.py` | temperature scaling |
| 8 | `src/uncertainty.py` | three confidence scores |
| 9 | `src/triage.py` | risk-coverage and inspection budget |
| 10 | `scripts/make_results.py`, `error_analysis.py`, `demo_figures.py` | turn runs into documents |

The numbers these produce are in `docs/results.md` and `docs/error_analysis.md`. This
page is about the code and the reasoning, not the results.

How the pieces connect:

```
xBD tiles ──build_crops──> crop cache ──CropDataset──> DataLoader ──> model ──> logits
                                                                         │
                 runs/<run>/preds_*.npz  <── train.py (best epoch) ──────┘
                          │
      calibrate (T) ──> uncertainty (confidence per building) ──> triage (what it buys)
```

---

## 1. `src/split.py`: the split and the test-set guard

**What it does.** Reads `splits/split_v1.csv` and returns the tiles (or the buildings) of
the parts you ask for. Asking for `test` raises `TestSplitError` unless you pass
`final=True`.

**Key lines.**

- `require_final(parts, final)`: the whole guard is this one function. Every loader calls
  it (`load_split`, `load_part_labels`, `CropDataset.__init__`, `build_crops.py`).
- `load_part_labels`: loops over the *tiles of the requested parts* and opens only their
  label files. Test tiles are not filtered out after loading; they are never opened.

**Design choices.**

- *A guard in code rather than a promise.* The alternative is discipline ("I just won't
  run it on test"). Discipline fails at 2 a.m.; an exception does not.
- *Opt-in with a loud name.* `--final` says what it means. A flag like `--include-test`
  would invite casual use.
- *The crop cache does not contain test crops at all.* Even code that bypassed the guard
  would find no file to read. Defence in depth.
- *Alternative:* keep the test data on a different disk or encrypted until the end.
  Stronger, but more ceremony than a solo project needs.

**Interview questions.**

1. *Why can't you just evaluate on the test set a few times during development?*
   Each look lets test-set information steer a decision (which model, which threshold,
   when to stop). After enough looks the chosen model is the one that happens to do well
   on that particular test set, and its score over-estimates performance on new data.
   The test set measures generalisation only if no decision depended on it.
2. *You have two validation sets. Why?*
   `val-tiles` (held-out tiles of the training disasters) has all four classes and all
   disaster types, so it can be used to stop training and to calibrate. `val` (two unseen
   disasters) answers the real question, "how does it do on a new disaster?", but it is
   floods only, with few minor-damage and destroyed buildings. The gap between the two
   is itself a result: it shows how much the model leans on familiar surroundings.
3. *What leaks are still possible with this guard?*
   The guard stops loading; it cannot stop indirect leaks. Examples: choosing the split
   itself after seeing per-disaster statistics (done here, deliberately, from labels
   only); ImageNet pretraining (unrelated data, not a leak); and val-tiles being
   spatially next to training tiles (measured in `docs/split_final.md`).

---

## 2. `src/crops.py` and `scripts/build_crops.py`: the crop cache

**What they do.** For every building in train, val-tiles and val: take its bounding box,
make a square around it (longest side + 16 px on each side), cut that square from the pre
image and from the post image, resize both to 128x128, and store them in two big `uint8`
arrays per part plus an `index.csv`.

**Key lines.**

- `crop_window`: the geometry. Square, centred on the box, padded, rounded to whole pixels.
- `cut_crop`: `image.crop(window).resize(...)`. PIL fills anything outside the image with
  black, which handles buildings at the tile border.
- `np.lib.format.open_memmap(..., mode="w+")`: creates a `.npy` file of the final size on
  disk and fills it piece by piece, so 13 GB never has to fit in memory.
- `ProcessPoolExecutor ... pool.map(crop_tile, tasks)`: tiles are independent, so the PNG
  decoding is spread over processes.

**Design choices.**

- *Cache instead of cropping on the fly.* On-the-fly cropping would decode two 1024x1024
  PNGs per building, per epoch. The cache trades disk (about 15 GB) for speed.
  The cost: crop size and padding are frozen. Changing them means rebuilding (about a
  minute and a half).
- *Fixed 128x128 output.* Every building looks the same size to the network; real size
  is lost. Kept in `box_side` so accuracy can be reported per size bucket. Alternative:
  a fixed ground window (say 96 px around the centroid, no resize) keeps scale but cuts
  large buildings and leaves small ones tiny.
- *16 px of context.* Enough to cover a few pixels of pre/post misalignment and to show
  the building's surroundings. More context helps with floods and fire (the ground
  changes), but lets neighbours' damage bleed into the crop.
- *Black padding at the tile edge* rather than mirroring: "no data" should look like no
  data. The `at_edge` flag lets us check whether those buildings are harder.
- *uint8, uncompressed.* Compression (e.g. `.npz`) would shrink the files but make random
  access slow; memory-mapping needs a plain array.

**Interview questions.**

1. *Why is a memory-mapped array a good fit here?*
   Training reads crops in random order. A memmap lets the OS fetch just the requested
   49 KB from disk and cache it; nothing is loaded up front, several worker processes
   can share the same file, and after the first epoch most of it is in the page cache.
2. *What information does resizing every crop to 128x128 destroy, and does it matter?*
   Absolute size and resolution: a 15 px shed is blown up 4x and looks blurry, a 300 px
   warehouse is shrunk. The model cannot use "this is a big building". It matters if
   damage patterns depend on scale; the per-size table in `docs/error_analysis.md` is how
   to find out.
3. *How would you verify the cache is correct?*
   Automated: tests cut a synthetic building and check it lands centred with the right
   padding (`tests/test_crops.py`), and that the darkened "destroyed" building of the
   fixture is darker in post than pre (`tests/test_split_guard.py`). By eye: a contact
   sheet of random crops per class. And the dataset asserts that index rows and array
   positions are in step.

---

## 3. `src/dataset.py`: the PyTorch Dataset

**What it does.** `CropDataset(cache_dir, part)` gives PyTorch one building at a time as
`(pre, post, label)`: two float tensors `[3, 128, 128]` and an integer. With
`augment=True` it applies random flips, 90-degree rotations and light colour jitter.

**Key lines.**

- `__len__` and `__getitem__`: all a Dataset has to provide.
- `.permute(2, 0, 1).float() / 255.0`: HWC bytes to CHW floats.
- `(pre - IMAGENET_MEAN) / IMAGENET_STD`: the normalisation the pretrained weights expect.
- `augment_pair`: one random decision per transform, applied to **both** images.
- `self._pre = None` and `_open()`: the file is opened lazily, inside each worker.

**Design choices.**

- *Flips and 90-degree rotations only.* Overhead imagery has no "up", so all eight
  symmetries are valid, and they are lossless. Arbitrary rotations would need
  interpolation and leave empty corners.
- *Same geometric transform for pre and post; independent colour jitter.* The geometry
  must match because the model compares the two. Colour differs between captures for
  reasons unrelated to damage, so small independent jitter teaches the model to ignore
  it. It is kept light because colour is also signal (burn scars, flood water). This is
  a judgement call worth experimenting with, especially given how differently the
  unseen disasters look.
- *Augmentation in the Dataset (CPU)* rather than on the GPU: simplest to read and test.
  GPU augmentation would be faster but harder to follow.
- *Not done:* random crops/scale jitter, mixup, heavier colour augmentation. Each is a
  possible next experiment.

**Interview questions.**

1. *Why must validation data not be augmented?*
   The validation score is used to compare epochs and models. Random augmentation would
   add noise unrelated to the model, and you would be measuring performance on altered
   images rather than on the data you care about.
2. *Why open the memmap lazily instead of in `__init__`?*
   With `num_workers > 0` on Windows, each worker is a new process that receives a
   pickled copy of the Dataset. An open memmap cannot be pickled sensibly; a path can.
   Each worker opens its own handle on first use.
3. *Where does randomness in augmentation come from, and is it reproducible?*
   From `torch.rand`. The DataLoader seeds each worker's torch generator from the main
   seed plus the worker id, so a run with the same seed and worker count sees the same
   augmentations. (numpy's global generator is *not* reseeded per worker; using it in a
   Dataset is a classic bug that gives every worker identical "random" numbers.)

---

## 4. `src/model.py`: siamese and stack

**What it does.** `build_model("siamese")` or `build_model("stack")` returns a network
that maps `(pre, post)` to four logits. Both use one pretrained ResNet-18.

| | siamese (late fusion) | stack (early fusion) |
|---|---|---|
| where pre and post meet | after global pooling: `[pre, post, post - pre]` | at the first convolution: 6 input channels |
| backbone passes per building | 2 (shared weights) | 1 |
| sensitivity to pre/post shift | low: position is pooled away first | higher: compares pixel to pixel |
| pretrained weights | used as they are | first layer adapted (copied to both halves, x0.5) |

**Key lines.**

- `self.backbone.fc = nn.Identity()`: removes ImageNet's classifier; the backbone now
  outputs the 512 pooled features.
- `torch.cat([pre, post], dim=0)`: both images through the backbone in one call.
- `torch.cat([pre_feat, post_feat, post_feat - pre_feat], dim=1)`: the fusion.
- `_six_channel_conv`: `new.weight[:, :3] = conv.weight * 0.5` (and the same for `3:`).

**Design choices.**

- *ResNet-18.* Small, fast, fits an 8 GB card with a large batch, and is well understood.
  Alternatives: ResNet-34/50 or a ConvNeXt (more capacity, slower), or a U-Net over the
  whole tile (what the xView2 winners did: segmentation instead of per-building
  classification).
- *Fine-tune everything* rather than freezing the backbone. Satellite images differ from
  ImageNet photos enough that frozen features would hold the model back.
- *Halving the copied first-layer weights* keeps the first layer's output at the scale the
  rest of the network was trained on. `tests/test_model.py` checks that feeding the same
  image into both halves reproduces the original layer's output exactly.
- *Difference features given explicitly.* Alternatives: only concatenate (the head must
  learn to subtract), or only the difference (loses "what kind of building is this").
- *Misalignment* (`docs/data_quality.md` section 6: unmeasured, plausibly a few pixels) is
  the main argument for late fusion; local change detection is the main argument for
  early fusion. Which wins is empirical: see `docs/results.md` section 3 (at the time of
  writing only the siamese model has been trained; see `BLOCKERS.md`).

**Interview questions.**

1. *What does "siamese" mean and why share the weights?*
   The same network, with the same weights, processes both inputs. Sharing guarantees
   that the two feature vectors are computed the same way, so their difference means
   "what changed" rather than "how two networks differ". It also halves the number of
   weights to learn.
2. *You changed a 3-channel layer into a 6-channel one. Why not initialise it randomly?*
   The first layer holds the most general, most transferable filters (edges, colours).
   Random initialisation would discard them and feed noise into pretrained later layers
   at the start of training. Copying keeps them; scaling by 0.5 keeps activations in the
   range the next layers expect.
3. *When would early fusion clearly beat late fusion?*
   When the two images are precisely registered and the change is small and local, e.g.
   a hole in a roof. A pixel-aligned comparison in the first layers can see it; a
   comparison of globally pooled summaries may not.

---

## 5. `src/metrics.py`: what "good" means

**What it does.** Confusion matrix, per-class recall/precision/F1, macro-F1, accuracy,
softmax, expected calibration error (ECE), reliability bins. Plain numpy.

**Key lines.**

- `np.bincount(y_true * n_classes + y_pred, ...)`: the confusion matrix in one line.
- `per_class_f1`: `2 TP / (2 TP + FP + FN)`.
- `expected_calibration_error`: bin by confidence, average `|accuracy - confidence|`
  weighted by bin size.

**Design choices.**

- *Macro-F1 as the headline metric.* With about 74% no-damage, accuracy rewards ignoring
  the rare classes. Macro-F1 gives each class a quarter of the score. Alternative: the
  xView2 score (harmonic mean of per-class F1), which punishes a single weak class even
  harder.
- *Classes absent from a subset are skipped in the macro average* (`nanmean`), instead of
  counting as zero. This matters for small groups in the error analysis; it makes
  macro-F1 not comparable between groups with different classes present.
- *ECE with 15 equal-width bins.* Standard, simple, but sensitive to binning and
  dominated by the crowded high-confidence bins. NLL is reported next to it as a
  binning-free check.

**Interview questions.**

1. *A model has 88% accuracy on val. Is that good?*
   Not by itself: predicting "no-damage" for everything scores about that on val. Compare
   against the baselines in `docs/results.md` and look at macro-F1 and per-class recall.
2. *Macro vs micro vs weighted F1?*
   Macro: average of per-class F1, every class equal. Micro: computed from total TP/FP/FN,
   equals accuracy in single-label classification. Weighted: per-class F1 weighted by
   class size, so again dominated by the majority. For rare-class problems, macro.
3. *What does ECE = 0.05 mean, and what does it hide?*
   On average, stated confidence is 5 points away from actual accuracy. It hides
   direction (over or under), per-class differences, and it can be near zero for a model
   that is 75% confident on everything with 75% accuracy, which is calibrated but useless
   for ranking. That is why triage is evaluated with risk-coverage curves, not ECE.

---

## 6. `src/train.py`: the training loop

**What it does.** Builds data, model, loss, optimiser and schedule from a `Config`, trains,
evaluates on val-tiles and val every epoch, keeps the best weights by val-tiles macro-F1,
stops early, and writes everything to `runs/<run_name>/`.

**Key lines.**

```python
with torch.autocast("cuda", enabled=amp):     # mixed precision
    logits = model(pre, post)
    loss = criterion(logits, label)
optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()                 # gradients
scaler.step(optimizer)                        # update weights
scaler.update()
scheduler.step()                              # learning rate follows the schedule
```

- `model.train()` / `model.eval()`: dropout and BatchNorm behave differently in each.
- `@torch.no_grad()` on `predict`: no gradient bookkeeping during evaluation.
- `class_weights`: `1 / sqrt(frequency)`, normalised to mean 1.
- `lr_factor`: linear warm-up, then cosine decay to zero.
- The early-stopping block: save `best.pt` on improvement, count epochs without.

**Design choices.**

- *Inverse-square-root class weights.* A middle path between no weights (majority
  dominates) and inverse frequency (rare classes over-predicted). Alternatives:
  oversampling rare classes, focal loss, or adjusting thresholds after training.
- *AdamW, one learning rate for all layers (5e-4), weight decay 0.01.* Simple and robust.
  A common refinement is a smaller rate for the pretrained backbone than for the new
  head.
- *Warm-up + cosine.* Warm-up protects pretrained weights from the head's large early
  gradients. Alternatives: step decay, one-cycle, reduce-on-plateau.
- *Batch size.* The probe finds the largest power of two that fits in 80% of GPU memory,
  capped at 512: beyond that, an epoch has too few update steps for a 10-epoch budget,
  and throughput (samples per second) was the same at 128, 256 and 512 anyway. On
  Windows the probe must not exceed the card: the driver spills into system RAM and
  crawls instead of failing (this happened once during the session).
- *Early stopping on val-tiles macro-F1.* Not on val: it is floods only and too small in
  two classes to steer by. The price: val-tiles is spatially close to train, so the
  stopping point is tuned to "familiar" data.
- *Sanity check first (`--overfit 256`).* Under a minute, and it rules out a broken pipeline.

**Interview questions.**

1. *What do `model.train()` and `model.eval()` actually change?*
   Dropout: random zeroing on / off. BatchNorm: normalise with the current batch's
   statistics and update running averages / use the stored averages. Nothing else. They
   do not turn gradients on or off; that is `torch.no_grad()`.
2. *What is mixed precision and what is the GradScaler for?*
   Running most operations in 16-bit floats: less memory, faster. 16-bit floats cannot
   represent very small gradients, so the scaler multiplies the loss before `backward()`
   (bigger gradients) and divides before the optimiser step.
3. *Your training loss keeps falling but val-tiles macro-F1 peaked at an early epoch. What is
   happening and what do you do?*
   Overfitting: the model is fitting particulars of the training crops. Early stopping
   already keeps the best epoch. To move the peak later and higher: more or stronger
   augmentation, more dropout or weight decay, a lower learning rate for the backbone,
   or more data.

---

## 7. `src/calibrate.py`: temperature scaling

**What it does.** Finds one number T on val-tiles so that `softmax(logits / T)` is as
well-calibrated as possible (minimum NLL), reports ECE and NLL before and after, and
draws reliability diagrams.

**Key lines.**

- `fit_temperature`: optimises `log_t` with L-BFGS on `cross_entropy(logits / exp(log_t), labels)`.
- `calibrated_probs`: `softmax(logits / T)`.

**Design choices.**

- *One parameter.* Cannot change the predicted class, cannot overfit. Alternatives: vector
  or matrix scaling (per-class parameters; can change predictions), isotonic regression,
  Dirichlet calibration. More flexible, need more data, can overfit.
- *Fitted on val-tiles, not on train.* On training data the model is near-certain and
  near-perfect; T fitted there would be useless. Fitting on val-tiles means the
  val-tiles "after" ECE is optimistic; val is the honest check.
- *Unweighted NLL.* Calibration is about the probabilities as they will be used, so the
  class weights of the training loss are not applied here.

**Interview questions.**

1. *Does temperature scaling change accuracy?* No. Dividing all logits by the same
   positive number keeps their order, so the argmax is unchanged.
2. *Why are neural networks over-confident?* Cross-entropy keeps rewarding larger logit
   margins on training examples even after they are classified correctly; with enough
   capacity the model pushes training probabilities towards 1, and that habit carries
   over to data where it is less accurate.
3. *T was fitted on data from the training disasters. What happens on a new disaster?*
   Calibration is a property of model *and* data distribution. Under distribution shift
   accuracy usually drops while confidence does not, so the model is over-confident
   again and a single T fitted in-distribution does not fix it. Compare the val-tiles and
   val rows in `docs/results.md` section 4.

---

## 8. `src/uncertainty.py`: three confidence scores

**What it does.** Produces, for every prediction, a confidence by three methods: max
softmax probability, MC dropout (20 stochastic passes), and a deep ensemble (average of
models trained with different seeds). `method_scores` returns them in one common format.

**Key lines.**

- `enable_mc_dropout`: `model.eval()` first, then `.train()` on dropout layers only.
- `mc_dropout_probs`: each batch goes through the model `passes` times.
- `predictive_entropy`: `-(p * log p).sum()`.
- `ensemble_probs`: mean of the models' softmax outputs.

**Design choices.**

- *Dropout only in the head.* That is where the models have it, so MC dropout here only
  samples the classifier head; the backbone is deterministic. Dropout inside the
  backbone would give a richer estimate but would change training.
- *Ensemble of 3.* Each extra member costs a full training run. Gains usually flatten
  around 5.
- *Average probabilities, not logits.* Keeps one over-confident member from dominating.
- *Two MC-dropout scores:* mean of per-pass max probability, and entropy of the mean.
  They mostly agree; entropy also reacts when the passes disagree about *which* class.

**Interview questions.**

1. *What is the difference between aleatoric and epistemic uncertainty, and which do these
   methods capture?* Aleatoric: noise in the data itself (a blurry crop, an ambiguous
   label); more data does not remove it. Epistemic: the model's ignorance; more data
   would. Max softmax mostly reflects aleatoric uncertainty as the model sees it.
   Disagreement between MC passes or ensemble members reflects epistemic uncertainty.
2. *Why keep BatchNorm in eval mode during MC dropout?* In training mode BatchNorm uses
   the batch's statistics, so a building's prediction would depend on its batch mates,
   and the running statistics would be updated by validation data. Only dropout should be
   random.
3. *Why do ensembles usually beat MC dropout?* Members trained from different seeds land
   in different regions of weight space and make different mistakes. MC dropout samples
   around one solution, so its samples are strongly correlated.

---

## 9. `src/triage.py`: is the confidence useful?

**What it does.** Risk-coverage curves and their area (AURC), accuracy and macro-F1 at
fixed coverage levels, and the inspection-budget view (share of damaged buildings
reached when only K% can be visited).

**Key lines.**

- `risk_coverage_curve`: sort by confidence descending; `cumsum(wrong) / n`.
- `at_coverage`: keep the top `round(c * N)`; the last kept confidence is the threshold.
- `inspection_recall`: sort by priority; `cumsum(is_damaged) / total_damaged`.

**Design choices.**

- *Risk-coverage rather than only ECE.* Triage needs a good *ranking* of predictions. A
  score can be perfectly calibrated and rank badly, or rank well and be miscalibrated.
- *AURC as the summary.* Mixes accuracy and ranking quality; fine for comparing methods on
  the same data. Alternative: AUROC of "is this prediction correct?", which isolates
  ranking.
- *Two inspection policies.* "Least confident" and "most likely damaged" serve different
  goals (catch model errors vs find damage). A real deployment would probably combine
  them, e.g. weight by expected cost of a miss.
- *"Damaged" = any of minor, major, destroyed.* A simplification; a missed destroyed
  building costs more than a missed minor one.

**Interview questions.**

1. *What does a flat risk-coverage curve tell you?* The confidence score carries no
   information about correctness: rejecting the least confident predictions removes
   right and wrong ones in equal proportion.
2. *How would you choose the operating threshold?* Fix a target (e.g. at most 5% error
   among auto-accepted, or the inspectors' capacity), read the threshold off held-out
   data, then verify on data from a different disaster, because the same threshold gives
   a different coverage and risk under distribution shift (see `docs/demo.md`).
3. *Why can macro-F1 on the accepted set go down while accuracy goes up?* The accepted set
   is dominated by easy no-damage buildings. The few rare-class buildings left in it are
   a tiny, noisy sample, and a handful of errors moves their F1 a lot.

---

## 10. The report scripts

- `scripts/make_results.py` reads `runs/*/metrics.json` and `preds_*.npz` and writes
  `docs/results.md` plus figures. No GPU, no training, safe to re-run.
- `scripts/error_analysis.py` joins predictions with `index.csv` (disaster, size, edge,
  position) and writes `docs/error_analysis.md` and a gallery under `runs/<run>/gallery/`.
- `scripts/demo_figures.py` draws whole tiles with predictions and REVIEW flags.

**Design choice worth copying:** training saves raw logits once; everything afterwards
(calibration, uncertainty comparison, triage, error analysis) is computed from those
files. Analyses are then fast, repeatable, and cannot accidentally differ in how the
model was run.

**Interview questions.**

1. *Why save logits rather than predicted classes or probabilities?* Logits are the most
   informative: probabilities, classes, temperature-scaled probabilities and ensembles
   can all be derived from them; the reverse is not true.
2. *How do you know the predictions line up with the right buildings?* Evaluation loaders
   do not shuffle, so prediction `i` is row `i` of that part in `index.csv`; the analysis
   scripts assert that the saved labels equal the index labels before using them.
3. *What would you check before trusting a per-group table (per disaster, per size)?*
   Group sizes and class counts inside each group. A recall computed from 12 buildings
   has a wide error bar; macro-F1 over groups with different classes present is not
   comparable.

---

## Things to try next (in rough order of expected value)

1. Close the val gap: stronger colour/contrast augmentation, per-image normalisation, or
   training on more disaster types. Check whether the gap is in the pre image, the post
   image, or both.
2. A lower learning rate for the backbone than for the head.
3. Larger context (padding 32 instead of 16) for flood and fire disasters.
4. Dropout inside the backbone, so MC dropout samples more than the head.
5. Ordinal loss: the four classes are ordered, and calling "destroyed" "no-damage" is
   worse than calling it "major-damage"; cross-entropy treats both the same.
