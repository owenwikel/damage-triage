# Session report: first modelling session (2026-10-01, 00:15-01:15)

**Short version.** The pipeline is built end to end and one model (siamese ResNet-18) is
trained and analysed on validation data. The session stopped after about 1 hour of the
planned 5: the machine ran low on memory during the first training run, the queued
training jobs were stopped, and I was told not to restart them without being asked. So
the stack model, the ensemble and MC dropout are **not done**. `BLOCKERS.md` has the
details; `scripts\finish_runs.ps1` runs everything that is missing.

**The test set was never loaded, evaluated or looked at.** Every loader goes through
`src/split.py`, which raises unless `--final` / `final=True` is given, and the crop cache
contains no test crops. I also did not run the optional test that re-derives the split
from the raw data (`XBD_DATA_DIR`), because it reads the coordinates in test tiles'
label files.

## What ran

| step | status | output | commit |
|---|---|---|---|
| 0. move split_decision.md | done | `docs/split_decision.md`; `docs/docs/` removed | `7ae8125` |
| 1. crop cache | done | `scripts/build_crops.py`, `src/crops.py`, `src/split.py` (guard). 168,477 buildings (train 134,660, val-tiles 14,718, val 19,099), **15.45 GB** in `C:\Users\wikel\dev\xbd-data\crops\`, built in 82 s | `45129ad` |
| 2. dataset | done | `src/dataset.py`, tests on a synthetic cache | `1cb4668` |
| 3. models | done | `src/model.py` (siamese, stack), `src/metrics.py` | `e0fc09c` |
| 4. training loop | done | `src/train.py`, `configs/`. Sanity check: 256 crops overfitted to **100%** in 4 epochs | `56a4f63` |
| 5. baselines | done | `docs/results.md` section 1 | `d83850f` |
| 6. siamese and stack | **half**: siamese trained; stack not | `docs/results.md` section 3 | `d83850f` |
| 7. calibration | done (on siamese) | `src/calibrate.py`, `docs/results.md` section 4 | `e5acc4f`, `d83850f` |
| 8. uncertainty methods | **a only**: max softmax. MC dropout and ensemble code is written and tested, not run | `src/uncertainty.py`, `docs/results.md` section 5 | `e5acc4f`, `d83850f` |
| 9. triage | done for the max-softmax method | `src/triage.py`, `docs/results.md` section 6 | `e5acc4f`, `d83850f` |
| 10. error analysis | done | `docs/error_analysis.md`, gallery in `runs/siamese_s0/gallery/` | `fa2202c` |
| 11. demo figures | done | `outputs/demo/` (6 tiles), `docs/demo.md` | `1d7946f` |
| 12. walkthrough, README, this report | done | `docs/WALKTHROUGH.md`, `README.md` | `c355c93`, final commit |

Tests: **112 pass, 1 skipped** (the real-data test mentioned above). CI now installs
CPU PyTorch so the model and dataset tests run there too.

## Results (validation only)

`siamese_s0`: 10 epochs, best epoch 8 by val-tiles macro-F1, batch 512, about 150 s of
training per epoch (196 s with evaluation), GPU peak 5.0 GB allocated / 8.0 GB reserved.

| | accuracy | macro-F1 | recall no-dmg / minor / major / destroyed |
|---|---|---|---|
| val-tiles, siamese | 0.879 | 0.708 | 0.93 / 0.50 / 0.74 / 0.74 |
| val-tiles, majority baseline | 0.809 | 0.224 | 1 / 0 / 0 / 0 |
| val-tiles, weighted random | 0.615 | 0.248 | 0.74 / 0.09 / 0.09 / 0.08 |
| val, siamese | **0.451** | **0.243** | 0.47 / 0.28 / 0.35 / 0.21 |
| val, majority baseline | 0.878 | 0.234 | 1 / 0 / 0 / 0 |
| val, weighted random | 0.660 | 0.236 | 0.74 / 0.09 / 0.09 / 0.08 |

| | val-tiles | val |
|---|---|---|
| ECE before / after temperature scaling (T = 1.60) | 0.045 / 0.011 | 0.272 / 0.170 |
| accuracy at 100 / 80 / 50% coverage | 0.879 / 0.965 / 0.994 | 0.451 / 0.496 / 0.586 |
| area under the risk-coverage curve | 0.020 | 0.371 |
| damaged buildings reached visiting 20%, "most likely damaged first" | 80.7% | 29.8% |
| same, visiting at random | 20% | 20% |

## Surprises

1. **The model fails on unseen disasters.** On val it is less accurate than always
   answering "no-damage", and its macro-F1 is at guessing level. midwest-flooding is the
   worst: only 31% of undamaged buildings are called undamaged. The crops were checked by
   eye and are correct; val imagery is visibly darker and greener than train, and the
   post-flood images are brown water. This is the main result of the session and the
   first thing to work on.
2. **Even val-tiles is uneven.** 0.879 overall hides mexico-earthquake at 0.997 (nearly
   all no-damage) next to hurricane-matthew at 0.537 (no-damage recall 0.10) and
   hurricane-harvey at 0.655.
3. **Your boundary hypothesis holds on val-tiles, not on val.** Mean confidence rises
   from 0.67 for buildings within 25 px of a differently-labelled building to 0.89 beyond
   200 px (rank correlation +0.50). On val there is no relation (+0.02): there the model
   is unsure everywhere.
4. **Label-noise candidates are lower-confidence** on val-tiles (mean 0.68 against 0.88),
   and the model agrees with their labels only 43% of the time (89% elsewhere).
5. **Some of the 20 most confident mistakes look like label problems.** Three are labelled
   no-damage where the building is simply gone in the post image (palu-tsunami tile 170).
   Most of the rest are damaged buildings called no-damage that look intact from above.
   Worth ten minutes with `runs/siamese_s0/gallery/confident_mistakes.png`.
6. **Size and tile edge matter less than expected.** No clear trend with size on
   val-tiles; edge-cut buildings are slightly worse (0.855 against 0.881 accuracy).
7. **Memory, not the GPU, was the bottleneck.** 24 data-loader processes used about 59 of
   62 GB. Throughput was the same at batch 128, 256 and 512, so a big batch bought
   nothing in speed.

## Decisions waiting for you

1. **Run the rest?** `powershell -ExecutionPolicy Bypass -File scripts\finish_runs.ps1`
   trains stack, then seeds 1 and 2 of the better architecture, runs MC dropout and
   rebuilds the documents: roughly 1.5 hours. The script has not been executed, and the
   lighter loader settings have not been through a full run; watch memory at the start.
2. **What to do about the val gap** before spending GPU time on ensembles of a model that
   does not transfer. Options, cheapest first: stronger colour and contrast augmentation;
   per-image normalisation; a lower learning rate for the backbone; larger context
   around buildings for floods. My recommendation: try the augmentation first, and judge
   it on val, not val-tiles.
3. **Early stopping on val-tiles picks epoch 8, but val is flat or worse after epoch 4.**
   val-tiles rewards fitting the training disasters. Do you want to keep selecting on
   val-tiles (as decided in `docs/split_decision.md`), or select on val and accept that
   val then stops being a clean estimate?
4. **Batch size.** You asked for the largest batch that fits 8 GB. The probe found 512 at
   4.2 GB; 1024 does not fit. I also cap at 512 by default, because a larger batch means
   too few update steps in 10 epochs. Say if you want the cap removed.
5. **Stack model's first layer.** You asked for the pretrained weights copied into both
   halves. I copy them and multiply by 0.5, so the layer's output keeps its original
   scale. Remove the 0.5 in `_six_channel_conv` if you want the plain copy.
6. **`outputs/demo/` is now tracked in git** (the rest of `outputs/` is still ignored), so
   the README and `docs/demo.md` can show the figures on GitHub. Other figures are in
   `docs/figures/`. 1.8 MB in total.
7. **Demo tiles.** val has only two disasters, so the six tiles are those two plus four
   val-tiles tiles (harvey, matthew, palu, socal). They were picked from the labels, not
   by model performance; the Florence tile is 27% correct and is shown as it is.

## Problems hit

- Batch-size probe stalled the first launch (it tried batches larger than the GPU;
  fixed, about 8 minutes lost).
- Memory pressure stopped the queue (see above and `BLOCKERS.md`).
- GPU reserved memory peaked at the full 8.0 GB during the run (5.0 GB actually in use;
  the rest is cuDNN trying out convolution algorithms). It did not crash, but there is no
  headroom at batch 512 for the siamese model.

The report of the earlier data-quality session is in git history (`08e2cb1`).
