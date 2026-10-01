# Blockers

## 1. Training queue stopped: the machine ran low on memory (2026-10-01, about 00:46)

**What happened.** While the first full run (`siamese_s0`) was training, Windows was down
to about 5 GB of free memory out of 62 GB. Claude Code noticed and killed the two jobs
that were queued behind it, and told me not to start them again without being asked:

- `stack_s0` (the stack architecture, step 6b), which was waiting for `siamese_s0`;
- seeds 1 and 2 of the better architecture (step 8c, the ensemble) and the 20-pass
  MC-dropout run (step 8b), which were waiting for `stack_s0`.

Neither had started training: they were waiting. `siamese_s0` itself was not killed; its
memory use was high but steady, so I let it finish.

**Cause (mine).** The data loaders, not the model. That run used 8 worker processes for
training and 8 more for *each* of the two evaluation loaders, with evaluation batches of
1024: 24 processes at about 1.6 GB each, plus about 20 GB of prepared batches waiting in
the main process. Together about 59 GB.

**Fixed in the code** (`src/train.py`): evaluation now loads in the main process (no
extra workers) with batches of 512, and training defaults to 6 workers. Expected use is
roughly 15 GB. Not yet confirmed by a full run, because I was told not to start one.

**Not done because of this:**

| step | what is missing |
|---|---|
| 6 | the `stack` model, so no siamese-vs-stack comparison |
| 8b | MC dropout (needs the GPU; it was part of a stopped job) |
| 8c | the deep ensemble (seeds 1 and 2) |
| 9, 10, 11 | done, but only for the methods available from `siamese_s0` (max softmax probability, raw and temperature-scaled) |

**To finish, run these in order** (each training run is about 25-30 minutes for siamese,
less for stack; watch memory in Task Manager during the first one):

```powershell
.\.venv\Scripts\python.exe -m src.train --config configs\stack.json
# then, with <arch> = whichever has the higher val-tiles macro-F1 in docs/results.md:
.\.venv\Scripts\python.exe -m src.train --config configs\<arch>.json --seed 1
.\.venv\Scripts\python.exe -m src.train --config configs\<arch>.json --seed 2
.\.venv\Scripts\python.exe -m src.uncertainty --run runs\<arch>_s0 --passes 20
.\.venv\Scripts\python.exe scripts\make_results.py
.\.venv\Scripts\python.exe scripts\error_analysis.py --run runs\<arch>_s0
.\.venv\Scripts\python.exe scripts\demo_figures.py --run runs\<arch>_s0 --data-dir C:\Users\wikel\dev\xbd-data
```

`make_results.py` picks up whatever runs exist, so it can be re-run after each step.
None of these commands touches the test set.

## 2. Resolved, for the record: the batch-size probe stalled the first launch

The first attempt at `siamese_s0` sat for several minutes before training began. The
probe that looks for the largest batch that fits was trying batch sizes beyond the GPU's
8 GB; on Windows the driver then spills into system RAM and crawls instead of failing.
Fixed by stopping the probe at `max_batch_size` (512). About 8 minutes lost.
