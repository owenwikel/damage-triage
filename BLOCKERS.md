# Blockers

**Nothing is blocked at the moment.** Both entries below are resolved and kept for the
record.

## 1. Resolved: training queue stopped when the machine ran low on memory (2026-10-01)

**What happened.** During the first full run (`siamese_s0`), Windows was down to about
5 GB of free memory out of 62 GB. Claude Code killed the two jobs queued behind it (the
stack model; seeds 1-2 and MC dropout) and they were not restarted that night.

**Cause.** The data loaders, not the model: 8 worker processes for training and 8 more
for each of the two evaluation loaders, with evaluation batches of 1024. 24 processes at
about 1.6 GB each plus about 20 GB of prepared batches: about 59 GB.

**Fix** (`src/train.py`, `src/dataset.py`):

- 2 training workers; evaluation loads in the main process;
- the per-item work was made about four times cheaper, so 2 workers keep up;
- every epoch logs RAM and GPU memory to `runs/<run>/log.jsonl` and to the console.

**Confirmed the next morning.** Four full runs (stack, stack with strong augmentation,
and its seeds 1 and 2) and the MC-dropout pass completed without incident. Each run owned
at most 9.6 GB (main process plus workers). The machine as a whole showed 35-37 GB in
use and 25-27 GB free throughout; that figure includes about 16 GB used by other programs
before training started and the parts of the crop files that Windows keeps cached. No
run died, so the retry-with-smaller-settings rule was never needed.

## 2. Resolved: the batch-size probe stalled the first launch

The first attempt at `siamese_s0` sat for several minutes before training began. The
probe that looks for the largest batch that fits was trying batch sizes beyond the GPU's
8 GB; on Windows the driver then spills into system RAM and crawls instead of failing.
Fixed by stopping the probe at `max_batch_size` (512). About 8 minutes lost.
