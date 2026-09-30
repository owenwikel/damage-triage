# Setup Notes

## Environment

- OS: Windows 11 Pro, PowerShell
- Python: 3.11.9, via `py -3.11` (Python launcher), venv created at `.venv/`
- GPU: NVIDIA RTX 2000 Ada Generation Laptop GPU, 8 GB VRAM
- Driver: 595.95, supports up to CUDA 13.2 (confirmed via `nvidia-smi`), so CUDA 12.4 wheels are compatible

## What was installed

Installed in this order, each as its own `pip install` (no `-r` batch installs):

1. `torch` from `https://download.pytorch.org/whl/cu124` → resolved to `torch==2.6.0+cu124`
   - Verified: `torch.cuda.is_available()` returns `True`
   - Verified: `torch.cuda.get_device_name(0)` returns `NVIDIA RTX 2000 Ada Generation Laptop GPU`
   - Verified: `torch.version.cuda` returns `12.4`
2. `pandas` → 3.0.6
3. `numpy` → 2.4.6 (installed before pandas; pandas depends on it)
4. `matplotlib` → 3.11.2 (pulled in `pillow` as a dependency)
5. `shapely` → 2.1.2
6. `pillow` → 12.3.0 (already satisfied by matplotlib's dependency)
7. `pytest` → 9.1.1
8. `jupyter` → 1.1.1 (pulls in jupyterlab, notebook, ipykernel, etc.)

Full pinned list saved via `pip freeze` to `requirements.txt`.

## Problems encountered

- The `torch` install (2.5 GB wheel + CUDA runtime DLLs) took roughly 45–50 minutes to download and extract/copy into `site-packages` on this machine. It did not hang — `nvidia-smi`/process checks confirmed steady CPU usage and growing `site-packages/torch` size throughout — but if you're watching it live, expect it to look "stuck" during the file-copy phase for CUDA 12.4 wheels. No other issues.
- `pip freeze` showed two benign `WARNING: Cache entry deserialization failed, entry ignored` messages during the `pandas` install — cosmetic, did not affect the installed package.
- A stray `.claude/scheduled_tasks.lock` file (Claude Code's own local state, not project content) appeared in the working directory; added `.claude/` to `.gitignore` rather than committing it.

## How to activate the venv

From the repo root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

To run scripts without activating:

```powershell
.\.venv\Scripts\python.exe your_script.py
```

## Data folder

Per instructions, the `data/` folder was **not** touched (dataset download was still in progress). `data/` is included in `.gitignore` and does not exist in this repo yet — it should be populated separately once the download completes.

The dataset lives outside the repo at `C:\Users\wikel\dev\xbd-data\` and every script takes it via `--data-root`:

- `test\` — extracted from `test_images_labels_targets.tar`; 933 tiles / 1866 images.
- `train_images_labels_targets.tar` — download finished 2026-09-29 (last write 21:02 EDT, 8,381,754,539 bytes).
  Checked at 21:11: no curl process running, no writer holding the file, size unchanged for 9 min.
  SHA1 verified: `b37a4ef4ee9c909e2b19d046e49d42ee3965714b` (matches the expected value).
  The archive extracts to `train\`. Keep the `.tar` files.
