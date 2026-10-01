# Run what the overnight session could not (see BLOCKERS.md), in order:
#   1. train the stack model
#   2. pick the architecture with the higher val-tiles macro-F1
#   3. train seeds 1 and 2 of it (for the 3-model ensemble)
#   4. MC dropout (20 passes) on its seed-0 model
#   5. rebuild docs/results.md, docs/error_analysis.md and the demo figures
#
# Usage, from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\finish_runs.ps1
#
# NOT RUN YET: this script was written but never executed. Each line is a command that
# was tested on its own, except the training runs with the new, lighter loader settings.
# Watch memory in Task Manager during the first run.
# Nothing here touches the test set: there is no --final anywhere.

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$data = "C:\Users\wikel\dev\xbd-data"

function Run($what) {
    Write-Host ">>> $what"
    & $py @args
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" }
}

if (-not (Test-Path "runs\stack_s0\metrics.json")) {
    Run "train stack, seed 0" -m src.train --config configs\stack.json
}

$better = & $py -c "import json; f=lambda a: json.load(open('runs/'+a+'_s0/metrics.json'))['best']['val-tiles']['macro_f1']; print('siamese' if f('siamese') >= f('stack') else 'stack')"
Write-Host ">>> better architecture on val-tiles macro-F1: $better"

foreach ($seed in 1, 2) {
    if (-not (Test-Path "runs\${better}_s$seed\metrics.json")) {
        Run "train $better, seed $seed" -m src.train --config "configs\$better.json" --seed $seed
    }
}

Run "MC dropout, 20 passes" -m src.uncertainty --run "runs\${better}_s0" --passes 20 --num-workers 2
Run "results" scripts\make_results.py
Run "error analysis" scripts\error_analysis.py --run "runs\${better}_s0"
Run "demo figures" scripts\demo_figures.py --run "runs\${better}_s0" --data-dir $data
Write-Host "Done. Read docs\results.md, then commit."
