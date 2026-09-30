# Session report: unattended data-quality session (2026-09-30)

No model, dataset class, training loop or PyTorch training code was written. No `.tar`
file or data folder was modified. No `BLOCKERS.md` was needed: nothing blocked a task.
Details of each finding are in the linked docs; this page is the summary.

## What was done

| task | output | commit |
|---|---|---|
| 1. README | status, data source (xBD Challenge set, stored outside the repo), links to docs | `1b6d442` |
| 2. Data-quality audit | `docs/data_quality.md` sections 1-6, `scripts/audit_data.py`, `src/alignment.py` | `d686ac3` |
| 3. Split candidates | `docs/split_candidates.md`, `scripts/split_candidates.py` | `3a5aaad` |
| 4. Crop sizes | `docs/data_quality.md` section 7, `scripts/crop_sizes.py`, `outputs/crops_example.png` | `a87e89c` |
| 5. CI | `.github/workflows/tests.yml`, `requirements-ci.txt`, synthetic fixture + smoke tests | `5cf5249` |
| 6. Model primer | `docs/model_primer.md` (teaching only, toy snippets, 8 exercises) | `0deeb70` |
| 6b. Disaster table | `docs/disasters.md`, `scripts/disaster_table.py` | `c529e7c`, `ab56b40` |
| 6c. Label-noise candidates | `docs/label_noise.md`, `scripts/label_noise.py`, gallery in `outputs/label_noise/` | `6dfd10d` |

`git pull` at the start: already up to date (no README edits on GitHub to merge).

Tests: **41 pass locally**. They also pass in a fresh venv built only from
`requirements-ci.txt` (no PyTorch), which is what CI installs. Every script now has an
end-to-end smoke test on a synthetic one-tile dataset (`tests/conftest.py`). Those smoke
tests caught three edge-case bugs, all fixed:
- the val-tiles split could empty a small disaster out of train;
- crop examples crashed when no building matched a size;
- label noise crashed when there were no candidates.

None of the three changed any real-data result.

Everything the scripts generate is reproducible: each doc says which script made it, and
the scripts take `--data-dir`.

## Surprising things in the data

1. **Pre/post misalignment is real, but my automated measurement of it failed.** Phase
   correlation (whole-tile, per-patch, on intensity and gradients) reports shifts of a
   few pixels, but a null control on *unrelated* image pairs reports nearly the same
   sizes. So the numbers are mostly noise floor (`docs/data_quality.md` section 6). By
   eye, though, the post image is visibly off its polygon for some buildings: about 10 px
   for a Palu example, a few px for Michael and Florence (section 7 figure). Polygons
   match the *pre* image.
2. **The post image was usually taken from a steeper angle and at coarser resolution.**
   Post off-nadir is larger than pre for 9 of 10 disasters (Harvey 43° vs 25°), and
   `pan_resolution` is coarser for 7 of 10 (Mexico 0.66 vs 0.34 m). That fits both the
   misalignment (parallax) and the visibly blurrier Mexico post images
   (`docs/disasters.md`).
3. **Pre and post label files agree almost exactly.** Same buildings in every tile. For
   92% of buildings the outline is identical; almost all the rest differ by about 1 px,
   with no consistent direction. So the post polygon wasn't redrawn to follow the post
   image, and the labels don't correct item 1.
4. **Guatemala is 19.8% un-classified**, against 0.2-5.5% for every other disaster.
5. **Hurricane Michael has the most label-noise candidates by far:** 9.8% of its buildings
   disagree with all 5 nearest neighbours, against 0.2-1.6% for most disasters (Matthew
   5.8%). They are mostly damaged buildings surrounded by no-damage ones: 1,091 labelled
   minor and 563 labelled major, out of 2,753 (`outputs/label_noise/candidates.csv`).
6. **No tile appears in both the train and test folders, and no building appears whole in
   two tiles**, so pooling the folders for the split is safe. But 90% of tiles have a
   building within 50 px of the border, so neighbouring tiles share streets.
7. **One invalid polygon** (hurricane-michael_00000172, self-intersecting). `buffer(0)`
   repairs it to a valid polygon of the same area.
8. **Building sizes vary 2x between disasters**: median bbox side 23 px (Matthew,
   Guatemala) to 49 px (Harvey). With 16 px of context, a 64 px crop fits only 17% of
   Harvey's buildings but 76% of Matthew's.

## Open questions for you

1. **Label-noise gallery:** could you mark the 30 examples in `docs/label_noise.md`
   (the last column) after looking at `outputs/label_noise/gallery_0*.png`? That gives a
   first estimate of how many candidates are real mistakes, especially for Michael.
2. **Misalignment:** is it worth measuring properly? Options are hand-marking matching
   points on ~10 tiles, or feature matching with OpenCV (not installed). Or is "a few
   pixels, sometimes ~10" enough to design around (padding, siamese features)?
3. **torchvision is not installed** in the venv (only torch 2.6.0+cu124). Pretrained
   backbones need it; the matching build is torchvision 0.21.0 from the cu124 index. I
   didn't install it because the environment setup is yours (see `SETUP_NOTES.md`).
4. **CI:** the workflow runs on this push. I couldn't check the result from here (no `gh`
   CLI); please look at the repo's Actions tab. It's untested on Linux until then.

## Decisions waiting on you

1. **Split:** choose among P1, P1 + val-tiles, P2 and P3 (`docs/split_candidates.md`), or
   an "unseen hazard type" variant. If you use val-tiles, decide whether to group
   neighbouring tiles first, since tiles share borders.
2. **Un-classified buildings:** drop them, ignore them in the loss, or keep them for
   building-vs-background only. This matters most for Guatemala.
3. **The invalid polygon:** repair with `buffer(0)`, or drop it. Either way, do it in the
   label loader.
4. **Tiny and edge-cut buildings:** keep, filter by a size threshold, or weight. 3,097
   buildings are under 50 px², and 6.6% touch the tile edge (15.5% of un-classified).
5. **Crop size and context padding:** `docs/data_quality.md` section 7 gives the
   trade-offs per disaster and class.
6. **Pre/post fusion** (stacking vs siamese vs difference image), in light of the
   misalignment and resolution differences above. The trade-offs are in
   `docs/model_primer.md` section 3.
