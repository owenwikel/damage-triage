# Split proposal: hold out whole disasters (DRAFT, not final)

Status: **draft for discussion.** Nothing in the code uses this split yet, and no model
has been trained.

## Why hold out whole disasters

The downloaded `train/` and `test/` folders are the official xBD split. They split
**tiles** at random *within* the same 10 disasters (both folders contain all 10). A model
evaluated that way has already seen the same sensor, season, roof styles, and often the
neighbouring street. The real use case is a *new* disaster, so every tile of a disaster
goes into exactly one of train / val / test.

To do that, the two folders are **pooled** (2,799 + 933 = 3,732 tiles; 213,644 classified
buildings plus 4,005 un-classified) and re-split by disaster.

## Pooled counts per disaster (train + test folders)

| disaster | type | tiles | no-damage | minor | major | destroyed | un-classified | total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| palu-tsunami | tsunami | 155 | 36,382 | 1 | 821 | 6,203 | 547 | 43,954 |
| mexico-earthquake | earthquake | 159 | 43,350 | 194 | 49 | 3 | 86 | 43,682 |
| hurricane-harvey | flooding | 427 | 15,591 | 3,534 | 10,612 | 597 | 395 | 30,729 |
| hurricane-michael | wind | 441 | 18,235 | 6,570 | 2,311 | 940 | 287 | 28,343 |
| hurricane-matthew | wind | 311 | 3,132 | 8,887 | 2,281 | 2,826 | 1,002 | 18,128 |
| santa-rosa-wildfire | fire | 300 | 12,507 | 97 | 72 | 4,435 | 65 | 17,176 |
| socal-fire | fire | 1,130 | 12,221 | 111 | 86 | 1,785 | 544 | 14,747 |
| midwest-flooding | flooding | 359 | 10,405 | 190 | 144 | 123 | 426 | 11,288 |
| hurricane-florence | flooding | 427 | 6,365 | 182 | 1,626 | 64 | 477 | 8,714 |
| guatemala-volcano | volcano | 23 | 665 | 12 | 9 | 26 | 176 | 888 |
| **all** | | **3,732** | **158,853** | **19,778** | **18,011** | **17,002** | **4,005** | **217,649** |

(`type` is the label file's `disaster_type`; note Harvey and Florence are hurricanes
labelled `flooding`.)

## The constraints that drive the choice

1. **Minor damage is almost only in three hurricanes.** Harvey, Michael and Matthew hold
   18,991 of 19,778 minor-damage buildings (96%). Every other disaster has fewer than 200.
   Any held-out set without one of those three will have a few hundred minor examples at
   most, and minor-damage scores on it will be noisy.
2. **Major damage is concentrated in Harvey** (10,612 of 18,011, 59%). Harvey should stay
   in train, or the model sees very little major damage.
3. **Destroyed is spread over Palu, Santa Rosa, Matthew and SoCal.** Holding out one of
   these gives a test set with plenty of destroyed buildings.
4. **Keep every disaster *type* represented in train** for a first model, so test measures
   "new disaster" rather than "never-seen hazard type". (The harder "unseen type" test is
   a good *second* experiment; see open questions.)
5. **Mexico and Palu are huge but nearly all no-damage** (Mexico has 3 destroyed). They add
   background variety to train but make poor evaluation sets.
6. The test set should be the **strongest** held-out set, because it's the number you report.
   Val mainly drives early stopping and threshold choices.

## Candidate splits

Percentages in brackets are the class mix within that part; "share of all" is how much of
each class that part takes.

### P1: current front-runner

- **Test:** hurricane-michael (wind) + santa-rosa-wildfire (fire)
- **Val:** hurricane-florence (flooding) + midwest-flooding (flooding)
- **Train:** harvey, matthew, palu, mexico, socal, guatemala

| part | tiles | buildings | no-damage | minor | major | destroyed |
|---|---:|---:|---:|---:|---:|---:|
| train | 2,205 | 152,128 | 111,341 (74.5%) | 12,739 (8.5%) | 13,858 (9.3%) | 11,440 (7.7%) |
| val | 786 | 20,002 | 16,770 (87.8%) | 372 (1.9%) | 1,770 (9.3%) | 187 (1.0%) |
| test | 741 | 45,519 | 30,742 (68.1%) | 6,667 (14.8%) | 2,383 (5.3%) | 5,375 (11.9%) |
| share of all (train / val / test) | | | 70 / 11 / 19% | 64 / 2 / 34% | 77 / 10 / 13% | 67 / 1 / 32% |

For: train keeps the two strongest minor sources (Harvey, Matthew), 77% of major, and one
of each hazard type (fire via SoCal, wind via Matthew, flooding via Harvey). Test has
thousands of every class and two different hazard types.
Against: **val is weak.** 372 minor and 187 destroyed is enough for early stopping on
overall loss, but not for per-class tuning.

### P2: variant of P1 with a fire in val

- Test as P1. **Val:** hurricane-florence + socal-fire. Train: harvey, matthew, palu, mexico, midwest, guatemala.

| part | tiles | buildings | minor | major | destroyed |
|---|---:|---:|---:|---:|---:|
| train | 1,434 | 148,669 | 12,818 | 13,916 | 9,778 |
| val | 1,557 | 23,461 | 293 | 1,712 | 1,849 |
| test | 741 | 45,519 | 6,667 | 2,383 | 5,375 |

For: val gets real destroyed counts (1,849). Against: **no fire disaster left in train**,
so the fire half of test becomes an unseen hazard type. SoCal is also 1,130 tiles (30% of
all tiles), so train shrinks to 1,434 tiles.

### P3: a hurricane in each held-out set

- **Test:** hurricane-matthew + santa-rosa-wildfire. **Val:** hurricane-michael. Train: the other 7.

| part | tiles | buildings | minor | major | destroyed |
|---|---:|---:|---:|---:|---:|
| train | 2,680 | 154,002 | 4,224 (2.8%) | 13,347 | 8,801 |
| val | 441 | 28,343 | 6,570 | 2,311 | 940 |
| test | 611 | 35,304 | 8,984 | 2,353 | 7,261 |

For: both val and test have thousands of minor-damage examples, so both are usable for
per-class decisions. Against: **train keeps only 21% of minor damage** (4,224, almost all
Harvey), which is likely to cripple the class that's already hardest. Matthew is also
unusual (82% of its classified buildings are damaged), which makes it a skewed test.

## Recommendation (for discussion)

Start from **P1**, and fix its weak val with a second, in-distribution val set:

- **val-disaster** = Florence + Midwest (as P1): reported each epoch as the
  "new disaster" signal.
- **val-tiles** = about 10% of the *train disasters'* tiles, split by **tile** (never by
  building, so no tile is in two parts), stratified by disaster. Used for per-class
  threshold tuning, where P1's val has too few minor/destroyed examples.
- **test** = Michael + Santa Rosa, touched once at the end.

The gap between val-tiles and val-disaster scores is itself useful: it measures how much
the model depends on having seen the disaster before.

## Open questions before finalising

1. Is "new disaster, known hazard type" (P1) the right headline, or should the headline
   be "unseen hazard type" (e.g. hold out both fires)? Could be run as a second split.
2. What to do with **un-classified** (4,005 buildings; painted as background in the
   official masks): drop, ignore in the loss, or keep as a separate class? The split
   above ignores them.
3. **Empty tiles** (516 train + 181 test have no buildings): keep them in whichever
   part their disaster lands in (current assumption), or subsample?
4. Tile-level near-duplicates: xBD tiles are cut from larger scenes, so neighbouring
   tiles overlap in context. That's fine *across* disasters, but it matters for the
   val-tiles idea above; grouping by scene would be safer if scene ids can be recovered
   (for example from `metadata.catalog_id` or `lng_lat` extents).
5. The full xBD release also has `tier3` (extra disasters) and `hold` sets that weren't
   downloaded. If they're added later, they change these numbers and may give a better
   val set.
6. Metric: with 74% no-damage, report macro-F1 over the four classes (and per class),
   not accuracy.

Numbers above were computed from `outputs/train/stats/class_counts_by_disaster.csv` and
`outputs/stats/class_counts_by_disaster.csv`, plus a count of post label files per disaster.
