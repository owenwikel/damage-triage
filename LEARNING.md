# LEARNING.md: a guided tour of the data-exploration code

This file walks through every file added for data exploration, in the order they were
built. It assumes you know basic Python (functions, lists, dicts, loops) and are still
learning pandas and NumPy. Obvious lines are skipped; anything that relies on a pandas,
NumPy, matplotlib or shapely behaviour you might not know yet is explained.

Run everything from the repo root with the venv's Python:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_xbd.py   --data-root C:\Users\wikel\dev\xbd-data\test
.\.venv\Scripts\python.exe scripts\dataset_stats.py --data-root C:\Users\wikel\dev\xbd-data\test
.\.venv\Scripts\python.exe scripts\visualize.py     --data-root C:\Users\wikel\dev\xbd-data\test
.\.venv\Scripts\python.exe -m pytest
```

Contents:

1. [The data in one page](#1-the-data-in-one-page)
2. [`scripts/inspect_xbd.py`](#2-scriptsinspect_xbdpy)
3. [`src/labels.py`](#3-srclabelspy)
4. [`tests/test_labels.py` and `pyproject.toml`](#4-teststest_labelspy-and-pyprojecttoml)
5. [`src/colors.py`](#5-srccolorspy)
6. [`scripts/dataset_stats.py`](#6-scriptsdataset_statspy)
7. [`scripts/visualize.py`](#7-scriptsvisualizepy)
8. [pandas / NumPy cheat sheet for this project](#8-pandas--numpy-cheat-sheet-for-this-project)
9. [Five exercises before writing a model](#9-five-exercises-before-writing-a-model)

---

## 1. The data in one page

A split folder (e.g. `xbd-data/test`) has three subfolders, and every file name follows
`<disaster>_<tile id>_<pre|post>_disaster`:

| folder | one file per image | what it is |
|---|---|---|
| `images/` | `socal-fire_00000319_post_disaster.png` | 1024x1024 RGB satellite image |
| `labels/` | `socal-fire_00000319_post_disaster.json` | building polygons + metadata |
| `targets/` | `socal-fire_00000319_post_disaster_target.png` | the polygons painted as a per-pixel mask |

A **tile** is one place on the ground, photographed twice: before (`pre`) and after
(`post`) the disaster. The test split has 933 tiles, so 1866 images.

Inside a label JSON:

```text
{
  "features": {
    "xy":      [ {"properties": {...}, "wkt": "POLYGON ((x y, x y, ...))"}, ... ],
    "lng_lat": [ same buildings, but in longitude/latitude ]
  },
  "metadata": { "disaster": "socal-fire", "disaster_type": "fire", "gsd": ..., ... }
}
```

- `xy` coordinates are **pixels** in the image (x = column, y = row, origin top-left).
- `wkt` ("well-known text") is a standard text format for shapes; `shapely` parses it.
- The damage class lives in `properties["subtype"]`, and **only in post files**. The exact
  strings are `no-damage`, `minor-damage`, `major-damage`, `destroyed`, `un-classified`.
- In the target masks, post pixels are 0 = background, 1..4 = the four damage levels.
  `un-classified` buildings are painted 0 (background), which `inspect_xbd.py` checks.

The full measured picture is in `outputs/data_report.md` (the `outputs/` folder is in
`.gitignore`, so regenerate it with the commands above if it's missing).

---

## 2. `scripts/inspect_xbd.py`

**Job:** walk the folder, count everything, check that files pair up, and record the
label format and damage strings. It deliberately does *not* use pandas; it's plain Python,
so it is a good warm-up.

```python
NAME_RE = re.compile(r"^(?P<disaster>.+)_(?P<tile>\d+)_(?P<phase>pre|post)_disaster")
```
A regular expression with **named groups**. `(?P<tile>\d+)` means "one or more digits,
and call that part `tile`". `.+` for the disaster is *greedy*, so for
`hurricane-harvey_00000123_pre_disaster` it grabs as much as it can while still letting
the rest match, giving `hurricane-harvey`. `m.group("tile")` pulls a part out afterwards.
Tile ids are kept as **strings** (`"00000123"`) so the leading zeros survive.

```python
counts = defaultdict(Counter)
counts[disaster][phase] += 1
```
`Counter` is a dict that starts every missing key at 0. `defaultdict(Counter)` is a dict
whose missing values are brand-new Counters. Together they let you count in two levels
without writing any "if key not in dict" checks.

```python
img_k, lab_k, tgt_k = tile_keys(images), tile_keys(labels), tile_keys(targets)
img_no_label = sorted(img_k - lab_k)
```
Sets of `(disaster, tile, phase)` tuples. `a - b` on sets = "in a but not in b". This is
the quickest way to find unpaired files.

```python
with Image.open(p) as im:
    sizes[im.size] += 1
```
Pillow's `Image.open` is **lazy**: it reads the header (size, mode) but doesn't decode
the pixels until you ask. That's why checking all 1866 images takes seconds.

```python
rng = np.random.default_rng(0)
sample_idx = rng.choice(len(targets), size=..., replace=False)
```
NumPy's modern random API. You create a **generator** with a fixed **seed** (0), then ask
it for numbers. Same seed means same "random" sample every run, which makes reports
reproducible. `replace=False` means no index is picked twice.

```python
arr = np.asarray(im)
vals, cnts = np.unique(arr, return_counts=True)
```
`np.asarray` turns the image into a NumPy array (shape `(1024, 1024)` for a one-channel
mask). `np.unique` returns the sorted distinct values and, with `return_counts=True`, how
many times each appears. It's the NumPy equivalent of `value_counts()`.

```python
for v, c in zip(vals.tolist(), cnts.tolist()):
```
`.tolist()` converts NumPy numbers (`np.uint8`) into plain Python ints. Without it you can
get surprising results: `np.uint8(250) + np.uint8(10)` wraps around to `4`, because
`uint8` only holds 0-255.

```python
pt = shapely_wkt.loads(f["wkt"]).representative_point()
uncls_vals[int(mask[int(pt.y), int(pt.x)])] += 1
```
To see what mask value sits under an `un-classified` building, we need a point
guaranteed to be *inside* the polygon. A centroid can fall outside an L-shaped building;
`representative_point()` never does. Note the index order `mask[row, col]` =
`mask[y, x]`. **NumPy indexes rows first**, which is the opposite of (x, y) order. This
flip causes a lot of image bugs.

The rest of `main()` builds a list of markdown lines `L` and joins them with `"\n"`.

---

## 3. `src/labels.py`

**Job:** turn ~933 post label files into **one pandas DataFrame with one row per building**.
Everything else in the project builds on this.

```python
DAMAGE_CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed", "un-classified"]
DAMAGE_LEVEL = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
```
Two views of the label. `damage` is the original string. `damage_level` is an ordered
number for the four real classes. `un-classified` gets **no** number on purpose: it means
"annotators couldn't tell", and giving it a level would silently teach a model something
false.

```python
for feat in data["features"]["xy"]:
```
We only read `xy` (pixel coordinates), because images and models work in pixels.

```python
if damage not in DAMAGE_CLASSES:
    raise ValueError(...)
```
**Fail loudly on surprises.** If the train set turns out to contain a sixth label string,
this stops you instead of quietly producing a wrong table.

```python
poly = shapely_wkt.loads(feat["wkt"])
centroid = poly.centroid
```
`shapely` turns the WKT text into a `Polygon` object you can ask questions:
`.area` (in square pixels here), `.centroid`, `.bounds`, `.is_valid`,
`.exterior.coords` (the corner points).

```python
rows.append({...})          # one dict per building
...
df = pd.DataFrame(rows, columns=COLUMNS)
```
**The most important pandas pattern in this repo.** Build a Python list of dicts (one per
row), then make the DataFrame *once* at the end. Appending to a DataFrame inside a loop is
very slow because pandas copies the whole table every time. Passing `columns=COLUMNS` fixes
the column order and means an empty folder still gives a DataFrame with the right columns
(the test `test_empty_folder_gives_empty_frame` checks this).

```python
"polygon": poly,
```
A DataFrame cell can hold any Python object. This column has dtype `object` and pandas
can't do maths on it, but you can use `.map()` to call a function on every polygon, e.g.
`df["polygon"].map(lambda p: p.is_valid)`.

```python
df["damage"] = pd.Categorical(df["damage"], categories=DAMAGE_CLASSES)
df["disaster"] = df["disaster"].astype("category")
```
A **categorical** column stores each distinct string once plus a small integer code per
row. Benefits: less memory, and the category list is fixed and ordered, so
`value_counts()`, `groupby` and plots always list classes in `DAMAGE_CLASSES` order and show
classes with zero rows. You can see the codes with `df["damage"].cat.codes`.

A catch you'll hit: **`groupby` on a categorical column** creates a group for *every*
category, even ones that don't appear. That's why later code passes `observed=True`
("only groups that actually exist") or `observed=False` ("include empty ones") explicitly.

```python
df["damage_level"] = df["damage_level"].astype("Int64")
```
Plain NumPy integers can't represent "missing". If a column of ints has one `None`,
pandas turns the whole column into floats (`0.0, 1.0, NaN`). `"Int64"` with a **capital
I** is pandas' *nullable* integer type: it keeps real ints and uses `<NA>` for missing.
Check for missing with `pd.isna(x)` or `df["damage_level"].isna()`, never `x == None`.

Using it:

```python
from src.labels import load_labels
df = load_labels(r"C:\Users\wikel\dev\xbd-data\test")
df.shape                      # (54862, 12)
df["damage"].value_counts()   # counts per class
df[df["disaster"] == "socal-fire"]            # boolean-mask filtering
df.groupby("disaster", observed=True)["area_px"].median()
```

**Known limitation:** the polygon comes from the *post* file. Pre files list the same
buildings by `uid`; whether their polygons are identical was not checked (see exercise 1).

---

## 4. `tests/test_labels.py` and `pyproject.toml`

`pyproject.toml` contains only pytest settings:

```toml
pythonpath = ["."]
```
Adds the repo root to Python's import path during tests, so `from src.labels import ...`
works. (`src/__init__.py` is an empty file that marks `src` as an importable package.)

The tests never touch the real dataset. They build **fake** label files in a temporary
folder, so they run in about a second and don't break when the data moves:

```python
@pytest.fixture
def fake_root(tmp_path):
```
A **fixture** is setup code pytest runs for you. Any test that takes an argument named
`fake_root` gets the result. `tmp_path` is a built-in fixture: a fresh empty folder per test.

```python
SQUARE = "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"   # area 100, centroid (5, 5)
```
Hand-made shapes whose area and centroid you can work out on paper, so the test can check
exact numbers.

```python
assert rect["area_px"] == pytest.approx(400)
```
Floats are rarely exactly equal (`0.1 + 0.2 != 0.3`). `pytest.approx` allows a tiny tolerance.

```python
levels = df.set_index("uid")["damage_level"]
assert levels["00000001-1"] == 3
```
`set_index("uid")` makes the uid column the row labels, so you can look rows up by uid
like a dict.

What's covered: one row per building, pre files ignored, paths built correctly, geometry
maths, dtypes (categorical, nullable Int64, `<NA>` for un-classified), empty folder,
missing folder, unknown label string, bad file name.

---

## 5. `src/colors.py`

One colour per damage class, used by every chart and overlay so "red" always means
major damage. The four real classes were checked with a colour-vision-deficiency
validator, so they stay distinguishable for red/green colourblind viewers.
`un-classified` is a deliberate neutral gray ("no answer"). Every chart still carries a
legend, because colour alone should never carry meaning.

---

## 6. `scripts/dataset_stats.py`

**Job:** load the DataFrame, then compute and chart class counts, class imbalance and
building sizes. It's the densest pandas file in the repo.

```python
import matplotlib
matplotlib.use("Agg")
```
Selects a backend that draws straight to image files and never opens a window. It must
run before `import matplotlib.pyplot`.

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```
`Path(__file__)` is this script's path; `.parents[1]` is two levels up (the repo root).
Putting it on `sys.path` lets a script inside `scripts/` do `from src.labels import ...`.
The `# noqa: E402` comments tell linters "yes, I know this import isn't at the top".

### `counts_table`
```python
table = pd.crosstab(df["disaster"], df["damage"])
```
`crosstab` counts how often each (row value, column value) pair occurs. Result: one row
per disaster, one column per damage class, cells = building counts. The same as
`df.groupby(["disaster", "damage"]).size().unstack()`, but shorter.

```python
table = table.reindex(columns=DAMAGE_CLASSES, fill_value=0)
```
`reindex` forces an exact column list and order; any class missing from the data becomes
a column of zeros.

```python
table["total"] = table.sum(axis=1)
```
**`axis`** confuses everyone. `axis=0` = go *down* the rows (one result per column).
`axis=1` = go *across* the columns (one result per row). Here: one total per disaster.

### `imbalance_table`
```python
real = df[df["damage"].isin(REAL_CLASSES)]
```
`isin` gives a True/False Series; using it inside `df[...]` keeps only the True rows. This
is **boolean indexing**, and you'll use it constantly.

```python
out["share"] = out["count"] / out["count"].sum()
```
**Vectorised arithmetic:** dividing a Series by a number divides every element. No loop.

```python
out["inverse_freq_weight"] = out["count"].sum() / (len(out) * out["count"])
```
A common starting point for class weights in a loss function: rare classes get bigger
weights. With perfectly balanced classes every weight is 1. It's reported for information
only; nothing is trained.

### `plot_by_disaster`
```python
data = data.div(data.sum(axis=1), axis=0)
```
Turns counts into row shares. `data.sum(axis=1)` is one total per disaster. `.div(...,
axis=0)` means "line the totals up with the **rows** and divide each row by its own
total". Plain `data / totals` would try to line the totals up with the *columns* and give
NaNs.

```python
data = data.iloc[::-1]
```
`iloc` selects by *position*; `[::-1]` reverses. `barh` draws the first row at the
bottom, so reversing puts the largest disaster at the top.

```python
left = np.zeros(len(data))
for cls in DAMAGE_CLASSES:
    ax.barh(..., vals, left=left, ...)
    left += vals
```
How a stacked bar works: each class's bar starts where the previous classes ended.
`left` is a NumPy array running total, and `+=` adds element-wise.

### `plot_size_hist`
```python
bins = np.logspace(np.log10(area.min()), np.log10(area.max()), 60)
```
Building areas range from ~16 to ~94,000 pixels. Evenly spaced bins would cram nearly
everything into the first bar. `np.logspace(a, b, n)` gives `n` bins evenly spaced on a
**log scale**, from `10**a` to `10**b`, which matches `ax.set_xscale("log")`.

```python
v = area.quantile(q)
```
The value below which a fraction `q` of the data falls (0.5 = median).

### `plot_size_by_class`
```python
groups = [df.loc[(df["damage"] == c) & (df["area_px"] > 0), "area_px"] for c in DAMAGE_CLASSES]
```
Combining conditions: use `&` (and), `|` (or), `~` (not), and **wrap each condition in
parentheses**. Python's `and`/`or` don't work on Series and raise
"truth value of a Series is ambiguous". `df.loc[rows, column]` selects rows by mask and a
column by name in one step.

`showfliers=False` hides the individual outlier dots so the boxes stay readable.

### `df_to_markdown`
```python
for idx, *row in df.itertuples():
```
`itertuples()` yields one tuple per row: the index first, then each value. `idx, *row`
unpacks the first item into `idx` and the rest into a list `row`. Why not `iterrows()`?
`iterrows` turns each row into a Series, and a Series has **one** dtype, so a row
containing ints and floats becomes all floats (`41427` prints as `41427.000`).
`itertuples` keeps each column's own type. (In general, avoid looping over rows at all when
a vectorised operation exists. Here the loop is only formatting text.)

### `main()`
```python
n_tiles = df.groupby(["disaster", "tile_id"], observed=True).ngroups
n_label_tiles = len(list((args.data_root / "labels").glob("*_post_disaster.json")))
```
Tiles with **zero buildings produce zero rows** in the DataFrame, so they can't be
counted from `df`. Counting label files gives the true number. In the test split that's
181 of 933 tiles with no buildings. That matters later: a model sees those empty tiles too.

```python
size_desc = df.groupby("damage", observed=False)["area_px"].describe(percentiles=[...])
size_desc.loc["all"] = df["area_px"].describe(percentiles=[...])
```
`groupby(...)[col].describe()` gives count/mean/std/min/percentiles/max per group, one row
per class. `.loc["all"] = ...` **adds a new row** labelled `all` (assigning to a label
that doesn't exist creates it).

```python
per_disaster = pd.DataFrame({...}).reindex(table.index)
```
When you build a DataFrame from a dict of Series, pandas **aligns them by index label**,
not by position. That's powerful (no mismatched rows, ever) but it re-sorts the result, so
`.reindex(table.index)` restores the "largest disaster first" order.

```python
real.idxmin(axis=1)
real.min(axis=1).replace(0, np.nan)
```
`idxmin(axis=1)` = the *column name* of the smallest value in each row (the rarest class).
Replacing 0 with `NaN` before dividing avoids a meaningless infinite ratio; the markdown
writer prints NaN as a blank cell.

```python
invalid = (~df["polygon"].map(lambda p: p.is_valid)).sum()
```
`.map(f)` calls `f` on every element; `~` flips True/False; `.sum()` on booleans counts the
Trues (True = 1).

---

## 7. `scripts/visualize.py`

**Job:** pick 10 random tiles and draw the pre and post images side by side, with each
building outlined in its post-disaster damage colour.

```python
tiles = df[["disaster", "tile_id"]].drop_duplicates().reset_index(drop=True)
chosen = tiles.iloc[rng.choice(len(tiles), size=..., replace=False)]
```
`df[[...]]` (double brackets) selects several columns and returns a DataFrame.
`drop_duplicates()` leaves one row per tile. `reset_index(drop=True)` renumbers rows
0..n-1 so positions and labels agree. Then we pick random positions with `iloc`. Because
`df` only contains tiles that have buildings, every chosen tile has something to draw.
Change `--seed` for a different set.

```python
for disaster, tile_id in chosen.itertuples(index=False):
    tile_df = df[(df["disaster"] == disaster) & (df["tile_id"] == tile_id)]
```
`index=False` leaves the index out of each tuple so it unpacks into exactly two names.
Filtering the full DataFrame 10 times is fine. For thousands of tiles you'd use
`for key, tile_df in df.groupby(["disaster", "tile_id"], observed=True):` instead, which
splits the data once.

```python
pre = np.asarray(Image.open(first["pre_image"]).convert("RGB"))
```
Image to NumPy array of shape `(height, width, 3)`, dtype `uint8` (0-255 per channel).

```python
xy = np.asarray(poly.exterior.coords)
patches.append(MplPolygon(xy, closed=True))
```
`exterior.coords` is the outline's corner points; as an array it has shape `(n_points, 2)`,
columns x and y. matplotlib's `Polygon` patch takes exactly that.

```python
ax.imshow(img)
...
ax.set_ylim(img.shape[0], 0)
```
`imshow` puts row 0 at the **top**, the same as the label's pixel coordinates (y grows
downward), so polygons land in the right place without any flipping. `set_ylim(height, 0)`
keeps that orientation explicit. `img.shape[0]` is the height (rows), `img.shape[1]` the
width: rows first again.

```python
ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor="white", linewidth=2.8, alpha=0.7))
ax.add_collection(PatchCollection(patches, facecolor="none", edgecolor=colors, linewidth=1.3))
```
A `PatchCollection` draws many shapes in one call; it's much faster than adding 1000
patches one by one. The first, thicker, white layer is a **halo** under the coloured
outline so that dark outlines (e.g. violet "destroyed") stay visible on dark burned ground.
The post image also gets a faint fill (`alpha=0.25`) so small buildings are easier to find.

Things to notice when you open `outputs/viz/`:
- Many tiles have **black areas**. Those are no-data regions outside the satellite
  swath, not damage. A model will see them as well.
- The same outline is drawn on both images, so you can check whether it lines up with
  the building **before** the disaster too. Off-nadir angle differences between the two
  captures can shift buildings by several pixels.

---

## 8. pandas / NumPy cheat sheet for this project

| you want | write |
|---|---|
| rows where a condition holds | `df[df["damage"] == "destroyed"]` |
| two conditions | `df[(cond1) & (cond2)]`, with parentheses required |
| value is one of several | `df["damage"].isin(["major-damage", "destroyed"])` |
| count per value | `df["damage"].value_counts()` (add `normalize=True` for shares) |
| stat per group | `df.groupby("disaster", observed=True)["area_px"].median()` |
| count table of two columns | `pd.crosstab(df["disaster"], df["damage"])` |
| several stats at once | `df.groupby("damage", observed=True)["area_px"].agg(["mean", "median", "count"])` |
| apply a function to each cell | `df["polygon"].map(lambda p: p.area)` |
| missing values | `df["damage_level"].isna()`, `pd.isna(x)` |
| select by position / by label | `df.iloc[0]`, `df.loc[label]` |
| row sums / column sums | `df.sum(axis=1)` / `df.sum(axis=0)` |
| divide each row by its own total | `df.div(df.sum(axis=1), axis=0)` |
| image as array | `np.asarray(Image.open(path))`, shape `(rows, cols[, channels])` |
| distinct values + counts in an array | `np.unique(arr, return_counts=True)` |
| reproducible randomness | `rng = np.random.default_rng(seed)`; `rng.choice(...)` |

Three traps worth remembering:

1. **`df[...]` returns a new DataFrame, but sometimes a view.** If you filter and then
   assign (`sub = df[mask]; sub["x"] = 1`), make it explicit with `sub = df[mask].copy()`.
2. **Integer columns with missing values** turn into floats unless you use `"Int64"`.
3. **Arrays are `[row, col]` = `[y, x]`; shapely and the labels are `(x, y)`.**

---

## 9. Five exercises before writing a model

Do these in a notebook (`notebooks/`) with `df = load_labels(...)`. Each one answers a
question that will change how you build or evaluate the model. Write down your answer and
the code that produced it.

**Exercise 1: Do pre and post polygons agree?**
`load_labels` takes each polygon from the post file only. Load the *pre* label files
yourself, build a DataFrame of `uid -> pre polygon`, and merge it with `df` on `uid`
(`pd.merge(df, pre_df, on="uid", how="outer", indicator=True)`). How many uids are in
only one of the two? For the rest, what fraction have identical WKT, and for the others,
what's the typical IoU (`a.intersection(b).area / a.union(b).area`)?
*Why it matters:* if they differ, a model that crops the pre image with the post polygon
is looking at slightly the wrong pixels.

**Exercise 2: Rebuild a target mask from polygons and compare.**
For one post tile, create `mask = np.zeros((1024, 1024), dtype=np.uint8)`, draw each
polygon onto it with `PIL.ImageDraw.Draw(...).polygon(...)` using `damage_level + 1` as
the fill (skip un-classified), and compare with the real `targets/*_target.png`:
`(mask == target).mean()` gives pixel agreement. Where do they disagree (plot
`mask != target`)?
*Why it matters:* you'll learn exactly how the official masks were made (overlap order,
edge pixels, un-classified handling), which decides whether you can trust them as
training targets.

**Exercise 3: Buildings per tile and the empty-tile problem.**
Using `groupby(["disaster", "tile_id"], observed=True).size()`, plot the distribution of
buildings per tile (log x-axis). Then add the 181 tiles with zero buildings (count them
from the label files). What share of all image area is covered by buildings? (Hint: sum
of `area_px` divided by `n_tiles * 1024 * 1024`.)
*Why it matters:* if ~95% of pixels are background, a pixel-wise model can reach 95%
"accuracy" by predicting nothing. You need to know the baseline your metric has to beat.

**Exercise 4: How much of the imbalance is really "which disaster"?**
Compute `pd.crosstab(df["disaster_type"], df["damage"], normalize="index")`. Then answer:
if a model only knew the disaster type and always guessed that type's most common class,
what accuracy would it get? And what **macro-F1**? (Compute per-class precision and recall
by hand with pandas, then average F1 over the four real classes.)
*Why it matters:* this is a "cheating" baseline. Any real model must beat it, and the gap
between accuracy and macro-F1 shows why accuracy is the wrong metric for this dataset.

**Exercise 5: Draft a leakage-safe split.**
Tiles from one disaster share weather, sensor, architecture and often neighbouring
streets. Write a function `split_by_disaster(df, val_disasters, test_disasters)` that
returns three DataFrames with **no disaster appearing in more than one**. For two
candidate choices of held-out disasters, print the class shares (`value_counts(normalize=True)`)
of each part side by side, and check that every part contains all four real classes.
Then compare against a naive random split of *buildings*: what fraction of val buildings
share a tile with a training building?
*Why it matters:* a random building-level split leaks near-duplicate context between
train and test and makes results look better than they will be on a new disaster.
