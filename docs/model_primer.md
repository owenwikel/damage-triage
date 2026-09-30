# Model primer: concepts for the modelling phase

This is teaching material, not a design. Everything here explains ideas you'll need when
you write the model, dataset class and training loop yourself. The code snippets are
**toys** on random data, to show what a function does and what shapes come out. None of
them is meant to be copied into the project.

Conventions: PyTorch tensors for images are `(batch, channels, height, width)`, written
`(N, C, H, W)`. "Logits" are a model's raw scores before softmax.

Contents:

1. [How a CNN classifies an image](#1-how-a-cnn-classifies-an-image)
2. [Transfer learning with a pretrained backbone](#2-transfer-learning-with-a-pretrained-backbone)
3. [Using the pre and post images together](#3-using-the-pre-and-post-images-together)
4. [Class imbalance and the right metrics](#4-class-imbalance-and-the-right-metrics)
5. [Calibration](#5-calibration)
6. [Uncertainty](#6-uncertainty)
7. [GPU memory on an 8 GB card](#7-gpu-memory-on-an-8-gb-card)
8. [Eight exercises](#8-eight-exercises)

---

## 1. How a CNN classifies an image

A convolutional neural network (CNN) turns an image into class scores in stages:

1. **Convolution layers** slide small filters (e.g. 3x3) over the image. Each filter
   responds to a local pattern: an edge, a corner, a texture. The output of a layer is a
   stack of "feature maps", one per filter.
2. **Non-linearities** (usually ReLU: `max(0, x)`) between layers let the network
   combine patterns in non-linear ways.
3. **Downsampling** (stride-2 convolutions or pooling) halves the spatial size every few
   layers. Deeper layers therefore see a larger area of the input (their **receptive
   field** grows) and respond to bigger structures: roofs, debris fields, flood water.
4. **Global pooling** averages each final feature map to one number, giving a fixed-length
   feature vector however big the input was.
5. A **linear layer** maps that vector to one score (logit) per class. **Softmax** turns
   logits into probabilities that sum to 1.

Training adjusts all the filter weights so that the logit for the correct class goes up,
using a loss (usually **cross-entropy**, `-log p(correct class)`) and gradient descent.

Toy shapes:

```python
import torch
import torch.nn as nn

tiny_cnn = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),     # (N,16,64,64)
    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # (N,32,32,32)
    nn.AdaptiveAvgPool2d(1),                                   # (N,32,1,1)
    nn.Flatten(),                                              # (N,32)
    nn.Linear(32, 4),                                          # (N,4) logits
)
x = torch.randn(8, 3, 64, 64)       # 8 random "images"
logits = tiny_cnn(x)
probs = logits.softmax(dim=1)       # each row sums to 1
print(logits.shape, probs.sum(dim=1))
```

Two properties matter for this project:

- **Translation tolerance.** The same filter is applied everywhere, and pooling blurs
  exact positions, so a CNN is fairly tolerant of a pattern shifting a few pixels. That's
  relevant to the pre/post misalignment in `docs/data_quality.md`.
- **Scale sensitivity.** A CNN is *not* naturally scale-invariant. A 13 px building and a
  126 px building look like different things unless training shows both sizes. The crop
  choices in `docs/data_quality.md` section 7 decide what scales the network sees.

---

## 2. Transfer learning with a pretrained backbone

Training a CNN from scratch needs a lot of data. Instead, start from a network already
trained on a large dataset (typically ImageNet: 1.2M everyday photos, 1000 classes). Its
early layers have learned general-purpose filters (edges, textures) that transfer well
even to satellite images. The convolutional part is called the **backbone**; you replace
only the final linear layer (the **head**) with one sized for your classes.

> **Setup note:** pretrained backbones come from `torchvision`, which is **not installed**
> in the project venv yet (only `torch` 2.6.0+cu124 is). It has to match the torch build:
> for torch 2.6.0 that's torchvision 0.21.0 from the same CUDA 12.4 index
> (`https://download.pytorch.org/whl/cu124`). The snippets in sections 2 and 3 that import
> it won't run until then.

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

net = resnet18(weights=ResNet18_Weights.DEFAULT)   # downloads ImageNet weights once
print(net.fc)                                      # Linear(in_features=512, out_features=1000)
net.fc = nn.Linear(net.fc.in_features, 4)          # new head: 4 damage classes
```

Things to know:

- **Input normalisation.** Pretrained weights expect images scaled the way they were
  trained: pixel values in 0..1, then per-channel mean/std normalisation
  (`ResNet18_Weights.DEFAULT.transforms()` shows the exact numbers). Feeding raw 0..255
  values silently ruins transfer.
- **Freezing vs fine-tuning.** You can freeze the backbone (`p.requires_grad = False`)
  and train only the head: fast, less overfitting, but limited. Or fine-tune everything,
  usually with a smaller learning rate for the backbone than for the head. A common
  compromise is to train the head first, then unfreeze.
- **Domain gap.** Satellite images differ from ImageNet photos: seen from above, a
  different colour balance, and small objects. Pretraining still helps, but less than
  on everyday photos, and the early layers usually need some fine-tuning.
- **Backbone size.** ResNet-18/34 or EfficientNet-B0 are good first choices on an 8 GB
  card. Bigger backbones cost memory (section 7) and overfit more easily on a few
  hundred thousand small crops.

---

## 3. Using the pre and post images together

Damage is a *change*, so the model should see both images. Three standard ways:

### a) Channel stacking (6-channel input)

Concatenate pre (3 channels) and post (3 channels) into one 6-channel image and feed it to
a single CNN.

```python
pre = torch.randn(8, 3, 64, 64)
post = torch.randn(8, 3, 64, 64)
x = torch.cat([pre, post], dim=1)       # (8, 6, 64, 64)
```

A pretrained first layer expects 3 channels, so it has to be widened. A common trick is to
copy the pretrained filters for both halves (and halve them so the output scale stays
similar):

```python
old = net.conv1                                   # Conv2d(3, 64, 7, stride 2)
new = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
with torch.no_grad():
    new.weight[:, :3] = old.weight / 2
    new.weight[:, 3:] = old.weight / 2
net.conv1 = new
```

- **For:** simplest, and cheapest in memory and compute. The very first layer can compare
  pre and post pixels directly.
- **Against:** the comparison happens at pixel level, so it's the most sensitive to
  misalignment. The first layer's pretraining only partly applies.

### b) Siamese network (shared backbone, two passes)

Run the *same* backbone (same weights) on pre and post separately, then combine the two
feature vectors (concatenate, subtract, or both) and classify.

```python
backbone = nn.Sequential(*list(resnet18(weights=None).children())[:-1], nn.Flatten())  # (N,512)
f_pre, f_post = backbone(pre), backbone(post)
combined = torch.cat([f_pre, f_post, f_post - f_pre], dim=1)   # (N, 1536)
head = nn.Linear(1536, 4)
logits = head(combined)
```

- **For:** full use of 3-channel pretraining. Comparing high-level features instead of
  pixels tolerates small misalignment and lighting differences better. It's a natural
  fit for "what changed?".
- **Against:** two forward passes, so about 2x the compute and activation memory of
  stacking. Comparing only at the end loses fine spatial detail about *where* the change
  is. (Variants compare feature maps at several depths before pooling.)

### c) Difference image

Feed `post - pre` (or `|post - pre|`), sometimes alongside post, to a normal CNN.

- **For:** makes change explicit; a cheap input.
- **Against:** throws away what the building looks like (the difference of a roof and the
  same roof is ~0 whether it's a house or a warehouse). It's the most sensitive to
  misalignment and to lighting, season and sensor differences: all of those look like
  "change". The Mexico example in `docs/data_quality.md` (a blurrier post image) would
  produce a strong difference with no damage at all.

A reasonable way to think about it: stacking and difference images compare **early**
(pixels), siamese networks compare **late** (features). Early comparison is cheaper and
spatially precise, but brittle to misalignment. Late comparison is more robust but
costlier.

---

## 4. Class imbalance and the right metrics

About 74% of classified buildings are no-damage (train and test pooled; see
`docs/split_proposal.md`). A model that
always answers "no-damage" is 73% accurate and useless.

### Why accuracy misleads

Accuracy counts every building equally, so it's dominated by the majority class. What
you care about is whether damaged buildings are found, per class.

### Metrics

For each class c, from the counts of true positives (TP), false positives (FP) and false
negatives (FN) for that class:

- **Recall** (sensitivity) = TP / (TP + FN): of the buildings that truly are class c,
  what share did we find? For triage, recall on `destroyed` and `major-damage` is
  critical.
- **Precision** = TP / (TP + FP): of the buildings we *called* class c, what share really
  are?
- **F1** = 2 · precision · recall / (precision + recall).
- **Macro-F1** = the plain average of the per-class F1s. Each class counts equally
  regardless of size, so a model can't score well by ignoring rare classes.

```python
import numpy as np

y_true = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 3])
y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 3])   # mostly "no-damage"
f1s = []
for c in range(4):
    tp = np.sum((y_pred == c) & (y_true == c))
    fp = np.sum((y_pred == c) & (y_true != c))
    fn = np.sum((y_pred != c) & (y_true == c))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
print("accuracy", (y_true == y_pred).mean(), "macro-F1", np.mean(f1s))   # 0.70 vs 0.37
```

Always look at the **confusion matrix** too: for ordered classes, confusing minor with
major is a smaller error than confusing no-damage with destroyed.

### Two ways to counter imbalance in training

**Weighted loss:** multiply each example's loss by a weight for its class, so rare-class
mistakes cost more. `docs/data_quality.md` and `outputs/stats/summary.md` list
inverse-frequency weights as a starting point.

```python
counts = torch.tensor([117426., 14980., 14161., 13227.])
weights = counts.sum() / (len(counts) * counts)          # rare classes > 1, common < 1
loss_fn = nn.CrossEntropyLoss(weight=weights)
loss = loss_fn(torch.randn(8, 4), torch.randint(0, 4, (8,)))
```

**Oversampling:** show rare-class examples more often, e.g. with
`torch.utils.data.WeightedRandomSampler` giving each example a sampling probability
proportional to its class weight.

Trade-offs: both shift the model towards predicting rare classes more often, which raises
their recall and lowers their precision. Oversampling repeats the same few examples, so it
can overfit them (augmentation helps). Weighting keeps each epoch's data unchanged but
makes gradients noisier. Either way, **predicted probabilities no longer reflect real
class frequencies**, which matters for calibration (next section).

---

## 5. Calibration

A model is **calibrated** if, among all buildings it labels "destroyed with 80%
confidence", about 80% really are destroyed. Modern CNNs are often **overconfident**:
their 95% predictions are right noticeably less than 95% of the time. For triage, where a
responder might act on a confidence score, this matters as much as accuracy.

### Reliability diagram

Sort predictions into bins by confidence (the top class probability), e.g. 0-0.1,
0.1-0.2, ... For each bin, plot **accuracy** (share correct) against **mean confidence**.
A calibrated model lies on the diagonal; bars below the diagonal mean overconfidence.

### Expected calibration error (ECE)

The weighted average gap between confidence and accuracy across bins:

```python
def ece(probs, y_true, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            total += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return total

rng = np.random.default_rng(0)
logits = rng.normal(size=(1000, 4)) * 3
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
labels = rng.integers(0, 4, size=1000)          # random labels: badly overconfident
print(ece(probs, labels))
```

ECE has known weaknesses (it depends on the bin count, and a single number hides *which*
confidences are off), so look at the diagram too. With imbalanced classes, per-class
reliability diagrams are often more informative than one overall diagram.

### Temperature scaling

The simplest fix after training: divide all logits by a single number T > 0 before
softmax. T > 1 softens the probabilities (less confident); T < 1 sharpens them. T is chosen
to minimise the negative log-likelihood on the **validation** set (never the test set). It
doesn't change which class is predicted (the ranking of logits is unchanged), so accuracy
and F1 are untouched; only the confidences move.

```python
def nll(logits, labels, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    return -logp[np.arange(len(labels)), labels].mean()

Ts = np.linspace(0.5, 5, 46)
best_T = Ts[np.argmin([nll(logits, labels, T) for T in Ts])]
```

Note the interaction with section 4: a model trained with class weights or oversampling
is systematically biased towards rare classes, and one temperature can't fully undo a
per-class bias. Calibration is measured on data with the *real* class frequencies.

---

## 6. Uncertainty

Calibration asks whether confidences are honest on average. Uncertainty estimation asks
**how sure the model is about this particular building**, so that unsure cases can be
sent to a human.

Two sources of uncertainty are often distinguished:

- **Aleatoric:** ambiguity in the data itself (a blurry or cloud-covered building,
  or a borderline minor/major case). More data doesn't remove it.
- **Epistemic:** the model hasn't seen enough like this (a new disaster type, an unusual
  roof). More or broader data reduces it. This is the kind that matters most for a
  disaster the model wasn't trained on.

### MC dropout

Dropout randomly zeroes activations during training. **Monte Carlo dropout** keeps it
switched on at prediction time, runs the same input K times (e.g. 20), and looks at how
much the predictions vary.

```python
drop_net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 4))
drop_net.train()                        # keeps dropout active (only for this purpose)
x = torch.randn(5, 10)
with torch.no_grad():
    samples = torch.stack([drop_net(x).softmax(dim=1) for _ in range(20)])  # (20, 5, 4)
mean_probs = samples.mean(dim=0)
entropy = -(mean_probs * mean_probs.clamp_min(1e-12).log()).sum(dim=1)     # per input
```

- **For:** one model, no extra training.
- **Against:** K times the prediction cost. The uncertainty it gives is often too
  small, and it depends on where dropout layers sit (many pretrained backbones have none
  in the convolutional part). Careful: `.train()` also switches batch norm to training
  behaviour, so in real code you enable only the dropout layers.

### Deep ensembles

Train M models (e.g. 5) from different random seeds, average their probabilities, and use
their disagreement as uncertainty.

- **For:** usually the most reliable uncertainty in practice, and the averaged prediction
  is also more accurate.
- **Against:** M times the training and prediction cost. On one 8 GB GPU that means
  training them one after another.

Common scores: the **entropy** of the averaged probabilities (high = unsure overall),
**1 - max probability**, and for ensembles/MC samples the **mutual information**
(disagreement between members, closer to epistemic uncertainty).

### Risk-coverage curve

This shows what uncertainty is *for*. Sort buildings from most to least confident. Keep
only the most confident fraction (the **coverage**, e.g. 80% = the model decides 80% and
passes 20% to humans) and measure the error rate on what's kept (the **risk**). Repeat for
every coverage from 0 to 100%.

```python
conf = probs.max(axis=1)
errors = (probs.argmax(axis=1) != labels)
order = np.argsort(-conf)                        # most confident first
cum_risk = np.cumsum(errors[order]) / np.arange(1, len(order) + 1)
coverage = np.arange(1, len(order) + 1) / len(order)
# plot coverage (x) vs cum_risk (y)
```

A useful uncertainty score gives a curve that starts low (confident predictions are
rarely wrong) and rises towards the overall error rate at 100% coverage. A random score
gives a flat line. The area under the curve (**AURC**) summarises it: lower is better.
For triage this is the key plot: "if analysts review the 20% least certain buildings,
how many mistakes are left in the rest?". Use a per-class or macro metric for the risk if
the rare classes are what matter.

---

## 7. GPU memory on an 8 GB card

During training, GPU memory holds:

1. **Weights**: ResNet-18 has ~11.7M parameters, about 47 MB in float32. Small.
2. **Gradients**: the same size as the weights.
3. **Optimizer state**: Adam keeps two extra copies of the weights (~2x).
4. **Activations**: the output of every layer for every image in the batch, kept for the
   backward pass. **This is usually the biggest part, and it's what you control.**

Activation memory scales roughly with **batch size x height x width** (x 2 for a siamese
network, which processes two images). Doubling the crop side (64 -> 128 px) quadruples
the pixels, and so roughly quadruples activation memory. Halving the batch size halves it.

Levers, roughly in order of how often you'll reach for them:

- **Batch size.** The first thing to reduce on an out-of-memory error. Very small batches
  (< 16) make batch-norm statistics noisy.
- **Input size.** Crop size is a modelling decision (see `docs/data_quality.md` section 7),
  but it drives memory quadratically.
- **Mixed precision.** Many operations run in float16/bfloat16 instead of float32, which
  roughly halves activation memory and is faster on tensor cores (your RTX 2000 Ada has
  them). The pattern in PyTorch:

  ```python
  with torch.autocast(device_type="cuda", dtype=torch.float16):
      logits = net(x)            # forward pass in mixed precision
  ```

  With float16 you also need a gradient scaler (`torch.amp.GradScaler`) so tiny gradients
  don't underflow to zero; the PyTorch "Automatic Mixed Precision" recipe shows where it
  goes. bfloat16 doesn't need the scaler.
- **Gradient accumulation.** Run several small batches and update the weights once, to
  get the statistical effect of a bigger batch without its memory cost.
- **Measure, don't guess:**

  ```python
  torch.cuda.reset_peak_memory_stats()
  # ... run one forward + backward pass ...
  print(torch.cuda.max_memory_allocated() / 2**30, "GiB")
  ```

  Try one batch at your intended settings before starting a long run. Leave headroom:
  the desktop and the CUDA context themselves use some of the 8 GB.

Also watch the CPU side: decoding PNGs and cropping can be slower than the GPU.
`DataLoader(num_workers=...)` loads batches in parallel. On Windows, worker processes
require the `if __name__ == "__main__":` guard in your training script.

---

## 8. Eight exercises

In increasing difficulty; the last ones add up to the dataset class you'll write. Do them
in a notebook, and keep your answers.

1. **Shapes by hand.** For `tiny_cnn` in section 1 with a 128x128 input, predict the shape
   after each layer on paper, then check with `print`s (or hooks). How many parameters
   does each layer have? (Hint: a Conv2d has `out x in x k x k + out`.)

2. **Metrics on a baseline.** Using the pooled label DataFrame, build the "always predict
   the most common class of this disaster type" baseline from `LEARNING.md` exercise 4.
   Compute accuracy, per-class recall, macro-F1 and the confusion matrix with your own
   NumPy code (no sklearn). Write down the numbers any model has to beat.

3. **Calibration on synthetic data.** Generate 10,000 fake "predictions": draw random
   logits, then sample the true labels from `softmax(logits / 2)` (so the model is
   overconfident by a known factor). Plot the reliability diagram, compute ECE, fit a
   temperature by grid search, and check that you recover T ≈ 2 and that ECE drops.

4. **One building crop.** Write a function that takes one row of the label DataFrame, a
   crop size S and a padding P, and returns the pre and post crops as `(S, S, 3)` uint8
   NumPy arrays (use the definition in `docs/data_quality.md` section 7). Handle buildings
   at the tile edge, and test it on the three synthetic buildings in `tests/conftest.py`.

5. **Array to tensor, correctly.** Convert your crops to a float tensor of shape
   `(6, S, S)` or `(2, 3, S, S)`: scale to 0..1, apply the ImageNet mean/std per channel,
   and move channels first. Write a test that checks the shape, the dtype, and that
   un-normalising gives back the original pixels.

6. **Labels and splits as data.** Write a function that, given a split definition (lists of
   val/test disasters, like `docs/split_candidates.md`), returns three DataFrames (train,
   val, test) and asserts that no disaster or tile appears in two of them. Decide, and
   write down, what happens to un-classified buildings and to the one invalid polygon
   (`docs/data_quality.md` sections 1-2).

7. **Your dataset class.** Implement a `torch.utils.data.Dataset` with `__len__` and
   `__getitem__(i)` returning `(input_tensor, label)` for building i, using exercises 4-6.
   Keep image decoding out of `__init__` (open the PNG inside `__getitem__`, or cache per
   tile) and think about what happens when 100 buildings share one 1024x1024 tile.
   Test it on the synthetic fixture.

8. **Loader, sampler and memory.** Wrap your dataset in a `DataLoader` with a
   `WeightedRandomSampler` and check, over a few hundred batches, that the class mix
   matches what you intended. Then time how many batches per second it produces with
   `num_workers` = 0, 2, 4, and (on the GPU) measure peak memory for one forward and
   backward pass of a pretrained ResNet-18 at your chosen crop size and batch size, with
   and without `torch.autocast`.
