"""Two ways to turn a (pre, post) pair of crops into four damage-class scores.

Both start from the same pretrained network, ResNet-18, and differ in WHERE the two
images meet. Pick one with ``build_model("siamese")`` or ``build_model("stack")``.

Transfer learning in one paragraph
----------------------------------
ResNet-18 was trained on 1.3 million everyday photos (ImageNet). Its early layers learned
general things: edges, corners, textures, colour blobs. Those are useful for roofs and
rubble too, so instead of learning them again from our ~135,000 crops we start from the
pretrained weights and *fine-tune*: keep training all layers, at a small learning rate,
on our task. Only the very last layer (ImageNet's 1000-way classifier) is thrown away
and replaced by one with 4 outputs.

Anatomy of ResNet-18 (for a 128x128 input)
    conv1 + pooling          3 channels -> 64,  spatial size 128 -> 32
    4 groups of conv blocks  64 -> 512 channels, spatial size 32 -> 4
    global average pool      [512, 4, 4] -> 512 numbers: "what is in this image",
                             with "where" averaged away
    fc                       512 -> number of classes

Option a) "siamese": late fusion
--------------------------------
The SAME network (one set of weights: "siamese twins") turns pre into 512 numbers and
post into 512 numbers. The classifier then sees
    [pre_feat, post_feat, post_feat - pre_feat]         (3 * 512 = 1536 numbers)
The difference is given explicitly because "what changed" is the question; a small MLP
could learn to subtract, but handing it over saves it the effort. The raw features stay
in as well, because the same change can mean different things for different buildings.

  + Shared weights mean the two feature vectors live in the same "language", so their
    difference is meaningful. Two separate networks would have no reason to agree.
  + Tolerant to misalignment. The images only meet AFTER global average pooling, which
    has discarded exact position. Whether the post image is shifted 3 px relative to the
    pre image (docs/data_quality.md section 6: the shift is unmeasured, plausibly a few
    pixels) barely changes a pooled feature.
  + ImageNet weights are used exactly as trained (3-channel input).
  - Two passes through the backbone per building: about twice the compute and memory.
  - Because position is gone before the comparison, it cannot easily notice a small
    local change ("this corner of the roof is missing"); it compares summaries.

Option b) "stack": early fusion
-------------------------------
Pre and post are stacked into one 6-channel image and go through ONE network. The very
first convolution sees both images at the same pixel location.

  + Can learn local change detectors from the first layer on (a filter that fires where
    post differs from pre at the same spot).
  + One backbone pass: roughly half the compute of the siamese model.
  - Sensitive to misalignment. A filter that compares pre and post at the same pixel
    will fire along every edge when the images are offset by a few pixels, even for an
    undamaged building. Later layers and pooling blur small offsets, so this is a
    matter of degree, not a fatal flaw; which effect wins is an empirical question, and
    that is why both are trained and compared (docs/results.md).
  - The pretrained first layer expects 3 channels, so it has to be adapted (see
    ``_six_channel_conv``); the network starts a little further from where ImageNet
    left it.
"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

NUM_CLASSES = 4
FEATURES = 512   # size of ResNet-18's feature vector after global average pooling


def _backbone(pretrained):
    """ResNet-18, with ImageNet weights if ``pretrained`` (downloaded once, then cached).

    Tests pass pretrained=False so they need no download.
    """
    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)


class SiameseNet(nn.Module):
    """Late fusion: shared ResNet-18 -> [pre, post, post - pre] -> MLP -> 4 scores."""

    def __init__(self, pretrained=True, dropout=0.3, hidden=256):
        super().__init__()
        self.backbone = _backbone(pretrained)
        # Remove ImageNet's 1000-way classifier. nn.Identity() passes its input through
        # unchanged, so the backbone now returns the 512 pooled features.
        self.backbone.fc = nn.Identity()
        # The "head": a small MLP (multi-layer perceptron) trained from scratch.
        #   Linear  mixes the 1536 inputs into 256 numbers
        #   ReLU    without a non-linearity, two Linear layers collapse into one
        #   Dropout randomly zeroes 30% of those numbers during training, so the head
        #           cannot rely on any single feature; a standard guard against
        #           overfitting. We also reuse it at test time for MC dropout
        #           (src/uncertainty.py).
        #   Linear  256 -> 4 scores ("logits"), one per damage class
        self.head = nn.Sequential(
            nn.Linear(3 * FEATURES, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, pre, post):
        # Run both images through the backbone in ONE call by stacking them along the
        # batch axis: [B, 3, H, W] + [B, 3, H, W] -> [2B, 3, H, W]. Same result as two
        # calls, but the GPU does it in one go, and the BatchNorm layers see pre and
        # post statistics together instead of alternately.
        n = pre.shape[0]
        feats = self.backbone(torch.cat([pre, post], dim=0))
        pre_feat, post_feat = feats[:n], feats[n:]
        fused = torch.cat([pre_feat, post_feat, post_feat - pre_feat], dim=1)
        return self.head(fused)


def _six_channel_conv(conv):
    """Make a 6-input-channel copy of ResNet's first conv layer, keeping what it learned.

    The pretrained layer has weights of shape [64, 3, 7, 7]: 64 filters, each looking at
    3 colour channels. We need [64, 6, 7, 7]. Starting the new layer from random numbers
    would throw away the most transferable part of the network (edge and colour
    detectors), so the pretrained filters are copied into both halves: channels 0-2
    (pre) and 3-5 (post).

    The copies are multiplied by 0.5. Each filter now adds up twice as many inputs, so
    without the halving its output would be about twice as large as anything the next
    layers saw during ImageNet training. With it, the layer initially computes the
    average of "filter applied to pre" and "filter applied to post": the same scale as
    before, and a sensible starting point from which training can make the two halves
    differ.
    """
    new = nn.Conv2d(6, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
                    padding=conv.padding, bias=False)
    with torch.no_grad():   # plain assignment, not something to compute gradients for
        new.weight[:, :3] = conv.weight * 0.5
        new.weight[:, 3:] = conv.weight * 0.5
    return new


class StackNet(nn.Module):
    """Early fusion: pre and post stacked to 6 channels -> one ResNet-18 -> 4 scores."""

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = _backbone(pretrained)
        self.backbone.conv1 = _six_channel_conv(self.backbone.conv1)
        # Same idea as the siamese head, but simpler: there is only one 512-vector, so a
        # single Linear layer is the usual choice. Dropout sits in front of it so that
        # both architectures have dropout (needed for the MC-dropout comparison).
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(FEATURES, NUM_CLASSES))

    def forward(self, pre, post):
        # dim=1 is the channel axis: [B, 3, H, W] + [B, 3, H, W] -> [B, 6, H, W]
        return self.backbone(torch.cat([pre, post], dim=1))


def build_model(arch, pretrained=True, dropout=0.3):
    """Return the model for ``arch`` ("siamese" or "stack")."""
    if arch == "siamese":
        return SiameseNet(pretrained, dropout)
    if arch == "stack":
        return StackNet(pretrained, dropout)
    if arch == "siamese_diff":
        # Experimental (docs/pre_image_plan.md, candidate B). Imported here, not at the
        # top: src/model_diff.py itself imports from this file.
        from src.model_diff import SiameseDiffNet
        return SiameseDiffNet(pretrained, dropout)
    raise ValueError(f"Unknown arch {arch!r}; choose 'siamese' or 'stack'")


def count_parameters(model):
    """Number of trainable weights; handy for comparing model sizes."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
