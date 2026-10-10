"""Siamese model whose classifier sees ONLY the difference between post and pre.

Backup candidate of docs/pre_image_plan.md (candidate B). It lives in its own file so
that the two existing architectures in src/model.py stay exactly as they are; pick it
with ``build_model("siamese_diff")``.

What is different from ``SiameseNet``
-------------------------------------
``SiameseNet`` gives its classifier three things, glued together:

    [pre features, post features, post features - pre features]      3 x 512 numbers

The middle block lets the classifier decide from the post image alone, and the base
model turned out to do just that (docs/gap_diagnosis.md). Here the classifier gets only

    post features - pre features                                     512 numbers

so there is nothing left to read the post image from on its own. Two consequences:

  * Same image twice -> the difference is exactly zero -> the classifier's output is a
    constant (its biases), whatever the building looks like.
  * Replace the pre image by another building's -> the difference changes completely.
    The answer cannot stay the same by accident.

What it gives up: the absolute look of the post image. Rubble looks like rubble without
any "before", and this model cannot use that directly; it only sees how the summary of
the crop moved. Whether that trade is worth it is what the experiment measures.
"""

import torch
from torch import nn

from src.model import FEATURES, NUM_CLASSES, _backbone


class SiameseDiffNet(nn.Module):
    """Shared ResNet-18 -> (post features - pre features) -> MLP -> 4 scores."""

    def __init__(self, pretrained=True, dropout=0.3, hidden=256):
        super().__init__()
        self.backbone = _backbone(pretrained)
        # Drop ImageNet's classifier: the backbone now returns the 512 pooled features.
        self.backbone.fc = nn.Identity()
        # The same small head as SiameseNet, except that its first layer takes 512
        # inputs (the difference) instead of 1536.
        self.head = nn.Sequential(
            nn.Linear(FEATURES, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, NUM_CLASSES),
        )

    def forward(self, pre, post):
        # Both images through the SAME backbone in one call (see SiameseNet.forward):
        # [B, 3, H, W] + [B, 3, H, W] -> [2B, 3, H, W] -> [2B, 512].
        n = pre.shape[0]
        feats = self.backbone(torch.cat([pre, post], dim=0))
        pre_feat, post_feat = feats[:n], feats[n:]
        # The sign is kept (post minus pre, not the absolute value): "got darker" and
        # "got brighter" are different events.
        return self.head(post_feat - pre_feat)
