# Copyright (c) 2026 The M²-Occ Authors. All rights reserved.
#
# This file is part of M2Occ, a derivative work based on SurroundOcc
# (Wei et al., ICCV 2023, Apache License 2.0).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F


class OccFeaturesMemory(nn.Module):
    """Class-level semantic prototype memory (FMM).

    A memory bank ``M`` stores semantic prototypes and refines ambiguous
    voxel features by injecting a prediction-guided residual. Two strategies
    are provided (paper Sec. III-C, Fig. 4, Eqs. 6-9):

    * Single-Proto (``num_feats_per_cls == 1``): one global centroid per
      class, updated by a momentum moving average (Eq. 6). This is the
      setting used by the released weights.
    * Multi-Proto (``num_feats_per_cls > 1``): ``Np`` sub-prototypes per
      class to model intra-class variance. Retrieval follows Eqs. 7-9:
      cosine similarity -> temperature-scaled softmax over sub-prototypes
      -> aggregation gated by the predicted class probability P(k).
      Update assigns each voxel to its nearest sub-prototype (Fig. 4b).

    Args:
        num_classes (int): number of semantic classes K.
        feat_dim (int): feature channel dimension D.
        num_feats_per_cls (int): prototypes per class Np (1 = single-proto).
        momentum (float): EMA update rate ``lambda`` (Eq. 6), default 0.1.
        temperature (float): softmax temperature ``tau`` (Eq. 8). The paper
            does not fix a value; 0.1 is a common choice. Only used by the
            multi-proto strategy.
        verbose (bool): print prototype-update logs.
    """

    def __init__(self, num_classes, feat_dim, num_feats_per_cls=1,
                 momentum=0.1, temperature=0.1, verbose=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.num_feats_per_cls = num_feats_per_cls
        self.momentum = momentum
        self.temperature = temperature
        self.verbose = verbose
        self.update_count = 0  # number of update() calls

        if num_feats_per_cls == 1:
            # Single prototype per class: zero-init is fine because the EMA
            # update moves it from the very first batch.
            self.memory = nn.Parameter(
                torch.zeros(num_classes, 1, feat_dim), requires_grad=False)
        else:
            # Multi-prototype: small random init keeps the K sub-prototypes
            # of a class distinct, so cosine assignment does not collapse
            # onto a single slot.
            self.memory = nn.Parameter(
                torch.randn(num_classes, num_feats_per_cls, feat_dim) * 0.01,
                requires_grad=False)

    def forward(self, feats, occ_pred=None):
        # feats: (B, N, D) voxel features; occ_pred: (B, K, W, H, Z)
        if occ_pred is None:
            return feats

        B, N, D = feats.shape

        # Predicted class probability P(k) (Eq. 9 gate).
        weight = occ_pred.permute(0, 2, 3, 4, 1).reshape(-1, self.num_classes)
        weight = F.softmax(weight, dim=-1)  # (B*N, K)

        if self.num_feats_per_cls == 1:
            # Single-Proto (Eq. 6 / Eq. 9 with Np=1): blended by P(k).
            mem = self.memory[:, 0, :]          # (K, D)
            selected = weight @ mem             # (B*N, D)
        else:
            # ---- Multi-Proto, Eqs. 7-9 ----
            x = feats.reshape(-1, D)            # (B*N, D)
            mem = self.memory                   # (K, Np, D)
            K, Np = self.num_classes, self.num_feats_per_cls

            # Eq. 7: cosine similarity s_{k,j} = x . m_{k,j} / (|x||m_{k,j}|)
            x_norm = F.normalize(x, dim=-1)                    # (B*N, D)
            m_norm = F.normalize(mem.reshape(-1, D), dim=-1)  # (K*Np, D)
            s = torch.matmul(x_norm, m_norm.t())              # (B*N, K*Np)
            s = s.reshape(-1, K, Np)                          # (B*N, K, Np)

            # Eq. 8: softmax over sub-prototypes j within each class k.
            alpha = F.softmax(s / self.temperature, dim=-1)    # (B*N, K, Np)

            # Eq. 9: sum_j alpha_{k,j} m_{k,j}, then gate by P(k) and sum over k.
            cls_proto = torch.einsum('bkj,kjd->bkd', alpha, mem)  # (B*N, K, D)
            selected = torch.einsum('bk,bkd->bd', weight, cls_proto)  # (B*N, D)

        selected = selected.view(B, N, D)
        return feats + selected

    @torch.no_grad()
    def update(self, feats, gt_occ):
        # feats: (B, N, D); gt_occ: (B, Z, H, W)
        B, N, D = feats.shape
        gt_flat = gt_occ.view(-1)
        feats_flat = feats.view(-1, D)

        self.update_count += 1

        for cls in gt_flat.unique():
            if cls < 0 or cls >= self.num_classes:
                continue
            cls_feats = feats_flat[gt_flat == cls]
            if cls_feats.numel() == 0:
                continue

            if self.num_feats_per_cls == 1:
                # Eq. 6: momentum EMA of the class mean feature.
                mean_feat = cls_feats.mean(0)
                self.memory[cls, 0] = (
                    (1 - self.momentum) * self.memory[cls, 0]
                    + self.momentum * mean_feat
                )
            else:
                # Fig. 4b / Eq. 6: assign each voxel to its nearest
                # sub-prototype by cosine similarity, then EMA per slot.
                x_norm = F.normalize(cls_feats, dim=-1)        # (M, D)
                m_norm = F.normalize(self.memory[cls], dim=-1)  # (Np, D)
                s = torch.matmul(x_norm, m_norm.t())           # (M, Np)
                assign = s.argmax(dim=1)                       # (M,)
                for j in range(self.num_feats_per_cls):
                    mj = assign == j
                    if not mj.any():
                        continue
                    mean_feat = cls_feats[mj].mean(0)
                    self.memory.data[cls, j] = (
                        (1 - self.momentum) * self.memory.data[cls, j]
                        + self.momentum * mean_feat
                    )

            if self.verbose:
                print(f"[Memory Update] class {cls.item()}: "
                      f"mean feat (first 5 dims) {cls_feats.mean(0)[:5]}")
