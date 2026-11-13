# Modified by The M²-Occ Authors (M2Occ), 2026. Derived from SurroundOcc
# (Wei et al., ICCV 2023, Apache License 2.0). See NOTICE and LICENSE.


# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import timm
from timm.models.vision_transformer import Block
from mmdet.models import HEADS
from mmcv.runner import force_fp32, auto_fp16
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from projects.mmdet3d_plugin.surroundocc.loss.loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.models.utils import build_transformer
from mmcv.cnn.utils.weight_init import constant_init
from .lw import LwBlock
from .occ_memory import OccFeaturesMemory
import os
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError:
    from itertools import  filterfalse as ifilterfalse

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):

    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def mmr_loss(x, x_ori):
    loss = (x - x_ori) ** 2
    loss = 0.05 * loss.sum() / (x.shape[2] * x.shape[3] * x.shape[4])
    return loss


def resolve_missing_views(incomplete_view, masked_num, num_cam, random_perm=None):
    """Resolve the test-time missing views into a list of camera indices.

    Args:
        incomplete_view: ``'random'`` to sample ``masked_num`` distinct views
            (``random_perm`` supplies the permutation); an ``int`` for a single
            missing view (only valid when ``masked_num == 1``); or a
            ``list``/``tuple`` of exactly ``masked_num`` indices.
        masked_num (int): number of views to mask at test time.
        num_cam (int): total number of camera views.
        random_perm (list, optional): a permutation of ``range(num_cam)``,
            required when ``incomplete_view == 'random'``.

    Returns:
        list[int]: the missing view indices, in the order to be masked.
    """
    if isinstance(incomplete_view, str):
        if incomplete_view.lower() not in ('random', 'rand'):
            raise ValueError(
                f"incomplete_view: unknown option {incomplete_view!r}; "
                "use 'random', an int, or a list of ints")
        if random_perm is None:
            raise ValueError("incomplete_view='random' requires random_perm")
        missing = list(random_perm[:masked_num])
    elif isinstance(incomplete_view, int) and not isinstance(incomplete_view, bool):
        missing = [incomplete_view]
    elif isinstance(incomplete_view, (list, tuple)):
        missing = list(incomplete_view)
    else:
        raise TypeError(
            f"incomplete_view must be 'random', an int, or a list/tuple, "
            f"got {type(incomplete_view).__name__}")

    if masked_num < 1 or masked_num >= num_cam:
        raise ValueError(
            f"masked_num must be in [1, {num_cam - 1}], got {masked_num}")
    if len(missing) != masked_num:
        raise ValueError(
            f"masked_num={masked_num} but incomplete_view specifies "
            f"{len(missing)} view(s): {missing}. Provide exactly {masked_num} "
            f"indices (e.g. a list), or use 'random'.")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in missing):
        raise TypeError(f"incomplete_view indices must be ints, got {missing}")
    if any(v < 0 or v >= num_cam for v in missing):
        raise ValueError(
            f"incomplete_view indices must be in [0, {num_cam - 1}], got {missing}")
    if len(set(missing)) != len(missing):
        raise ValueError(f"incomplete_view has duplicate indices: {missing}")
    return missing


def build_test_shuffle_indice(missing, num_cam):
    """Build the test-time view order: missing views first, then the rest."""
    others = [i for i in range(num_cam) if i not in missing]
    return list(missing) + others


@HEADS.register_module()
class OccHead(nn.Module):
    def __init__(self,
                 *args,
                 transformer_template=None,
                 position_level=0,
                 num_classes=17,
                 volume_h=200,
                 volume_w=200,
                 volume_z=16,
                 upsample_strides=[1, 2, 1, 2],
                 out_indices=[0, 2, 4, 6],
                 conv_input=None,
                 conv_output=None,
                 embed_dims=None,
                 img_channels=None,
                 use_semantic=True,
                 mmr=None,
                 num_cameras=6,
                 masked_num=1,        # MMR: number of views masked per step (random at train, fixed at test)
                 incomplete_view=0,   # MMR: test-time missing view(s): int (masked_num=1), list of masked_num ints, or 'random'
                 use_memory=True,      # FMM: class-level semantic prototype memory (on/off switch)
                 num_feats_per_cls=1,  # FMM: prototypes per class (1=single-proto; >1=multi-proto)
                 temperature=0.1,      # FMM: softmax temperature tau (Eq. 8); only used by multi-proto
                 mask_ratio=76,        # MMR: central-image mask ratio (%) used by the reconstruction decoder
                 **kwargs):
        super(OccHead, self).__init__()
        self.num_cameras = num_cameras
        self.conv_input = conv_input
        self.conv_output = conv_output
        self.position_level = position_level
        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z

        self.img_channels = img_channels
        self.use_semantic = use_semantic
        self.embed_dims = embed_dims
        self.mmr = mmr
        self.masked_num = masked_num
        self.incomplete_view = incomplete_view
        self.use_memory = use_memory
        self.num_feats_per_cls = num_feats_per_cls
        self.temperature = temperature
        self.mask_ratio = mask_ratio

        self.fpn_level = len(self.embed_dims)
        self.upsample_strides = upsample_strides
        self.out_indices = out_indices
        self.transformer_template = transformer_template

        # Initialize MMR (Multi-view Masked Reconstruction) modules
        if self.mmr:
            self._init_mmr_modules()
        self._init_layers()

# ===== Initialize FMM (feature memory) module (toggleable) =====
        if self.use_memory:
            self.occ_memory = nn.ModuleList()
            for i in range(self.fpn_level):
                self.occ_memory.append(
                    OccFeaturesMemory(
                        num_classes=self.num_classes,
                        feat_dim=self.embed_dims[i],
                        num_feats_per_cls=self.num_feats_per_cls,
                        momentum=0.1,
                        temperature=self.temperature,
                        verbose=False
                    )
                )

    def _init_mmr_modules(self):

        self.mae_embed_dim = 512
        # MMR decoder — keep dim consistent with the input
        self.decoder_embed_dim = self.mae_embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.mae_embed_dim))

        # Build decoder blocks matching the input dim
        decoders = []
        for _ in range(6):
            decoders.append(LwBlock(self.mae_embed_dim, num_heads=8, mlp_ratio=4.,
                  qkv_bias=True, norm_layer=nn.LayerNorm))
        self.decoder_blocks = nn.ModuleList(decoders)

        self.decoder_norm = nn.LayerNorm(self.mae_embed_dim)
        self.decoder_pred = nn.Linear(self.mae_embed_dim, self.mae_embed_dim, bias=True)

    def _init_layers(self):
        self.transformer = nn.ModuleList()
        for i in range(self.fpn_level):
            transformer = copy.deepcopy(self.transformer_template)

            transformer.embed_dims = transformer.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.num_points[i]

            transformer.encoder.transformerlayers.feedforward_channels = \
                self.transformer_template.encoder.transformerlayers.feedforward_channels[i]

            transformer.encoder.transformerlayers.embed_dims = \
                self.transformer_template.encoder.transformerlayers.embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].embed_dims[i]

            transformer.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = \
                self.transformer_template.encoder.transformerlayers.attn_cfgs[0].deformable_attention.embed_dims[i]

            transformer.encoder.num_layers = self.transformer_template.encoder.num_layers[i]

            transformer_i = build_transformer(transformer)
            self.transformer.append(transformer_i)




        self.deblocks = nn.ModuleList()
        upsample_strides = self.upsample_strides

        out_channels = self.conv_output
        in_channels = self.conv_input

        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg=dict(type='deconv3d', bias=False)
        conv_cfg=dict(type='Conv3d', bias=False)

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1:
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1)


            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))

            self.deblocks.append(deblock)


        self.occ = nn.ModuleList()
        for i in self.out_indices:
            if self.use_semantic:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)
            else:
                occ = build_conv_layer(
                    conv_cfg,
                    in_channels=out_channels[i],
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0)
                self.occ.append(occ)


        self.volume_embedding = nn.ModuleList()
        for i in range(self.fpn_level):
            self.volume_embedding.append(nn.Embedding(
                    self.volume_h[i] * self.volume_w[i] * self.volume_z[i], self.embed_dims[i]))


        self.transfer_conv = nn.ModuleList()
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        conv_cfg=dict(type='Conv2d', bias=True)
        for i in range(self.fpn_level):
            transfer_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=self.img_channels[i],
                    out_channels=self.embed_dims[i],
                    kernel_size=1,
                    stride=1)
            transfer_block = nn.Sequential(transfer_layer,
                    nn.ReLU(inplace=True))

            self.transfer_conv.append(transfer_block)


    def init_weights(self):
        for i in range(self.fpn_level):
            self.transformer[i].init_weights()

        for m in self.modules():
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def apply_rvm_mmr(self, x, masks, shuffle_indice):
        # Apply Random View Masking (RVM) + Multi-view Masked Reconstruction (MMR)
        #
        # Args:
        #   x (Tensor): input features [B, N, C, H, W]
        #   masks (Tensor): optional mask of valid regions
        #
        # Returns:
        #   Tensor: reconstructed features
        #   Tensor: original features used for the reconstruction loss

        B, N, C, H, W = x.shape
        # Cache positional embedding if not built or dim mismatch
        if not hasattr(self, 'mae_pos_embed') or self.mae_pos_embed.shape[1] != H * W:
            mae_pos_embed = get_2d_sincos_pos_embed(self.mae_embed_dim, (H, W), cls_token=False)
            self.mae_pos_embed = nn.Parameter(
                torch.from_numpy(mae_pos_embed).float().unsqueeze(0).to(x.device),
                requires_grad=False
            )
        batch_size, num_cams = x.size(0), x.size(1)

        x_ori = x.clone()
        x_new = torch.empty_like(x_ori, device=x_ori.device)

        # Random view masking (RVM)
        masked_num = self.masked_num


        mask_ind, unmask_ind = shuffle_indice[:masked_num], shuffle_indice[masked_num:]
        mask_inds = mask_ind
        unmask_inds = unmask_ind
        x[:, mask_inds] = 0

        # Adjacent-view relation table
        relation_dict = {
            '0': [1, 2], '1': [5, 0], '2': [0, 4],
            '3': [4, 5], '4': [2, 3], '5': [3, 1]
        }

        if self.mmr:
            # MMR reconstruction
            # mask_ratio: percentage of the central image width to mask (0~100).
            #   - Recommended 70~90 (default 76): balances reconstruction difficulty and feature quality.
            #   - Too low (<30): trivial task, model cannot learn view complementarity.
            #   - Too high (>95): near unconditional generation, training may diverge.
            #   - Ratio is scaled per feature level so the effective mask area stays consistent.
            mask_ratio = self.mask_ratio  # mask ratio (%)
            # Keep ratio per side (fraction of full width)
            keep_ratio_per_side = (100 - mask_ratio) / 100 / 2
            overlap = int(W * keep_ratio_per_side)

            # Reshape to [N, H*W, C]
            patches = x.permute(1, 0, 2, 3, 4).reshape(N, H * W, C)

            # Build patch indices
            left_index = torch.cat([torch.arange(0, overlap) + _ * W for _ in range(H)]).type(torch.long).to(x.device)
            right_index = torch.cat([torch.arange(W - overlap, W) + _ * W for _ in range(H)]).type(torch.long).to(
                x.device)
            mid_index = torch.cat([torch.arange(overlap, W - overlap) + _ * W for _ in range(H)]).type(torch.long).to(
                x.device)
            # Init reference token tensor
            x_ref_token = torch.zeros((1, H * W, self.mae_embed_dim), device=x.device)
            x_m = None
            refine_mask_inds = []

            for i in range(masked_num):
                mask_idx = mask_ind[i].item()
                if str(mask_idx) not in relation_dict:
                    continue

                refine_mask_inds.append(mask_idx)

                # Borrow features from the right adjacent view
                if relation_dict[str(mask_idx)][0] in unmask_ind:
                    x_ref_token[0, left_index] = patches[relation_dict[str(mask_idx)][0]][right_index]
                else:
                    x_ref_token[:, left_index] = x_ref_token[:, left_index] + self.mask_token

                # Borrow features from the left adjacent view
                if relation_dict[str(mask_idx)][1] in unmask_ind:
                    x_ref_token[0, right_index] = patches[relation_dict[str(mask_idx)][1]][left_index]
                else:
                    x_ref_token[:, right_index] = x_ref_token[:, right_index] + self.mask_token

                # Mask the central region
                x_ref_token[:, mid_index] = x_ref_token[:, mid_index] + self.mask_token

                # Reconstruct via the transformer decoder
                # Add positional embedding
                x_input = x_ref_token + self.mae_pos_embed
                slice_size = 1450  # adjustable slice size to bound memory
                slice_count = (H * W + slice_size - 1) // slice_size
                x_r_slices = []

                for s in range(slice_count):
                    start_idx = s * slice_size
                    end_idx = min((s + 1) * slice_size, H * W)

                    # Get slice
                    x_input_slice = x_input[:, start_idx:end_idx, :].contiguous()

                    # Process slice
                    x_r_slice = x_input_slice
                    for blk in self.decoder_blocks:
                        x_r_slice = blk(x_r_slice)

                    x_r_slices.append(x_r_slice)

                # Concat slices
                x_r = torch.cat(x_r_slices, dim=1)

                x_r = self.decoder_norm(x_r)
                x_r = self.decoder_pred(x_r)
                # Reshape back to [1, C, H, W]
                x_r = x_r.reshape(1, H, W, -1).permute(0, 3, 1, 2)  # [1, C, H, W]
                # Accumulate reconstruction results
                if x_m is None:
                    x_m = x_r
                else:
                    x_m = torch.cat((x_m,x_r),dim=0)
            refine_mask_inds = torch.tensor(refine_mask_inds, device=unmask_inds.device)
            unmask_views = x[:,unmask_inds]
            shuffle_indices = torch.cat((refine_mask_inds,unmask_inds),dim=0)
            concat_tokens = torch.cat((x_m, unmask_views.squeeze(0)),dim=0)
            # Reorder to original view order
            x_new = torch.empty_like(concat_tokens, device=concat_tokens.device)
            x_new[shuffle_indices] = concat_tokens
            x = x_new.unsqueeze(0)
        return mmr_loss(x, x_ori),x

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas):

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        # Apply RVM + MMR. shuffle_indice is generated dynamically here
        # (the runtime value of the "module parameter"):
        #   train -> torch.randperm(num_cam)  (random masking, with MMR auxiliary loss)
        #   test  -> missing views first, then the remaining views. The missing
        #            views come from incomplete_view: a single int (masked_num=1),
        #            a list of exactly masked_num ints, or 'random' to sample them.
        if self.mmr and self.training:
            mloss = 0.0
            num_cam = mlvl_feats[0].size(1)
            shuffle_indice = torch.randperm(num_cam, device=mlvl_feats[0].device)
            for i in range(len(mlvl_feats)):
                with torch.cuda.amp.autocast():
                    # Pass a clone: apply_rvm_mmr zeroes the masked views
                    # in place, so keep the original mlvl_feats[i] intact for
                    # the main head during training.
                    amloss, mmrxx = self.apply_rvm_mmr(
                        mlvl_feats[i].clone(), None, shuffle_indice)
                    mloss += amloss
        elif self.mmr:
            num_cam = mlvl_feats[0].size(1)
            device = mlvl_feats[0].device
            random_perm = None
            if isinstance(self.incomplete_view, str) and \
                    self.incomplete_view.lower() in ('random', 'rand'):
                random_perm = torch.randperm(num_cam, device=device).tolist()
            missing = resolve_missing_views(
                self.incomplete_view, self.masked_num, num_cam, random_perm)
            order = build_test_shuffle_indice(missing, num_cam)
            shuffle_indice = torch.tensor(order, device=device, dtype=torch.long)
            for i in range(len(mlvl_feats)):
                with torch.cuda.amp.autocast():
                    mloss, mlvl_feats[i] = self.apply_rvm_mmr(mlvl_feats[i], None, shuffle_indice)
        else:
            x_ori = None

        volume_embed = []
        for i in range(self.fpn_level):
            volume_queries = self.volume_embedding[i].weight.to(dtype)

            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]

            _, _, C, H, W = mlvl_feats[i].shape
            view_features = self.transfer_conv[i](mlvl_feats[i].reshape(bs*num_cam, C, H, W)).reshape(bs, num_cam, -1, H, W)

            volume_embed_i = self.transformer[i](
                [view_features],
                volume_queries,
                volume_h=volume_h,
                volume_w=volume_w,
                volume_z=volume_z,
                img_metas=img_metas
            )
            volume_embed.append(volume_embed_i)


        volume_embed_reshape = []
        for i in range(self.fpn_level):
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]

            volume_embed_reshape_i = volume_embed[i].reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)

            volume_embed_reshape.append(volume_embed_reshape_i)

        outputs = []
        result = volume_embed_reshape.pop()
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)

            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks) - 2:
                volume_embed_temp = volume_embed_reshape.pop()
                result = result + volume_embed_temp



        occ_preds = []
        for i in range(len(outputs)):
            occ_pred = self.occ[i](outputs[i])
            occ_preds.append(occ_pred)

# ========== Memory Query (Semantic Enhancement, FMM) ==========
        if self.use_memory:
            volume_embed_enhanced = []
            for i in range(self.fpn_level):
                volume_embed_enhanced_i = self.occ_memory[i](
                    feats=volume_embed[i],
                    occ_pred=occ_preds[2-i].detach()  # prediction-guided memory
                )
                volume_embed_enhanced.append(volume_embed_enhanced_i)
        else:
            # FMM disabled: use original volume_embed
            volume_embed_enhanced = volume_embed

        volume_embed_reshape = []
        for i in range(self.fpn_level):
            volume_h = self.volume_h[i]
            volume_w = self.volume_w[i]
            volume_z = self.volume_z[i]

            volume_embed_reshape_i = volume_embed_enhanced[i].reshape(bs, volume_z, volume_h, volume_w, -1).permute(0, 4, 3, 2, 1)

            volume_embed_reshape.append(volume_embed_reshape_i)

        outputs = []
        result = volume_embed_reshape.pop()
        for i in range(len(self.deblocks)):
            result = self.deblocks[i](result)

            if i in self.out_indices:
                outputs.append(result)
            elif i < len(self.deblocks) - 2:
                volume_embed_temp = volume_embed_reshape.pop()
                result = result + volume_embed_temp



        occ_preds = []
        for i in range(len(outputs)):
            occ_pred = self.occ[i](outputs[i])
            occ_preds.append(occ_pred)

        if self.mmr:
            outs = {
                'volume_embed': volume_embed,
                'occ_preds': occ_preds,
                'mmr_loss': mloss
            }
        else :
            outs = {
                'volume_embed': volume_embed,
                'occ_preds': occ_preds,
            }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_occ,
             preds_dicts,
             img_metas):


        if not self.use_semantic:
            loss_dict = {}
            for i in range(len(preds_dicts['occ_preds'])):

                pred = preds_dicts['occ_preds'][i][:, 0]

                ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)

                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)

                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(), semantic=False))

                loss_occ_i =  loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))

                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        else:
            pred = preds_dicts['occ_preds']

            criterion = nn.CrossEntropyLoss(
                ignore_index=255, reduction="mean"
            )

            loss_dict = {}

            for i in range(len(preds_dicts['occ_preds'])):

                pred = preds_dicts['occ_preds'][i]
                ratio = 2**(len(preds_dicts['occ_preds']) - 1 - i)

                gt = multiscale_supervision(gt_occ.clone(), ratio, preds_dicts['occ_preds'][i].shape)

                loss_occ_i = (criterion(pred, gt.long()) + sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))

                loss_occ_i = loss_occ_i * ((0.5)**(len(preds_dicts['occ_preds']) - 1 -i))

                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

# ===== FMM memory update (toggleable) =====
            if self.use_memory:
                volume_embed = preds_dicts['volume_embed']
                volume_ratios = [2, 4, 8]  # multiscale ratios: 100×100×8 / 50×50×4 / 25×25×2
                volume_h = [100, 50, 25]  # 3D height levels
                volume_w = [100, 50, 25]  # 3D width levels
                volume_z = [8, 4, 2]      # 3D depth levels

                for i in range(len(volume_embed)):
                    gt_i = multiscale_supervision(
                        gt_occ=gt_occ.clone(),
                        ratio=volume_ratios[i],
                        gt_shape=(
                            1,
                            17,
                            volume_w[i],
                            volume_h[i],
                            volume_z[i],
                        )
                    )

                    self.occ_memory[i].update(
                        feats=volume_embed[i].detach(),
                        gt_occ=gt_i.long()
                    )

        if self.mmr:
            loss_dict['loss_occ_mmr_rvm'] = preds_dicts['mmr_loss']

        return loss_dict
