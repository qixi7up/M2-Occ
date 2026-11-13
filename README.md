# M²-Occ (M2Occ)

**Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.7-blue.svg)](docs/install.md)
[![Framework](https://img.shields.io/badge/PyTorch-1.10.1-orange.svg)](docs/install.md)

> **Status:** anonymous submission, under review. Code is released for reproducibility.
>
> **Authors:** Anonymous — the final author list (including any co-authors who contributed to writing) will be filled in upon acceptance.
> **Contact:** `<EMAIL>`  <!-- corresponding-author email; withhold during the blind review period -->

---

## Overview

Camera-based 3D semantic occupancy prediction usually assumes a full surround-view rig. In practice, cameras fail, get occluded, or are dropped to save bandwidth. M²-Occ addresses this setting: given **incomplete camera inputs**, it reconstructs a dense 3D semantic occupancy volume without retraining a separate model for every missing-view pattern.

This repository is a **derivative work of [SurroundOcc](https://github.com/weiyithu/SurroundOcc)** (ICCV 2023, Apache 2.0). The baseline architecture, occupancy ground-truth generation pipeline, and dataset tooling are inherited from SurroundOcc; M²-Occ adds two modules on top of its occupancy head:

| Module | Full name | What it does | Source |
|---|---|---|---|
| **MMR** | Multi-view Masked Reconstruction | Randomly masks surround views during training and supervises a lightweight decoder to reconstruct them; at test time, masks the specified missing view(s) and reconstructs them from neighbouring views before the main head consumes the features. | `surroundocc/dense_heads/lw.py` (`LwBlock`), `surroundocc/dense_heads/occ_head.py` (`apply_rvm_mmr`, `mmr_loss`) |
| **FMM** | Feature Memory Module | Maintains class-level semantic prototypes and uses them, gated by the predicted class distribution, to refine voxel features and keep semantics consistent when views are missing. | `surroundocc/dense_heads/occ_memory.py` (`OccFeaturesMemory`) |

The two modules are wired into `OccHead` and are independently switchable (see [Module switches](#module-switches)). Benchmark: nuScenes-based SurroundOcc, 200×200×16 voxels, ResNet-101 + FPN + FCOS3D-pretrained backbone, 6 epochs, AdamW with lr 2e-4.

---

## News

- **2026/11** — Initial public code release.

---

## Key Results

Numbers below are reported in the manuscript (Tables I / III / IV) and are obtained by adding the MMR and FMM modules on top of the SurroundOcc baseline.

| Setting | IoU | Note |
|---|---|---|
| Missing rear view | **+4.36 IoU** over the baseline | MMR + FMM recover the dropped view |
| 5 missing views | **13.57 → 20.24 IoU (+6.67)** | severe incomplete-input case |
| Full 6 views | no degradation | M²-Occ does not trade off clean-input accuracy |

---

## Method

### MMR — Multi-view Masked Reconstruction

MMR follows a two-stage masking / reconstruction scheme built around **RVM (Random View Masking)**:

- **Training.** For every forward pass, `masked_num` of the 6 camera views are sampled at random (the same permutation is shared across all FPN levels). Their features are zeroed out, `LwBlock` reconstructs the masked central image patches from the remaining views (mask ratio `mask_ratio=76%`), and an MSE reconstruction loss `loss_occ_mmr_rvm` supervises the decoder.
- **Testing.** The views listed in `incomplete_view` are masked instead, `LwBlock` reconstructs them from adjacent views, and the reconstructed features are written back before the main occupancy head runs. `incomplete_view` accepts an `int`, a list of indices, or `'random'`.

The neighbour relation for reconstruction is defined for the standard 6-camera rig.

### FMM — Feature Memory Module

FMM keeps one (or several) prototype feature vector(s) per semantic class and refines voxel features with a prediction-guided read-out. Prototypes are updated by an EMA of class-aggregated features (Eq. 6, λ = 0.1) and read out with a dot-product retrieval weighted by the predicted class probabilities. By default one prototype is kept per class (`num_feats_per_cls=1`); a multi-prototype variant (`num_feats_per_cls > 1`) implements Eq. 7–9 with an intra-class softmax temperature `temperature`, assigned by nearest cosine neighbour and updated by EMA.

---

## Repository Structure

```
M2Occ/
├── LICENSE                 # Apache License 2.0
├── NOTICE                  # attribution to SurroundOcc and upstream projects
├── assets/                 # baseline visualisations inherited from SurroundOcc
├── docs/                   # installation / dataset / run guides (from SurroundOcc)
├── extensions/             # chamfer_dist CUDA extension (must be compiled)
├── projects/
│   ├── configs/
│   │   ├── surroundocc/    # BASELINE configs (no MMR, no FMM)
│   │   └── m2occ/          # M²-Occ METHOD configs
│   └── mmdet3d_plugin/
│       └── surroundocc/
│           └── dense_heads/
│               ├── occ_head.py       # modified from SurroundOcc: MMR + FMM wiring
│               ├── occ_memory.py     # NEW: FMM
│               └── lw.py             # NEW: MMR decoder (LwBlock)
└── tools/                  # train / test / inference / visualisation scripts
```

---

## Installation

```bash
git clone https://github.com/qixi7up/M2-Occ
cd M2Occ
```

Environment used by the upstream codebase: **Python 3.7, PyTorch 1.10.1, CUDA 11.1, mmcv-full 1.4.0, mmdet 2.14.0, mmsegmentation 0.14.1, mmdet3d 0.17.1**. See [`docs/install.md`](docs/install.md) for the full step-by-step guide.

```bash
conda create -n m2occ python=3.7 -y
conda activate m2occ

pip install torch==1.10.1 torchvision==0.11.2
pip install mmcv-full==1.4.0 mmdet==2.14.0 mmsegmentation==0.14.1 mmdet3d==0.17.1
pip install timm numpy opencv-python scikit-learn open3d
pip install nuscenes-devkit pyquaternion shapely scikit-image scipy Pillow matplotlib pycocotools tqdm lyft-dataset-sdk PyYAML

# compile the chamfer_dist extension
cd extensions/chamfer_dist
python setup.py install
cd ../..
```

Runtime imports also include `mmdet3d` tooling from `projects/mmdet3d_plugin`. The plugin is loaded via `plugin_dir` in the configs, so no editable install of the plugin package is required.

> `docs/install.md`, `docs/data.md` and `docs/run.md` are inherited from SurroundOcc. Where they mention `projects/configs/surroundocc/*.py`, substitute the `m2occ/` configs in this README for M²-Occ experiments.

---

## Data Preparation

```
data/
├── nuscenes/                     # nuScenes V1.0 full dataset (samples/, sweeps/, maps/, v1.0-trainval/)
├── nuscenes_infos_train.pkl      # train info file
├── nuscenes_infos_val.pkl        # val info file
├── infos_inference.pkl           # only for visualisation; see below
└── <dense occupancy GT>          # 200x200x16, 0.5 m voxels
ckpts/
└── r101_dcn_fcos3d_pretrain.pth  # FCOS3D-pretrained ResNet-101-DCN backbone
```

1. Download **nuScenes V1.0 full** from [nuscenes.org](https://www.nuscenes.org/download) and unpack it into `data/nuscenes/`.
2. Download the SurroundOcc-generated info pickles and dense occupancy labels (links in [`docs/data.md`](docs/data.md)) and place them under `data/`.
3. Download the FCOS3D-pretrained ResNet-101-DCN checkpoint and place it at `ckpts/r101_dcn_fcos3d_pretrain.pth` (referenced by `load_from` in the training config).

---

## Training

```bash
./tools/dist_train.sh \
    projects/configs/m2occ/surroundocc_m2occ.py \
    8 \
    work_dirs/m2occ
```

Add `--cfg-options` / other arguments after the third positional argument; `tools/dist_train.sh` forwards `${@:4}` to `train.py`. Checkpoints are written every epoch (`checkpoint_config=dict(interval=1)`).

---

## Evaluation

```bash
./tools/dist_test.sh \
    projects/configs/m2occ/surroundocc_m2occ.py \
    <CHECKPOINT> \
    8
```

### Reproducing incomplete-camera settings

Missing-view evaluation is controlled by two config entries in `pts_bbox_head`, not by a separate config file. Edit `projects/configs/m2occ/surroundocc_m2occ.py` (or pass `--cfg-options`):

| Goal | `masked_num` | `incomplete_view` |
|---|---|---|
| drop the first camera | `1` | `0` |
| drop two specific cameras | `2` | `[0, 3]` |
| drop three randomly chosen cameras | `3` | `'random'` |
| drop five cameras (paper Table III) | `5` | `'random'` |

Constraints: `masked_num` must be in `[1, num_cam-1]`; a list/tuple must contain exactly `masked_num` unique indices in `[0, num_cam-1]`; an `int` is only valid when `masked_num=1`. Invalid values raise an informative error at test time.

---

## Visualisation

The inference config runs with `is_vis=True`, writes `.npy` predictions and `.ply` point clouds to `visual_dir/`, and expects `data/infos_inference.pkl`:

```bash
cp data/nuscenes_infos_val.pkl data/infos_inference.pkl
./tools/dist_inference.sh \
    projects/configs/m2occ/surroundocc_m2occ_inference.py \
    <CHECKPOINT> \
    8
```

Open the `.ply` files with MeshLab, or run `tools/visual.py` on the `.npy` files. Note that `is_vis=True` bypasses the normal evaluation path — switch it off for metric reporting.

---

## Module Switches

All switches live in `pts_bbox_head` of `projects/configs/m2occ/*.py`.

| Switch | Default | Meaning |
|---|---|---|
| `mmr` | `True` | enable Multi-view Masked Reconstruction |
| `use_memory` | `True` | enable the Feature Memory Module; when off, the head falls back to the original volume embedding |
| `masked_num` | `1` | number of views masked per step; valid range `[1, num_cam-1]` |
| `incomplete_view` | `0` | test-time missing views: `int` / list / `'random'`; ignored during training |
| `mask_ratio` | `76` | central-image patch mask ratio (%) for MMR reconstruction |
| `num_feats_per_cls` | `1` | FMM prototypes per class; `>1` enables the multi-prototype (Eq. 7–9) branch |
| `temperature` | `0.1` | softmax temperature τ (Eq. 8); only used when `num_feats_per_cls > 1` |

Setting both `mmr=False` and `use_memory=False` reduces the head to the SurroundOcc baseline behaviour.

---

## Citation

```bibtex
@misc{m2occ2026,
  title  = {M2Occ: Resilient 3D Semantic Occupancy Prediction for Autonomous Driving with Incomplete Camera Inputs},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

If you use the baseline, dataset pipeline, or dense occupancy labels, please also cite SurroundOcc:

```bibtex
@inproceedings{wei2023surroundocc,
  title     = {SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving},
  author    = {Wei, Yi and Zhao, Linqing and Zheng, Wenzhao and Zhu, Zheng and Zhou, Jie and Lu, Jiwen},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2023}
}
```

---

## Acknowledgements

This repository builds on [SurroundOcc](https://github.com/weiyithu/SurroundOcc) (Yi Wei et al., ICCV 2023), which is distributed under the Apache License 2.0. We also acknowledge [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [MonoScene](https://github.com/astra-vision/MonoScene), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and [TPVFormer](https://github.com/wzzheng/TPVFormer). Original copyright and attribution notices are retained in the corresponding source files; see [`NOTICE`](NOTICE).

---

## License

M²-Occ additions are © 2026 The M²-Occ Authors, released under the **Apache License 2.0** (see [`LICENSE`](LICENSE)). Portions derived from SurroundOcc remain © 2023 the original authors under the same license. Third-party components keep their respective licenses.
