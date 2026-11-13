# Modified by The M²-Occ Authors (M2Occ), 2026. Derived from SurroundOcc
# (Wei et al., ICCV 2023, Apache License 2.0). See NOTICE and LICENSE.

# Base config inheritance (avoid duplicating settings)
_base_ = [
    '../datasets/custom_nus-3d.py',  # dataset loading config
    '../_base_/default_runtime.py'   # default runtime config (logging, hooks, etc.)
]
# Plugin system config
plugin = True  # enable plugin mode
plugin_dir = 'projects/mmdet3d_plugin/'  # plugin directory

# Point cloud range (meters): [x_min, y_min, z_min, x_max, y_max, z_max]
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
occ_size = [200, 200, 16]  # occupancy grid size (length, width, height)
use_semantic = True  # whether to use semantic segmentation

# Image normalization (Caffe-style preprocessing)
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],  # BGR mean (OpenCV channel order)
    std=[1.0, 1.0, 1.0],               # std
    to_rgb=False)                       # keep BGR (no RGB conversion)
# NuScenes 17 classes (dynamic objects + static scene elements)
class_names =  ['barrier','bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                'other_flat', 'sidewalk', 'terrain', 'manmade','vegetation']
# Input modality (camera-only)
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
# Network dimension parameters
_dim_ = [128, 256, 512]          # feature dims (multi-scale)
_ffn_dim_ = [256, 512, 1024]      # FFN dims
volume_h_ = [100, 50, 25]         # 3D volume height (downsampling per level)
volume_w_ = [100, 50, 25]         # 3D volume width
volume_z_ = [8, 4, 2]             # 3D volume depth
_num_points_ = [2, 4, 8]          # deformable attention sampling points
_num_layers_ = [1, 3, 6]          # encoder layers
# Main model config
model = dict(
    type='SurroundOcc',            # surround occupancy prediction
    use_grid_mask=True,            # use grid mask augmentation
    use_semantic=use_semantic,     # inherit global semantic switch
    is_vis=True,                   # enable visualization
    # Image backbone (ResNet101 + DCNv2)
    img_backbone=dict(
       type='ResNet',
       depth=101,                  # network depth
       num_stages=4,               # number of stages
       out_indices=(1,2,3),        # output feature levels (FPN input)
       frozen_stages=1,            # freeze first stage (fine-tuning strategy)
       norm_cfg=dict(type='BN2d', requires_grad=False),
       norm_eval=True,             # freeze BN statistics
       style='caffe',              # Caffe style (BGR input)
       #with_cp=True, # using checkpoint to save GPU memory
       dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),  # original DCNv2 prints a log when loading state_dict
       stage_with_dcn=(False, False, True, True)),  # enable DCN at stages 3/4
    # Feature pyramid network (FPN)
    img_neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],  # input channels (ResNet stage outputs)
        out_channels=512,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=3,
        relu_before_extra_convs=True),
    # 3D occupancy prediction head
    pts_bbox_head=dict(
        type='OccHead',
        volume_h=volume_h_,
        volume_w=volume_w_,
        volume_z=volume_z_,
        num_query=900,
        num_classes=17,
        conv_input=[_dim_[2], 256, _dim_[1], 128, _dim_[0], 64, 64],
        conv_output=[256, _dim_[1], 128, _dim_[0], 64, 64, 32],
        out_indices=[0, 2, 4, 6],
        upsample_strides=[1,2,1,2,1,2,1],
        embed_dims=_dim_,
        img_channels=[512, 512, 512],
        use_semantic=use_semantic,
        # Perception transformer config
        transformer_template=dict(
            type='PerceptionTransformer',
            embed_dims=_dim_,
            encoder=dict(
                type='OccEncoder',
               num_layers=_num_layers_,
                pc_range=point_cloud_range,
                return_intermediate=False,
                transformerlayers=dict(
                    type='OccLayer',
                    # Spatial cross-attention config
                    attn_cfgs=[
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=_num_points_,
                                num_levels=1),
                            embed_dims=_dim_,
                        )
                    ],
                   feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    embed_dims=_dim_,
                    conv_num=2,
                    operation_order=('cross_attn', 'norm',
                                     'ffn', 'norm', 'conv')))),
),
)
# Dataset config
dataset_type = 'CustomNuScenesOccDataset'
data_root = 'data/nuscenes/'  # dataset root
file_client_args = dict(backend='disk')

# Training data pipeline
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
]
# Test data pipeline (no augmentation)
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D', keys=['img'])
]

# Data loading config
find_unused_parameters = True
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    # Training set
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        use_semantic=use_semantic,
        classes=class_names,
        box_type_3d='LiDAR'),
    # Validation set
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file='data/nuscenes_infos_val.pkl',
             pipeline=test_pipeline,
             occ_size=occ_size,
             pc_range=point_cloud_range,
             use_semantic=use_semantic,
             classes=class_names,
             modality=input_modality),
    # Test set
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file='data/infos_inference.pkl',
              pipeline=test_pipeline,
              occ_size=occ_size,
              pc_range=point_cloud_range,
              use_semantic=use_semantic,
              classes=class_names,
              modality=input_modality),
    # Sampler config
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)
# Optimizer config (AdamW + layer-wise lr)
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
# Gradient clipping (prevent explosion)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 2
evaluation = dict(interval=1, pipeline=test_pipeline)
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from = 'ckpts/r101_dcn_fcos3d_pretrain.pth'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)
