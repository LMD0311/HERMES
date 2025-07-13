_base_ = [
    "../../../configs/_base_/datasets/nus-3d.py",
    "../../../configs/_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
frustum_range = [0, 0, 1.0, 1600, 928, 60.0]
frustum_size = [32.0, 32.0, 0.5]
cam_sweep_num = 1
lidar_sweep_num = 10
fp16_enabled = True
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
img_norm_cfg = dict(mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD, to_rgb=True)
_dim_ = 256
_pos_dim_ = 128
_ffn_dim_ = 512
_num_levels_ = 4
_num_mono_levels_ = 5
group_detr = 11
bev_h_ = 200
bev_w_ = 200
voxel_z_ = 32
unified_voxel_size = [102.4 / bev_h_, 102.4 / bev_w_, 8 / voxel_z_]
frames = (0, 2, 4, 6,)
voxel_size = [102.4 / bev_h_, 102.4 / bev_w_, 8]
point_nsample = 4096
unified_voxel_shape = [
    int((point_cloud_range[3] - point_cloud_range[0]) / unified_voxel_size[0]),
    int((point_cloud_range[4] - point_cloud_range[1]) / unified_voxel_size[1]),
    int((point_cloud_range[5] - point_cloud_range[2]) / unified_voxel_size[2]),
]

# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

model = dict(
    type="HERMES",
    num_levels=_num_levels_,
    frames=frames,
    frame_loss_weight=[1.0, 1.5, 2.0, 2.5],
    text_loss_weight=1.0,
    img_backbone=dict(
        type='OpenCLIPConvnext',
        model_args=dict(depths=[3, 3, 27, 3],
                        dims=[128, 256, 512, 1024],
                        drop_path_rate=0.1,
                        indices=[2, 3, 4],
                        intermediates_only=True,
                        grad_checkpointing=False,
                        ),
        pretrain_path='./ckpt/open_clip_convnext_base_w-320_laion_aesthetic-s13B-b82k.bin',
    ),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_outs=4),
    pts_bbox_head=dict(
        type="HERMESFutureRenderHead",
        fp16_enabled=fp16_enabled,
        in_channels=256,
        unified_voxel_size=unified_voxel_size,
        unified_voxel_shape=unified_voxel_shape,
        pc_range=point_cloud_range,
        frames=frames,
        #################### Need to be modified ####################
        group_detr=group_detr,
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_classes=10,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        use_downsample=True,
        use_atten_pool=False,
        use_max_pool=True,
        use_temporal_fusion=False,
        num_ego_layers=3,
        llm_cfg=dict(llm_path='./projects/mmdet3d_plugin/models/internvl_chat/pretrained/InternVL2-2B',
                     set_lora=True,
                     is_pretraining=True,
                     load_internvl_weight=True,
                     img_length=bev_h_ // 4,
                     attention_type='flash_attention_2',
                     chat_config=dict(
                         num_beams=1,
                         max_new_tokens=1000,
                         min_new_tokens=1,
                         do_sample=False,
                         temperature=0., ),
                     num_learnable_query=4,
                     ),
        use_can_bus=True,
        loss_weight=dict(
            channel_loss=0.0,
            l1_loss=0,
        ),
        transformer=dict(
            type='PerceptionTransformerV2',
            embed_dims=_dim_,
            frames=frames,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=6,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_),
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='SmoothL1Loss', loss_weight=0.75, beta=1.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        ############## Need to be modified ############
        view_cfg=None,
        render_conv_cfg=dict(out_channels=32, kernel_size=3, padding=1),
        ray_sampler_cfg=dict(
            close_radius=.0,
            only_img_mask=False,
            only_point_mask=False,
            replace_sample=False,
            point_nsample=1024,
            point_ratio=0.5,
            pixel_interval=4,
            sky_region=0.4,
            merged_nsample=1024,
        ),
        render_ssl_cfg=dict(
            type="NeuSModel",
            norm_scene=True,
            field_cfg=dict(
                type="SDFField",
                sdf_decoder_cfg=dict(
                    in_dim=32, out_dim=16 + 1, hidden_size=16, n_blocks=5
                ),
                rgb_decoder_cfg=dict(
                    in_dim=32 + 16 + 3 + 3, out_dim=32 + 16, hidden_size=16, n_blocks=3
                ),
                interpolate_cfg=dict(type="SmoothSampler", padding_mode="zeros"),
                beta_init=0.3,
            ),
            collider_cfg=dict(type="AABBBoxCollider", near_plane=1.0),
            sampler_cfg=dict(
                type="NeuSSampler",
                initial_sampler="UniformSampler",
                num_samples=96,
                num_samples_importance=42,
                num_upsample_steps=1,
                train_stratified=True,
                single_jitter=True,
            ),
            loss_cfg=dict(
                sensor_depth_truncation=0.1,
                sparse_points_sdf_supervised=False,
                weights=dict(
                    depth_loss=10.,
                    rgb_loss=0.,
                    vgg_loss=0.,
                ),
            ),
        ),
    ),
)

dataset_type = "CustomNuScenesDatasetV2"
data_root = "data/nuscenes/"

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="PointShuffle"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type='LoadAnnoatationVQA',
        llm_path='./projects/mmdet3d_plugin/models/internvl_chat/pretrained/InternVL2-2B',
        base_desc_path='./data/omnidrive_nusc/desc/train/',
        base_key_path='./data/omnidrive_nusc/keywords/train/',
        base_conv_path='./data/omnidrive_nusc/conv/train/',
        future_prediction=True,
    ),
    dict(
        type='CustomCollect3D',
        keys=['points', 'img', 'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation',
              'lidar2ego_rotation', 'timestamp', 'MASKED_CAM', 'conv']),
]
test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        file_client_args=file_client_args,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type='CustomCollect3D',
        keys=['points', 'img', 'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation',
              'lidar2ego_rotation', 'timestamp', 'MASKED_CAM']),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=8,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        frames=frames,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        data_root=data_root,
        ann_file=data_root + "nuscenes_infos_temporal_train.pkl",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d="LiDAR",
        filter_empty_gt=False,
    ),
    val=dict(
        type=dataset_type,
        frames=frames,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        data_root=data_root,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        ann_file=data_root + "nuscenes_infos_temporal_val.pkl",
        samples_per_gpu=1
    ),
    test=dict(
        type=dataset_type,
        frames=frames,
        ego_mask=(-0.8, -1.5, 0.8, 2.5),
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.1),
        )),
    weight_decay=0.01)
optimizer_config = dict(type="GradientCumulativeFp16OptimizerHook", cumulative_iters=4,
                        loss_scale='dynamic',
                        grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    by_epoch=False,
)

total_epochs = 36
evaluation = dict(interval=4, pipeline=test_pipeline)
checkpoint_config = dict(max_keep_ckpts=1, interval=1)
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
find_unused_parameters = False
runner = dict(type="EpochBasedRunner", max_epochs=total_epochs)

load_from = './ckpt/hermes_stage2_2.pth'
resume_from = None