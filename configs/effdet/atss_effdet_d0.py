# Training
total_epochs = 300
imgs_per_gpu = 32

lr_start = 8e-2
lr_end = 1e-4
warmup_iters = 917
warmup_ratio = 1/3
weight_decay = 4e-5

log_interval = 50
ckpt_interval = 1
evaluation = dict(interval=2, metric='bbox')

# Dataset
img_scale = (512, 512)
keep_ratio = False
size_divisor = 32
repeat_times = 1
workers_per_gpu = 10
data_root = 'datasets/coco/'
train_img_prefix = data_root + 'images/train2017/'
val_img_prefix = data_root + 'images/val2017/'
test_img_prefix = data_root + 'images/val2017/'
train_ann_file = data_root + 'annotations/instances_train2017.json'
val_ann_file = data_root + 'annotations/instances_val2017.json'
test_ann_file = data_root + 'annotations/instances_val2017.json'

# Model
conv_cfg = dict(type='ConvDWS')
norm_cfg = dict(type='SyncBN')
act_cfg = dict(type='Swish')

load_from = None
resume_from = None
work_dir = 'work_dirs/atss_effdet_d0/'

model = dict(
	type='RetinaNet',
	pretrained=None,
	backbone=dict(
		type='TimmBackbone',
		model_name='efficientnet_b0',
		norm_eval=True,
		frozen_stages=-1,
		pretrained=True,
		drop_rate=0.0,
		drop_path_rate=0.1,
		pad_type='same',
	),
	neck=dict(
		type='BiFPN',
		in_channels=[24,40,112,320],
		out_channels=64,
		start_level=1,
		num_outs=5,
		stack=3,
		add_extra_convs=True,
		extra_convs_on_inputs=True,
		conv_cfg=conv_cfg,
		norm_cfg=norm_cfg,
		activation=act_cfg,
	),
	bbox_head=dict(
		type='ATSSEffDetHead',
		num_classes=81,
		in_channels=64,
		stacked_convs=3,
		num_levels=5,
		feat_channels=64,
		octave_base_scale=8,
		scales_per_octave=1,
		anchor_ratios=[1.0],
		anchor_strides=[8, 16, 32, 64, 128],
		target_means=[.0, .0, .0, .0],
		target_stds=[0.1, 0.1, 0.2, 0.2],
		conv_cfg=conv_cfg,
		norm_cfg=norm_cfg,
		act_cfg=act_cfg,
		loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
		loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
		loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
	),
)
#training and testing settings
train_cfg = dict(
	assigner=dict(type='ATSSAssigner', topk=9),
	allowed_border=-1,
	pos_weight=-1,
	debug=False)
test_cfg = dict(
	nms_pre=1000,
	min_bbox_size=0,
	score_thr=0.05,
	nms=dict(type='nms', iou_thr=0.6),
	max_per_img=100)
# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
	mean=[123.675, 116.28, 103.53],
	std=[58.395, 57.12, 57.375],
	to_rgb=True)
train_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(type='LoadAnnotations', with_bbox=True),
	dict(type='Resize', img_scale=img_scale, keep_ratio=keep_ratio),
	dict(type='RandomFlip', flip_ratio=0.5),
	dict(type='Pad', size_divisor=size_divisor),
	dict(type='Normalize', **img_norm_cfg),
	dict(type='DefaultFormatBundle'),
	dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
	dict(type='LoadImageFromFile'),
	dict(
		type='MultiScaleFlipAug',
		img_scale=img_scale,
		flip=False,
		transforms=[
			dict(type='Resize', keep_ratio=keep_ratio),
			dict(type='RandomFlip'),
			dict(type='Normalize', **img_norm_cfg),
			dict(type='Pad', size_divisor=size_divisor),
			dict(type='ImageToTensor', keys=['img']),
			dict(type='Collect', keys=['img']),
		])
]
data = dict(
	imgs_per_gpu=imgs_per_gpu,
	workers_per_gpu=workers_per_gpu,
	train=dict(
		type='RepeatDataset',
		times=repeat_times,
		dataset=dict(
			type=dataset_type,
			ann_file=train_ann_file,
			img_prefix=train_img_prefix,
			pipeline=train_pipeline,
		),
	),
	val=dict(
		type=dataset_type,
		ann_file=val_ann_file,
		img_prefix=val_img_prefix,
		pipeline=test_pipeline,
	),
	test=dict(
		type=dataset_type,
		ann_file=test_ann_file,
		img_prefix=test_img_prefix,
		pipeline=test_pipeline,
	),
)
# optimizer
optimizer = dict(
	type='SGD',
	lr=lr_start, momentum=0.9, weight_decay=weight_decay,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
	policy='cosine', target_lr=lr_end, by_epoch=False,
	warmup='linear', warmup_iters=warmup_iters, warmup_ratio=warmup_ratio,
)
checkpoint_config = dict(interval=ckpt_interval)

# yapf:disable
log_config = dict(
	interval=log_interval,
	hooks=[
		dict(type='TextLoggerHook'),
	])
# yapf:enable
# runtime settings
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
