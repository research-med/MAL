model = dict(type='MALClassifier',
             backbone=dict(type='MMFM',
                           img_size=224,
                           depth=18,
                           in_channels=3,
                           out_indices=(0, 1, 2, 3),
                           patch_size=8,
                           embed_channels=768,
                           decoder_in_channel=1024,
                           decoder_channel=512,
                           decoder_num_convs=1,
                           decoder_kernel_size=3,
                           decoder_dilation=1,
                           decoder_concat_input=True,
                           decoder_dropout_ratio=0.1,
                           decoder_output_channel=3,
                           decoder_align_corners=False,
                           decoder_conv_cfg=None,
                           decoder_norm_cfg=dict(type='BN',
                                                 requires_grad=True),
                           decoder_act_cfg=dict(type='ReLU'),
                           final_input_channel=2048,
                           final_output_channel=1024),
             neck=dict(type='GlobalAveragePooling'),
             head=dict(type='BgNetLinearClsHead',
                       num_classes=1,
                       in_channels=1024,
                       init_cfg=None,
                       use_sigmod=True,
                       loss=dict(type='CrossEntropyLoss',
                                 loss_weight=1.0,
                                 use_sigmoid=True,
                                 class_weight=[0.3, 0.7])))

train_pipeline = [
    dict(type='InputLyaerFusion',
         to_float32=True,
         target_size=(224, 224),
         train_edge_range=(40, 60)),
    dict(type='CustomNormalize'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label', 'patient_id'])
]

test_pipeline = [
    dict(type='InputLyaerFusion',
         test_mode=True,
         to_float32=True,
         target_size=(224, 224),
         test_edge=40),
    dict(type='CustomNormalize'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img', 'patient_id'])
]

# =================== datasets ======================
train_json = '/path/to/your/train.json'
test_json = '/path/to/your/test.json'
data_root = '/path/to/your/data'
dataset_type = 'SpineDataset'

data = dict(samples_per_gpu=32,
            workers_per_gpu=6,
            train=dict(type='RepeatDataset',
                       times=16,
                       dataset=dict(type=dataset_type,
                                    data_prefix=data_root,
                                    pipeline=train_pipeline,
                                    json_file=train_json,
                                    classes=['Benign', 'Malignant'],
                                    age_on_decision=True)),
            val=dict(type=dataset_type,
                     data_prefix=data_root,
                     json_file=test_json,
                     classes=['Benign', 'Malignant'],
                     pipeline=test_pipeline,
                     test_mode=True,
                     flag='valid',
                     age_on_decision=True),
            test=dict(type=dataset_type,
                      data_prefix=data_root,
                      json_file=test_json,
                      classes=['Benign', 'Malignant'],
                      pipeline=test_pipeline,
                      test_mode=True,
                      flag='test',
                      age_on_decision=True))

# log
log_level = 'INFO'
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

dist_params = dict(backend='nccl')
load_from = None
resume_from = None
workflow = [('train', 1)]

optimizer = dict(type='SGD', lr=0.0002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# lr
lr_config = dict(policy='CosineAnnealing',
                 by_epoch=False,
                 min_lr_ratio=1e-3,
                 warmup='linear',
                 warmup_ratio=1e-3,
                 warmup_iters=2 * 78,
                 warmup_by_epoch=False)

# runner
runner = dict(type='EpochBasedRunner', max_epochs=20)

# evaluation
evaluation = dict(interval=5)

# checkpoint
checkpoint_config = dict(interval=1)

eval_by_patient = True

find_unused_parameters = True