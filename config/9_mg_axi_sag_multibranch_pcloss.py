_base_ = ['./base/comm_config.py']

model = dict(
    backbone=dict(
        model_type='resnet18_multibranch_pcloss',
        final_input_channel=2048,
        final_output_channel=1024))

data = dict(
    train=dict(dataset=dict(bgnet=True, positions=['axial', 'sagittal'])),
    val=dict(bgnet=True, positions=['axial', 'sagittal']),
    test=dict(bgnet=True, positions=['axial', 'sagittal']))
