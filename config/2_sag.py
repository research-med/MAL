_base_ = ['./base/comm_config.py']

model = dict(backbone=dict(
    model_type='resnet18', final_input_channel=512, final_output_channel=1024))

data = dict(train=dict(dataset=dict(dataset=dict(positions=['sagittal']))),
            val=dict(positions=['sagittal']),
            test=dict(positions=['sagittal']))
