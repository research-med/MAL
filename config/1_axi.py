_base_ = ['./base/comm_config.py']

model = dict(backbone=dict(
    model_type='resnet18', final_input_channel=512, final_output_channel=1024))

data = dict(train=dict(dataset=dict(dataset=dict(positions=['axial']))),
            val=dict(positions=['axial']),
            test=dict(positions=['axial']))
