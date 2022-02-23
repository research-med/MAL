# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.builder import HEADS, build_loss
from mmcls.models.heads.cls_head import ClsHead
from mmcls.models.losses.accuracy import accuracy


@HEADS.register_module()
class BgNetLinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 use_sigmod=False,
                 head_output_dim=1,
                 *args,
                 **kwargs):
        super(BgNetLinearClsHead, self).__init__(
            init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_sigmod = use_sigmod

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')
        self.fcs = nn.ModuleList()

        fc = nn.Sequential(
            nn.Linear(self.in_channels, 512),
            nn.Linear(512, head_output_dim),
        )
        self.fcs.append(fc)

    def loss(self, cls_score, gt_label):
        num_samples = len(cls_score)
        losses = dict()
        # compute loss
        loss = self.compute_loss(cls_score, gt_label, avg_factor=num_samples)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses[f'accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        prefix = prefix.strip('_')
        losses[f'loss'] = loss
        return losses

    def simple_test(self, x):
        """Test without augmentation."""
        if isinstance(x, tuple):
            x = x[-1]
        x = x.view(x.shape[0], -1)
        cls_score = self.fcs[0](x)

        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        if self.use_sigmod:
            pred = F.sigmoid(cls_score) if cls_score is not None else None
        else:
            pred = F.softmax(
                cls_score, dim=1) if cls_score is not None else None

        return self.post_process(pred)

    def forward_train(self, x, gt_label, patient_id=None):
        if isinstance(x, tuple):
            x = x[-1]
        x = x.view(x.shape[0], -1)
        cls_score = self.fcs[0](x)
        if cls_score.shape[-1] == 1:
            cls_score = cls_score.view((cls_score.shape[0], ))
        losses = self.loss(cls_score, gt_label.float())
        return losses
