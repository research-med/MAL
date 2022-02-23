from enum import Enum
import math
import time
import cv2
import numpy as np

import warnings

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcls.models.backbones import swin_transformer

from mmcls.models.backbones.resnet import ResNet
from mmcls.models.backbones.swin_transformer import SwinTransformer
from mmcls.models.builder import BACKBONES
from mmcls.models.backbones import ResNet

from mmcv.cnn import ConvModule

from timm.models.vision_transformer import Block


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Model_Type:
    resnet18 = 'resnet18'
    resnet18_multibranch = 'resnet18_multibranch'
    resnet18_pcloss = 'resnet18_pcloss'
    resnet18_tcloss = 'resnet18_tcloss'
    resnet18_pcloss_tcloss = 'resnet18_pcloss_tcloss'
    resnet18_multibranch_pcloss = 'resnet18_multibranch_pcloss'
    resnet18_multibranch_tcloss = 'resnet18_multibranch_tcloss'
    resnet18_multibranch_pcloss_tcloss = 'resnet18_multibranch_pcloss_tcloss'


@BACKBONES.register_module()
class MMFM(ResNet):  # multi_modal_fusion_model

    def __init__(self,
                 img_size=224,
                 patch_size=8,
                 mask_patch_size=8,
                 model_type=None,
                 embed_channels=768,
                 decoder_in_channel=256,
                 decoder_channel=512,
                 decoder_num_convs=2,
                 decoder_kernel_size=3,
                 decoder_dilation=1,
                 decoder_concat_input=True,
                 decoder_dropout_ratio=0.1,
                 decoder_output_channel=3,
                 decoder_align_corners=False,
                 decoder_conv_cfg=None,
                 decoder_norm_cfg=dict(type='SyncBN', requires_grad=True),
                 decoder_act_cfg=dict(type='ReLU'),
                 final_input_channel=512,
                 final_output_channel=1024,
                 *args,
                 **kwargs):
        super(MMFM, self).__init__(*args, **kwargs)
        if model_type is not None:
            assert model_type in [
                'resnet18', 'resnet18_multibranch', 'resnet18_pcloss',
                'resnet18_tcloss', 'resnet18_pcloss_tcloss',
                'resnet18_multibranch_pcloss', 'resnet18_multibranch_tcloss',
                'resnet18_multibranch_pcloss_tcloss'
            ]
        self.model_type = model_type
        in_channels = kwargs.get('in_channels', 3)
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_patch_size = mask_patch_size

        self.decoder_align_corners = decoder_align_corners
        self.final_input_channel = final_input_channel
        self.final_output_channel = final_output_channel

        # decoder config
        self.decoder_in_channel = decoder_in_channel
        self.decoder_channel = decoder_channel
        self.decoder_num_convs = decoder_num_convs
        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_dilation = decoder_dilation
        self.decoder_concat_input = decoder_concat_input
        self.decoder_dropout_ratio = decoder_dropout_ratio
        self.decoder_output_channel = decoder_output_channel
        self.decoder_align_corners = decoder_align_corners
        self.decoder_conv_cfg = decoder_conv_cfg
        self.decoder_norm_cfg = decoder_norm_cfg
        self.decoder_act_cfg = decoder_act_cfg

        # patch_embed
        self.patch_projection = ConvModule(in_channels,
                                           embed_channels,
                                           kernel_size=self.patch_size,
                                           stride=self.patch_size)
        self.atten_block = Block(dim=embed_channels,
                                 num_heads=4,
                                 mlp_ratio=4,
                                 qkv_bias=True)

        conv_padding = (decoder_kernel_size // 2) * decoder_dilation
        convs = []
        convs.append(
            ConvModule(decoder_in_channel,
                       decoder_channel,
                       kernel_size=decoder_kernel_size,
                       padding=conv_padding,
                       dilation=decoder_dilation,
                       conv_cfg=decoder_conv_cfg,
                       norm_cfg=decoder_norm_cfg,
                       act_cfg=decoder_act_cfg))

        for i in range(decoder_num_convs - 1):
            convs.append(
                ConvModule(decoder_channel,
                           decoder_channel,
                           kernel_size=decoder_kernel_size,
                           padding=conv_padding,
                           dilation=decoder_dilation,
                           conv_cfg=decoder_conv_cfg,
                           norm_cfg=decoder_norm_cfg,
                           act_cfg=decoder_act_cfg))

        self.convs = nn.Sequential(*convs)

        if patch_size == 8:
            self.atten_maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        elif patch_size == 16:
            self.atten_maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 32:
            self.atten_maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.final = ConvModule(final_input_channel,
                                final_output_channel,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                conv_cfg=decoder_conv_cfg,
                                norm_cfg=decoder_norm_cfg,
                                act_cfg=decoder_act_cfg)

        # add mlp projection head
        self.fc1 = nn.Sequential(nn.Linear(512 * 7 * 7, 128))
        self.fc2 = nn.Sequential(nn.Linear(512 * 7 * 7, 128))

        self.fc3 = nn.Sequential(nn.Linear(768 * 7 * 7, 128))
        self.fc4 = nn.Sequential(nn.Linear(768 * 7 * 7, 128))

        self.mse_loss = nn.MSELoss()

    def patch(self, x):
        x = self.patch_projection(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)
        return x

    def trans_input(self, latents):
        return latents[-1]

    def forward_encoder(self, x):
        x = super(MMFM, self).forward(x)
        return x

    def cnn_branch(self, imgs):
        # x: [B, 6, H, W]
        axial_imgs = imgs[:, :3]
        sagittal_imgs = imgs[:, 3:]
        latents_axi = self.forward_encoder(axial_imgs)
        latents_axi = self.trans_input(latents_axi)
        latents_sag = self.forward_encoder(sagittal_imgs)
        latents_sag = self.trans_input(latents_sag)
        x = torch.cat([latents_axi, latents_sag], 1)
        output = self.convs(x)
        return output, latents_axi, latents_sag

    def attention_branch(self, imgs):
        axi_imgs = imgs[:, :3]
        sag_imgs = imgs[:, 3:]

        axi_patches = self.patch(
            axi_imgs)  # B x C x (H//patch_size) x (W//patch_size)
        sag_patches = self.patch(
            sag_imgs)  # B x C x (H//patch_size) x (W//patch_size)

        axi_atten = self.atten_block(axi_patches)
        sag_atten = self.atten_block(sag_patches)

        B, L, C = axi_atten.shape
        axi_atten = axi_atten.permute(0, 2, 1).view(B, C, int(L**0.5),
                                                    int(L**0.5))
        sag_atten = sag_atten.permute(0, 2, 1).view(B, C, int(L**0.5),
                                                    int(L**0.5))

        axi_atten = self.atten_maxpool(axi_atten)
        sag_atten = self.atten_maxpool(sag_atten)

        atten = torch.cat([axi_atten, sag_atten], 1)
        return atten, axi_atten, sag_atten

    def forward_resnet(self, imgs):
        x = super(MMFM, self).forward(imgs)
        x = self.trans_input(x)
        return self.final(x)

    def forward(self, imgs, gt_label=None, patient_id=None):
        if imgs.shape[1] == 3:
            # for single view
            return self.forward_resnet(imgs)
        # for multi view
        # x: [B, 6, H, W]
        cnn_branch_feature, cnn_f1, cnn_f2 = self.cnn_branch(imgs)
        attention_branch_feature, atten_f1, atten_f2 = self.attention_branch(
            imgs)
        feature = torch.cat([cnn_branch_feature, attention_branch_feature], 1)
        loss = dict()
        if gt_label is not None:
            latents_cl_loss = self.comparative_learning_loss(
                cnn_f1, cnn_f2, gt_label, patient_id, 'cnn', self.fc1,
                self.fc2)
            atten_cl_loss = self.comparative_learning_loss(
                atten_f1, atten_f2, gt_label, patient_id, 'atten', self.fc3,
                self.fc4)

        if self.model_type == Model_Type.resnet18:
            return self.final(cnn_branch_feature)
        elif self.model_type == Model_Type.resnet18_multibranch:
            return self.final(feature)
        elif (self.model_type == Model_Type.resnet18_pcloss
              or self.model_type == Model_Type.resnet18_tcloss
              or self.model_type == Model_Type.resnet18_pcloss_tcloss):
            if gt_label is not None:
                loss.update(latents_cl_loss)
                return self.final(cnn_branch_feature), loss
            else:
                return self.final(cnn_branch_feature)
        elif (self.model_type == Model_Type.resnet18_multibranch_pcloss
              or self.model_type == Model_Type.resnet18_multibranch_tcloss
              or self.model_type
              == Model_Type.resnet18_multibranch_pcloss_tcloss):
            if gt_label is not None:
                loss.update(latents_cl_loss)
                loss.update(atten_cl_loss)
                return self.final(feature), loss
            else:
                return self.final(feature)

        return self.final(feature)

    def compute_sim_metrix(self, A, B):
        n = A.shape[0]
        C = torch.mm(A, B.t())
        A_ = torch.sqrt(torch.sum(A**2, 1).unsqueeze(-1).expand(n, n))
        B_ = torch.sqrt(torch.sum(B**2, 1).unsqueeze(0).expand(n, n))
        D = A_ * B_
        res = C / D
        return res

    def pcloss(self, fa, fb, ids):
        multi_id_dict = {}
        for i in range(ids.shape[0]):
            idd = ids[i].item()
            if idd not in multi_id_dict:
                multi_id_dict[idd] = []
            multi_id_dict[idd].append(i)

        data_list_axi = []
        data_list_sag = []
        for _, val in multi_id_dict.items():
            axi_f = fa[val]
            sag_f = fb[val]
            axi_f_mean = torch.mean(axi_f, 0)
            sag_f_mean = torch.mean(sag_f, 0)
            data_list_axi.append(axi_f_mean.unsqueeze(0))
            data_list_sag.append(sag_f_mean.unsqueeze(0))

        data_list_axi = torch.cat(data_list_axi, 0)
        data_list_sag = torch.cat(data_list_sag, 0)
        eye = torch.eye(data_list_axi.shape[0]).to(data_list_axi.device)

        sim_metrics = self.compute_sim_metrix(data_list_axi, data_list_sag)
        return self.mse_loss(sim_metrics, eye)

    def tcloss(self, fa, fb, ids):
        multi_id_dict = {}
        for i in range(ids.shape[0]):
            idd = ids[i].item()
            if idd not in multi_id_dict:
                multi_id_dict[idd] = []
            multi_id_dict[idd].append(i)

        data_list = []
        for key, val in multi_id_dict.items():
            axi_f = fa[val]
            sag_f = fb[val]
            axi_f_mean = torch.mean(axi_f, 0)
            sag_f_mean = torch.mean(sag_f, 0)
            mean_f = (axi_f_mean + sag_f_mean) / 2
            data_list.append(mean_f.unsqueeze(0))

        data_list = torch.cat(data_list, 0)
        eye = torch.eye(data_list.shape[0]).to(data_list.device)

        sim_metrics = self.compute_sim_metrix(data_list, data_list)
        tcloss = self.mse_loss(sim_metrics, eye)
        return tcloss

    def comparative_learning_loss(self, T1, T2, gt_label, patient_id, mode,
                                  fc1, fc2):
        T1 = T1.view(T1.shape[0], -1)
        T2 = T2.view(T2.shape[0], -1)

        T1 = fc1(T1)
        T2 = fc2(T2)

        base_multi = int(1e3)
        gt_label = gt_label % base_multi

        pcloss = self.pcloss(T1, T2, patient_id)

        tcloss = self.tcloss(T1, T2, gt_label)

        loss = None
        if self.model_type == Model_Type.resnet18_pcloss_tcloss or self.model_type == Model_Type.resnet18_multibranch_pcloss_tcloss:
            loss = pcloss + tcloss
        elif self.model_type == Model_Type.resnet18_pcloss or self.model_type == Model_Type.resnet18_multibranch_pcloss:
            loss = pcloss
        elif self.model_type == Model_Type.resnet18_tcloss or self.model_type == Model_Type.resnet18_multibranch_tcloss:
            loss = tcloss

        return {
            f'{mode}_loss': loss,
        }
