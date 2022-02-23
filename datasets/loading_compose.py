import random

import cv2
import numpy as np
import pydicom
from mmcls.datasets.builder import PIPELINES


@PIPELINES.register_module()
class InputLyaerFusion(object):

    def __init__(self,
                 to_float32=False,
                 test_mode=False,
                 target_size=(224, 224),
                 test_edge=0):
        self.to_float32 = to_float32
        self.test_mode = test_mode
        self.target_size = target_size
        self.test_edge = test_edge

    def get_img(self, results, target_size):
        # read image
        img_path = results['filename']
        data_dcm = pydicom.read_file(img_path)
        modality = data_dcm.Modality
        # get the bbox
        bbox = results['bbox']
        img_arr = data_dcm.pixel_array

        # for mri, change the value of each piexl to 0~255
        center = data_dcm.get('WindowCenter', None)
        width = data_dcm.get('WindowWidth', None)
        if 'MR' in modality:
            # for mri
            min_ = (2 * center - width) / 2.0 + 0.5
            max_ = (2 * center + width) / 2.0 + 0.5
            dFactor = 255 / (max_ - min_)
            img_arr = (img_arr - min_) * dFactor
            img_arr[img_arr < 0.0] = 0
            img_arr[img_arr > 255] = 255
        else:
            # for ct
            unknow = 'unknow'
            RescaleSlope = data_dcm.get('RescaleSlope', unknow)
            RescaleIntercept = data_dcm.get('RescaleIntercept', unknow)
            if RescaleSlope != data_dcm and RescaleIntercept != unknow:
                img_arr = img_arr * RescaleSlope + RescaleIntercept

            def rescale(center, width, img_):
                min_ = (2 * center - width) / 2.0 + 0.5
                max_ = (2 * center + width) / 2.0 + 0.5
                dFactor = 255 / (max_ - min_)
                img_ = (img_ - min_) * dFactor
                img_[img_ < 0.0] = 0
                img_[img_ > 255] = 255
                return img_

            img_arr = rescale(center=40, width=300, img_=img_arr)

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]

        high = img_arr.shape[0]
        width = img_arr.shape[1]
        if not self.test_mode:
            expand = random.randint(40, 60)
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(width, x2 + expand)
            y2 = min(high, y2 + expand)
        else:
            expand = self.test_edge
            x1 = max(0, x1 - expand)
            y1 = max(0, y1 - expand)
            x2 = min(width, x2 + expand)
            y2 = min(high, y2 + expand)
        img_patch = img_arr[int(y1):int(y2), int(x1):int(x2)]
        img_patch = cv2.resize(img_patch, (target_size[1], target_size[0]))

        # random flip
        if not self.test_mode:
            seed = np.random.randint(0, 3)
            if seed == 0:
                img_arr = np.flipud(img_arr)
                img_patch = np.flipud(img_patch)
            if seed == 1:
                img_arr = np.fliplr(img_arr)
                img_patch = np.fliplr(img_patch)

        img_patch = np.expand_dims(img_patch, -1)
        img_arr = cv2.resize(img_arr, (target_size[1], target_size[0]))
        img_arr = np.expand_dims(img_arr, -1)

        img = np.concatenate([img_patch, img_patch, img_patch], -1)

        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def __call__(self, results):
        self.single_frame = results['flag'] == 'single-frame'
        if self.single_frame:
            result = results
            # get image: for single modality
            patch_target_size = (self.target_size[0], self.target_size[1])
            img = self.get_img(result, patch_target_size)

            results['filename'] = [res['filename'] for res in [result]]
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
            results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                         dtype=np.float32),
                                           std=np.ones(num_channels,
                                                       dtype=np.float32),
                                           to_rgb=False)
            return results
        else:
            # get image: for multi-modality based on bipartite graph
            data = results['data']
            if not self.test_mode:
                # training stage, random activate the edge
                axial1_result = random.choice(data['axial'])
                sagittal1_result = random.choice(data['sagittal'])
            else:
                # testing stage
                axial1_result = data['axial']
                sagittal1_result = data['sagittal']

            assert axial1_result['gt_label'] == sagittal1_result['gt_label']
            # get images
            patch_target_size = (self.target_size[0], self.target_size[1])
            axial1_img = self.get_img(axial1_result, patch_target_size)
            sagittal1_img = self.get_img(sagittal1_result, patch_target_size)

            # combine the axial and sagittal up and down
            img = np.zeros((self.target_size[0], self.target_size[1],
                            axial1_img.shape[-1] * 2),
                           dtype=axial1_img.dtype)
            img[:, :, :3] = axial1_img
            img[:, :, 3:] = sagittal1_img

            results['filename'] = [
                res['filename'] for res in [axial1_result, sagittal1_result]
            ]
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            num_channels = 1 if len(img.shape) < 3 else img.shape[2]
            results['img_norm_cfg'] = dict(mean=np.zeros(num_channels,
                                                         dtype=np.float32),
                                           std=np.ones(num_channels,
                                                       dtype=np.float32),
                                           to_rgb=False)

            return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32})')
        return repr_str