import json
import os.path as osp
import random

import numpy as np
import pydicom
import torch.nn.functional as F
from mmcls.datasets import DATASETS, BaseDataset
from sklearn import metrics


@DATASETS.register_module()
class SpineDataset(BaseDataset):
    """This is a dataset class for spine tumor.

    Args:
        BaseDataset ([torch.data.utils.Dataset]): Base dataset of mmcls
    """
    positions = ['axial', 'sagittal']

    def __init__(self,
                 json_file,
                 positions=None,
                 flag='train',
                 age_on_decision=False,
                 bgnet=False,
                 **kwargs):
        """This is a dataset class for spine tumor.

        Args:
            json_file (str): json path for annotation
            sample_num (int, optional): The sample number for this dataset each epoch. Defaults to None.
            flag (str, optional): flag for identifying 'train', 'valid' and 'test'. Defaults to 'train'.
        """
        assert flag in ['train', 'test']
        self.flag_type = flag
        self.classes = kwargs['classes']
        self.json_file = json_file
        self.cat2label = {cat: idx for idx, cat in enumerate(self.classes)}
        self.bgnet = bgnet
        if positions:
            self.positions = positions
        self.age_on_decision = age_on_decision
        super(SpineDataset, self).__init__(**kwargs)
        if not self.test_mode:
            # shuffle the training set
            random.shuffle(self.data_infos)
            random.shuffle(self.data_infos)
            random.shuffle(self.data_infos)

    def convert_dataset(self, json_data):
        annotations = json_data['annotations']
        new_annotations = []
        pos_map = {'轴位': 'axial', '矢状位': 'sagittal'}
        for annotation in annotations:
            pos = pos_map[annotation['position']]
            annotation['position'] = pos
            new_annotations.append(annotation)
        json_data['annotations'] = new_annotations
        return json_data

    def load_annotations(self):
        data_infos = []
        json_data = json.load(open(self.json_file, 'r'))
        json_data = self.convert_dataset(json_data)
        images = json_data['images']
        annotations = json_data['annotations']
        categories = json_data['categories']

        id2img = {image['id']: image for image in images}
        id2cat = {categorie['id']: categorie for categorie in categories}
        ids = 0
        patient_id_map = {}
        patient_id_set = set()
        patient_id_age = {}

        # collect by patient
        for annotation in annotations:
            img_id = annotation['image_id']
            cat_id = annotation['category_id']
            bbox = annotation['bbox']  # x1, y1, w, h

            img = id2img[img_id]
            cat = id2cat[cat_id]

            file_name = img['file_name']
            img_path = osp.join(self.data_prefix, file_name)

            dcm_data = pydicom.read_file(img_path)
            age = dcm_data.PatientAge
            try:
                age = int(age[:-1])
            except:
                age = -1

            # get the label
            cat_name = cat['name']
            gt_label = self.cat2label[cat_name]

            # get current position
            position = annotation['position']
            if position not in self.positions:
                continue
            position_id = self.positions.index(position)

            # patient_id -> id
            patient_id = annotation['patient_id']
            if patient_id not in patient_id_set:
                patient_id_set.add(patient_id)
                patient_id_map[patient_id] = ids
                ids += 1

            patient_id = patient_id_map[patient_id]

            patient_id_age[patient_id] = age

            info = dict(filename=img_path,
                        gt_label=np.asarray(gt_label, np.int64),
                        position_id=np.asarray(position_id, np.int64),
                        patient_id=np.asarray(patient_id, np.int64),
                        bbox=bbox,
                        flag='single-frame')

            data_infos.append(info)

        # id -> patiend_id
        self.patient_id_map_inv = {
            id: patient_id
            for patient_id, id in patient_id_map.items()
        }
        self.patient_id_age = patient_id_age
        if self.bgnet:
            return self.build_bipartite_graph(data_infos)
        return data_infos

    def build_bipartite_graph(self, data_infos):
        data_infos_patient_id_dict = {}
        new_data_infos = []
        for info in data_infos:
            patient_id = info['patient_id'].item()
            position_id = info['position_id'].item()
            position = self.positions[position_id]
            if patient_id not in data_infos_patient_id_dict:
                data_infos_patient_id_dict[patient_id] = {}
            if position not in data_infos_patient_id_dict[patient_id]:
                data_infos_patient_id_dict[patient_id][position] = []
            data_infos_patient_id_dict[patient_id][position].append(info)
        for key in data_infos_patient_id_dict.keys():
            info = {
                'data':
                data_infos_patient_id_dict[key],
                'gt_label':
                data_infos_patient_id_dict[key]['axial'][0]['gt_label'],
                'patient_id':
                data_infos_patient_id_dict[key]['axial'][0]['patient_id'],
                'flag':
                'bg-multi-frame'
            }
            new_data_infos.append(info)

        data_infos = new_data_infos

        if self.test_mode:
            # for test stage, activate all edge
            test_data_infos = []
            for data_info in data_infos:
                data = data_info['data']
                axial_infos = data['axial']
                sagittal_infos = data['sagittal']

                for axial_info in axial_infos:
                    for sagittal_info in sagittal_infos:
                        info = {
                            'data': {
                                'axial': axial_info,
                                'sagittal': sagittal_info,
                            },
                            'gt_label': sagittal_info['gt_label'],
                            'patient_id': sagittal_info['patient_id'],
                            'flag': 'bg-multi-frame'
                        }
                        test_data_infos.append(info)
            data_infos = test_data_infos
        return data_infos

    def evaluate(self,
                 results,
                 patient_ids=None,
                 topk=300,
                 thr=0.6,):
        """Evaluate the dataset.
        """
        from sklearn.metrics import (accuracy_score, confusion_matrix,
                                     roc_auc_score)

        eval_results = {}
        results = np.vstack(results)
        patient_ids = np.vstack(patient_ids)
        gt_labels = self.get_gt_labels()

        # start patient-level fusion
        preds_dict = {}
        targets_dict = {}
        pred_scores_malignant_dict = {}
        print(f"Use age on decision: {self.age_on_decision}")
        for pred, target, patient_id in zip(results, gt_labels, patient_ids):
            patient_id = patient_id[0]
            age = self.patient_id_age[patient_id]
            if patient_id not in preds_dict:
                preds_dict[patient_id] = [[]]
            p = pred[0]
            preds_dict[patient_id][0].append(p)

            if patient_id not in targets_dict:
                targets_dict[patient_id] = set()
            targets_dict[patient_id].add(target)
            assert len(targets_dict[patient_id]) == 1, targets_dict[patient_id]

        # get the patient-level diagnoses results
        for patient_id, preds in preds_dict.items():
            preds = np.asarray(preds)
            p = preds[0].tolist()
            topk_ = max(len(p) // 2, topk)
            s = sorted(p, key=lambda x: x, reverse=True)[:topk_]
            s = np.asarray(s)
            b = np.sum(s < thr)
            m = np.sum(s > thr)
            preds = np.asarray([b / (b + m), m / (b + m)])
            preds_dict[patient_id] = np.argmax(preds, 0)
            pred_scores_malignant_dict[patient_id] = preds[1]

        preds = []
        targets = []
        pred_scores_malignant = []
        # get patient ids
        pk = set(preds_dict.keys())
        tk = set(targets_dict.keys())
        assert len(pk.intersection(tk)) == len(pk)

        for key in pk:
            preds.append(preds_dict[key])
            targets.append(list(targets_dict[key])[0])
            pred_scores_malignant.append(pred_scores_malignant_dict[key])

        # confusion matrix
        confusion_mat = confusion_matrix(targets, preds)
        # SE and SP
        SE = confusion_mat[1, 1] / (confusion_mat[1, 0] + confusion_mat[1, 1])
        SP = confusion_mat[0, 0] / (confusion_mat[0, 0] + confusion_mat[0, 1])
        acc = accuracy_score(targets, preds)
        # auc
        malignant_auc = roc_auc_score(targets, pred_scores_malignant)

        eval_results = {
            'patient_acc': acc,
            'patient_auc': malignant_auc,
            'patient_se': SE,
            'patient_sp': SP,
        }

        return eval_results