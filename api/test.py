# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import pickle
import time

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def multi_gpu_test_by_patient(model,
                              data_loader,
                              tmpdir=None,
                              tmpdir1=None,
                              gpu_collect=False):
    model.eval()
    results = []
    patient_ids = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        # Check if tmpdir is valid for cpu_collect
        if (not gpu_collect) and (tmpdir is not None and osp.exists(tmpdir)):
            raise OSError((f'The tmpdir {tmpdir} already exists.',
                           ' Since tmpdir will be deleted after testing,',
                           ' please make sure you specify an empty one.'))
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        patient_id = data['patient_id']
        data.pop('patient_id')
        patient_id = patient_id.numpy()
        B = patient_id.shape[0]
        patient_id_tmp = []
        for i in range(B):
            patient_id_tmp.append(np.asarray(patient_id[i], dtype=np.int64))
        patient_id = patient_id_tmp
        with torch.no_grad():
            result = model(return_loss=False, **data)
        if isinstance(result, list):
            results.extend(result)
            patient_ids.extend(patient_id)
        else:
            results.append(result)
            patient_ids.append(patient_id)
        if rank == 0:
            batch_size = data['img'].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results_gpu(results, len(dataset))
    patient_ids = collect_results_gpu(patient_ids, len(dataset))
    return results, patient_ids


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_result = pickle.loads(recv[:shape[0]].cpu().numpy().tobytes())
            part_list.append(part_result)
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
