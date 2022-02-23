# MAL

# Prepare your dataset

The dataset consists of two parts: 1 Original DCM file, 2 The tumor annotation area corresponding to each DCM file (JSON format).

For example:

data_root = '/path/to/your/data'

```json
train.json
{
    "images": [
        {
            "id": 1,
            "file_name": "0001.DCM (should be a relative path for data_root)",
            "width": 384,
            "height": 384
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "patient_id": "patient_id from dcm file, e.g. gqFv9tThLVuxG0SJKGfaWQ==",
            "position": "plane, e.g. axial",
            "category_id": 1,
            "bbox": [
                175.1849710982659,
                229.1907514450867,
                66.76300578034684,
                55.49132947976878
            ]
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "Malignant",
            "supercategory": "defect"
        },
        {
            "id": 0,
            "name": "Benign",
            "supercategory": "defect"
        }
    ]
}
```

The specific example of JSON file is as follows, the absolute path of 0001.DCM is /path/to/your/data/0001.DCM

```json
{
    "images": [
        {
            "id": 1,
            "file_name": "0001.DCM",
            "width": 384,
            "height": 384
        }
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "patient_id": "gqFv9tThLVuxG0SJKGfaWQ==",
            "position": "axial",
            "category_id": 1,
            "bbox": [
                175.1849710982659,
                229.1907514450867,
                66.76300578034684,
                55.49132947976878
            ]
        }
    ],
    "categories": [
        {
            "id": 1,
            "name": "Malignant",
            "supercategory": "defect"
        },
        {
            "id": 0,
            "name": "Benign",
            "supercategory": "defect"
        }
    ]
}
```

you should modify the `config/base/comm_config.py`:

```
train_json = 'to your train.json'
test_json = 'to your test.json'
data_root = 'to your data root'
```

The code will automatically construct the bipartite graph of DCM data in JSON according to patients and their planes, and carry out corresponding training and testing.

# Install the necessary libraries

```
pip install -r requirements.txt
```

# Train

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_train.sh config/11_mg_axi_sag_multibranch_pcloss_tcloss.py 4
```

The checkpoint will be saved ./work_dirs/11_mg_axi_sag_multibranch_pcloss_tcloss/

# Test

```
CUDA_VISIBLE_DEVICES=0,1,2,3 sh dist_test.sh work_dirs/11_mg_axi_sag_multibranch_pcloss_tcloss/11_mg_axi_sag_multibranch_pcloss_tcloss.py  work_dirs/11_mg_axi_sag_multibranch_pcloss_tcloss/latest.pth 4 --metrics accuracy
```