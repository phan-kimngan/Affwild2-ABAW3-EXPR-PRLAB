
#  Affwild2-ABAW3 @ CVPR 2022
## Task: EXPRESSION CLASSIFICATION

Name: Kim Ngan Ngan, Hong Hai Nguyen, Van Thong Huynh, Soo Huyng Kim

**Paper: Expression Classification using Concatenation of Deep Neural Network for the 3rd ABAW3 Competition**


### Set up environment
+ Create a python environment using conda or other tools.
```bash
conda create -n new_env python=3.6
```
+ Instead packages in requirements.txt
```bash
pip install -r requirements.txt
```
+ Activate new_env
```bash
conda activate new_env
```
### How to train?
+  Create dataset-folder that contains **cropped_aligned** folder and **3rd_ABAW_Annotations folder**

+  Run **data_preparation.py** in **tools** to create .npy file in out-data-folder
```bash
python data_preparation.py --root_dir path/to/dataset-folder --out_dir path/to/out-data-folder
```
+  Edit .yaml file in **conf** with

    OUT_DIR: path to save tmp file

    DATA_DIR: path to .npy file in out-data-folder

    MODEL_NAME: choose one in {combine, no_att_trans, only_att, only_trans} types
```bash
python main.py --cfg ./conf/EXPR_baseline.yaml
```

### How to predict?

+  Create batch-1-2-folder that contains entire videos in batch_1 and batch_2 folder

+  Create testset-folder that contains **EXPR_test_set_release.txt** file

+  Run prepare_test_data.py in **tools** to create EXPR_test.npy in out-data-folder
```bash
python prepare_test_data.py --root_video_dir path/to/batch-1-2-folder --dataset_dir path/to/out-data-folder
```
+  Get prediction file of test set at OUT_DIR
```bash
python main.py --cfg /path/to/config-yaml-file
```

