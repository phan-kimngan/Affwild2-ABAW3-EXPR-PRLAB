
Affwild2-ABAW3 @ CVPR 2022
Task: EXPRESION CLASSIFICATION

Name: Kim Ngan Ngan, Hong Hai Nguyen, Van Thong Huynh, Soo Huyng Kim

Paper: Expression Classification using Concatenation of Deep Neural Network for the 3rd ABAW3 Competition

How to train?

    1. Create a python environment using conda or other tools.
    
    conda create -n new_env python=3.6
    
    2. Instead packages in requirements.txt
    
    pip install -r requirements.txt
    
    3. Activate new_env
    
    conda activate new_env
    
    4.
    - Create dataset folder that contains cropped_aligned folder and 3rd_ABAW_Annotations folder
    
    - Run data_preparation.py in tools to create .npy in out data folder
    
    python data_preparation.py --root_dir path/to/dataset-folder --out_dir path/to/out-data-folder
    
    - Edit .yaml file in conf with

        OUT_DIR: path to save tmp file
        
        DATA_DIR: path to .npy file
        
        MODEL_NAME: choose one in {combine, no_att_trans, only_att, only_trans} types

    python main.py --cfg ./conf/EXPR_baseline.yaml


How to predict?

    4.
    - Create batch_1_2_folder that contains entire videos in batch_1 and batch_2 folder
    
    - Create testset_folder that contains EXPR_test_set_release.txt file
    
    - Run prepare_test_data.py in tools file to create EXPR_test.npy in out-data-folder

    python prepare_test_data.py --root_video_dir path/to/batch_1-2-folder --dataset_dir path/to/out-data-folder

    5.
    #- Edit config file in /path/to/config-yaml-file with TEST_ONLY:  path/to/checkpoints-ckpt-file
    
    - Get prediction file of test set at prediction folder

    python main.py --cfg --cfg /path/to/config-yaml-file


