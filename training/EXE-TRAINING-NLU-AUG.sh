#! /bin/bash

mkdir -p ../output

mkdir -p ../output/NLU_A_AUG
python3 m_training_NLU.py -epoch 100 -f_train ../corpus/augmentation/train/out/e2e_train_aug_16_1.json -f_valid ../corpus/e2e_refined_dataset/e2e_valid.json -f_test ../corpus/e2e_refined_dataset/e2e_test.json -d_model ../output/NLU_A_AUG -alg A > RES_NLU_A_AUG
