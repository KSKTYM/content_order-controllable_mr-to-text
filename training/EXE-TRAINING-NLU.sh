#! /bin/bash

mkdir -p ../output

# Transformer w/o order
mkdir -p ../output/NLU_O
python3 m_training_NLU.py -epoch 100 -f_train ../corpus/e2e_refined_dataset/e2e_train.json -f_valid ../corpus/e2e_refined_dataset/e2e_valid.json -f_test ../corpus/e2e_refined_dataset/e2e_test.json -d_model ../output/NLU_O -alg O > RES_NLU_O

# Transformer w/ order
mkdir -p ../output/NLU_A
python3 m_training_NLU.py -epoch 100 -f_train ../corpus/e2e_refined_dataset/e2e_train.json -f_valid ../corpus/e2e_refined_dataset/e2e_valid.json -f_test ../corpus/e2e_refined_dataset/e2e_test.json -d_model ../output/NLU_A -alg A > RES_NLU_A
