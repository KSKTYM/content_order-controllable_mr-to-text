#! /bin/bash

mkdir -p ../output

# Transformer w/o order
mkdir -p ../output/NLG_O
python3 m_training_NLG.py -epoch 100 -f_train ../corpus/e2e_refined_dataset/e2e_train.json -f_valid ../corpus/e2e_refined_dataset/e2e_valid.json -f_test ../corpus/e2e_refined_dataset/e2e_test.json -d_model ../output/NLG_O -alg O > RES_NLG_O

# Transformer w/ order
mkdir -p ../output/NLG_A
python3 m_training_NLG.py -epoch 100 -f_train ../corpus/e2e_refined_dataset/e2e_train.json -f_valid ../corpus/e2e_refined_dataset/e2e_valid.json -f_test ../corpus/e2e_refined_dataset/e2e_test.json -d_model ../output/NLG_A -alg A > RES_NLG_A
