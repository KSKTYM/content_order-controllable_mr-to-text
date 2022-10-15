#! /bin/bash

mkdir -p out
python3 m_make_reordered_data.py -np 5 -nv 4 -seed 1234 -d_i ../../e2e_refined_dataset -f_aug ../train/out/e2e_train_aug_16_1.json -d_o out
