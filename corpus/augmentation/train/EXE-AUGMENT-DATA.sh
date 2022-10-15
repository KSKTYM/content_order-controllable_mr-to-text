#! /bin/bash

num_v=(16)

mkdir -p out data

## MR augmentation
for v in ${num_v[@]}; do
    python3 m_augment_mr.py -d_i ../../e2e_refined_dataset -nv "$v" -d_data data
done

## MR-Text pairs augmentation
for v in ${num_v[@]}; do
    python3 m_augment_txt.py -f_train ../../e2e_refined_dataset/e2e_train.json -d_model_nlg ../../../output/NLG_A -d_model_nlu ../../../output/NLU_A -alg_nlg A -alg_nlu A -nv "$v" -d_data data
done

## merge original and augmentated training data
for v in ${num_v[@]}; do
    python3 m_augment_merge.py -f_train ../../e2e_refined_dataset/e2e_train.json -nv "$v" -d_data data -d_out out -n 1
done
