#! /bin/bash

mkdir -p result

# Transformer w/o order
mkdir -p result/NLU_O
python3 m_inference_NLU.py -d_model ../output/NLU_O -i ../corpus/e2e_refined_dataset/e2e_test.json -o result/NLU_O/inference_test.tsv -alg O

# Transformer w/ order
mkdir -p result/NLU_A
python3 m_inference_NLU.py -d_model ../output/NLU_A -i ../corpus/e2e_refined_dataset/e2e_test.json -o result/NLU_A/inference_test.tsv -alg A

# Transformer w/ order (augmented training data)
mkdir -p result/NLU_A_AUG
python3 m_inference_NLU.py -d_model ../output/NLU_A_AUG -i ../corpus/e2e_refined_dataset/e2e_test.json -o result/NLU_A_AUG/inference_test.tsv -alg A
