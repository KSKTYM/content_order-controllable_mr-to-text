#! /bin/bash

python3 m_MRcheck_NLU.py -i result/NLU_O/inference_test.tsv -o result/NLU_O/inference_test_MRcheck.tsv -alg O
python3 m_MRcheck_NLU.py -i result/NLU_A/inference_test.tsv -o result/NLU_A/inference_test_MRcheck.tsv -alg A
python3 m_MRcheck_NLU.py -i result/NLU_A_AUG/inference_test.tsv -o result/NLU_A_AUG/inference_test_MRcheck.tsv -alg A
