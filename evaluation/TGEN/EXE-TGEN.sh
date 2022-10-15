#! /bin/bash

mkdir -p ../result/NLG_TGEN

# convert tgen output to inference.tsv
python3 m_tgen_convert.py -c ../../corpus/e2e_refined_dataset/e2e_test.json -t outputs-test.txt -o ../result/NLG_TGEN/inference_test.tsv

# MRcheck
cd ..
python3 m_MRcheck_NLG.py -d_model_nlu ../output/NLU_A_AUG -i result/NLG_TGEN/inference_test.tsv -o result/NLG_TGEN/inference_test_MRcheck.tsv -alg_nlg A -alg_nlu A

# e2e-metrics
python3 m_e2e_metrics_make_target.py -i result/NLG_TGEN/inference_test.tsv -d_o result/NLG_TGEN
python3 e2e-metrics/measure_scores.py e2e-metrics-data/reference_value.txt result/NLG_TGEN/target.txt > result/NLG_TGEN/RESULT_value.txt
python3 e2e-metrics/measure_scores.py e2e-metrics-data/reference_order.txt result/NLG_TGEN/target.txt > result/NLG_TGEN/RESULT_order.txt
