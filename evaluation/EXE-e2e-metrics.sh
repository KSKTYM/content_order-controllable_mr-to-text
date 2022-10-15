#! /bin/bash

git clone https://github.com/tuetschek/e2e-metrics

## calculate correct data
mkdir -p e2e-metrics-data
python3 m_e2e_metrics_make_reference.py -i ../corpus/e2e_refined_dataset/e2e_test.json -d_o e2e-metrics-data > e2e-metrics-data/result.txt

## Transformer w/o order
python3 m_e2e_metrics_make_target.py -i result/NLG_O/inference_test.tsv -d_o result/NLG_O
python3 e2e-metrics/measure_scores.py e2e-metrics-data/reference_value.txt result/NLG_O/target.txt > result/NLG_O/RESULT_value.txt
python3 e2e-metrics/measure_scores.py e2e-metrics-data/reference_order.txt result/NLG_O/target.txt > result/NLG_O/RESULT_order.txt

## Transformer w/ order
python3 m_e2e_metrics_make_target.py -i result/NLG_A/inference_test.tsv -d_o result/NLG_A
python3 e2e-metrics/measure_scores.py e2e-metrics-data/reference_value.txt result/NLG_A/target.txt > result/NLG_A/RESULT_value.txt
python3 e2e-metrics/measure_scores.py e2e-metrics-data/reference_order.txt result/NLG_A/target.txt > result/NLG_A/RESULT_order.txt

## Transformer w/ order (augmented training data)
python3 m_e2e_metrics_make_target.py -i result/NLG_A_AUG/inference_test.tsv -d_o result/NLG_A_AUG
python3 e2e-metrics/measure_scores.py e2e-metrics-data/reference_value.txt result/NLG_A_AUG/target.txt > result/NLG_A_AUG/RESULT_value.txt
python3 e2e-metrics/measure_scores.py e2e-metrics-data/reference_order.txt result/NLG_A_AUG/target.txt > result/NLG_A_AUG/RESULT_order.txt
