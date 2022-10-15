#! /bin/bash

mkdir -p result

mkdir -p result/NLG_O
python3 m_MRcheck_NLG.py -d_model_nlu ../output/NLU_A_AUG -i result/NLG_O/inference_test.tsv -o result/NLG_O/inference_test_MRcheck.tsv -alg_nlg A -alg_nlu A
mkdir -p result/NLG_A
python3 m_MRcheck_NLG.py -d_model_nlu ../output/NLU_A_AUG -i result/NLG_A/inference_test.tsv -o result/NLG_A/inference_test_MRcheck.tsv -alg_nlg A -alg_nlu A
mkdir -p result/NLG_A_AUG
python3 m_MRcheck_NLG.py -d_model_nlu ../output/NLU_A_AUG -i result/NLG_A_AUG/inference_test.tsv -o result/NLG_A_AUG/inference_test_MRcheck.tsv -alg_nlg A -alg_nlu A

exnum=(0 1 2 3)
for m in ${exnum[@]}; do
    python3 m_MRcheck_NLG.py -d_model_nlu ../output/NLU_A_AUG -i result/NLG_A/inference_test_aug_"$m".tsv -o result/NLG_A/inference_test_aug_"$m"_MRcheck.tsv -alg_nlg A -alg_nlu A
    python3 m_MRcheck_NLG.py -d_model_nlu ../output/NLU_A_AUG -i result/NLG_A_AUG/inference_test_aug_"$m".tsv -o result/NLG_A_AUG/inference_test_aug_"$m"_MRcheck.tsv -alg_nlg A -alg_nlu A
done
