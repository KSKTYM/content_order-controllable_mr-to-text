#! /bin/bash

mkdir -p result

mkdir -p result/NLG_O
python3 m_inference_NLG.py -d_model_nlg ../output/NLG_O -d_model_nlu ../output/NLU_O -i ../corpus/e2e_refined_dataset/e2e_test.json -o result/NLG_O/inference_test.tsv -search best -alg_nlg O -alg_nlu O
mkdir -p result/NLG_A
python3 m_inference_NLG.py -d_model_nlg ../output/NLG_A -d_model_nlu ../output/NLU_A -i ../corpus/e2e_refined_dataset/e2e_test.json -o result/NLG_A/inference_test.tsv -search best -alg_nlg A -alg_nlu A
mkdir -p result/NLG_A_AUG
python3 m_inference_NLG.py -d_model_nlg ../output/NLG_A_AUG -d_model_nlu ../output/NLU_A_AUG -i ../corpus/e2e_refined_dataset/e2e_test.json -o result/NLG_A_AUG/inference_test.tsv -search best -alg_nlg A -alg_nlu A

exnum=(0 1 2 3)
for m in ${exnum[@]}; do
    python3 m_inference_NLG.py -d_model_nlg ../output/NLG_A -d_model_nlu ../output/NLU_A -i ../corpus/augmentation/test/out/e2e_test_aug_"$m".json -o result/NLG_A/inference_test_aug_"$m"_out.tsv -search best -alg_nlg A -alg_nlu A -aug
    python3 m_inference_NLG.py -d_model_nlg ../output/NLG_A_AUG -d_model_nlu ../output/NLU_A_AUG -i ../corpus/augmentation/test/out/e2e_test_aug_"$m".json -o result/NLG_A_AUG/inference_test_aug_"$m"_out.tsv -search best -alg_nlg A -alg_nlu A -aug
done
