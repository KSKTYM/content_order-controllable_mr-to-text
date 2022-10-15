#! /bin/bash

# download E2E refined dataset (distrubited under the CC4.0-BY-SA license)
curl -OL https://github.com/KSKTYM/E2E-refined-dataset/release/e2e_refined_dataset_v1_0_0.zip
unzip e2e_refined_dataset_v1_0_0.zip
mv release/e2e_refined_dataset .
rmdir -fr release
