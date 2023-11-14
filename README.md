## Title and Authors
"Content Order-Controllable MR-to-text"

_Authors: Keisuke Toyama, Katsuhito Sudoh, and Satoshi Nakamura_

## Development Environment
- OS
  + Ubuntu 20.04
- Python
  + 3.8.10
- Required Python libraries
  + torch
  + nltk
  + itertools

## Usage
1) get E2E refined dataset
```
$ cd corpus
$ ./EXE-GET-DATASET.sh
```
2) train NLU/NLG models
```
$ cd ../../training
$ ./EXE-TRAINING-NLU.sh
$ ./EXE-TRAINING-NLG.sh
```
3) generate augmented training data
```
$ cd ../corpus/augmentation/train
$ ./EXE-AUGMENT-DATA.sh
```
4) generate augmented(reordered) test data
```
$ cd ../test
$ ./EXE-MAKE-REORDERED-DATA.sh
```
5) train NLU/NLG models w/ augmented training data
```
$ cd ../../../training
$ ./EXE-TRAINING-NLU-AUG.sh
$ ./EXE-TRAINING-NLG-AUG.sh
```
6) evaluate models
```
$ cd ../evaluation
$ ./EXE-inference-NLU.sh
$ ./EXE-inference-NLG.sh
$ ./EXE-MRcheck-NLU.sh
$ ./EXE-MRcheck-NLG.sh
$ ./EXE-e2e-metrics.sh
```
## Contact
- Keisuke Toyama (toyama.keisuke.tb5@is.naist.jp)

## Reference
- PyTorch Seq2Seq (https://github.com/bentrevett/pytorch-seq2seq)

## License
- E2E refined dataset is distributed under the [Creative Common 4.0 Attribution-ShareAlike License (CC4.0-BY-SA)](https://creativecommons.org/licenses/by-sa/4.0/)
