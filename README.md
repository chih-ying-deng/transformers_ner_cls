# Transformer_NER

- [Annotation](README.md#processing-annotation-from-label-studio)
- [Input format](README.md#input-format)
- [Run NER](README.md#run-ner)
- [Run Classification](README.md#run-sentence-classification)
- [Run hyperparameters optimization](README.md#run-hyperparameters-optimization)
- [Output](README.md#output)

## Processing annotation from Label-Studio
- [Processing annotation](annotation/README.md)

## Input format
- CoNLL2003 format
```
U.N.         NNP  I-NP  I-ORG 
official     NN   I-NP  O 
Ekeus        NNP  I-NP  I-PER 
heads        VBZ  I-VP  O 
for          IN   I-PP  O 
Baghdad      NNP  I-NP  I-LOC 
.            .    O     O 
```
or replace the third column with start offset of each token
```
-DOCSTART-   _  [DOC_ID]     O
U.N.         _  [START_CHAR] I-ORG 
official     _  [START_CHAR] O 
Ekeus        _  [START_CHAR] I-PER 
heads        _  [START_CHAR] O 
for          _  [START_CHAR] O 
Baghdad      _  [START_CHAR] I-LOC 
.            _  [START_CHAR] O 
```

## Run NER
```
python ner.py \
  --dset data/symptom \
  --seed 123 \
  --lr 6e-5 \
  --decay 0.02 \
  --warmups 500 \
  --eps 1e-8 \
  --n_epochs 100 \
  --model_class electra \
  --pretrained_model google/electra-base-discriminator or MODEL PATH \
```

## Run sentence classification
```
python cls.py \
  --dset data/symptom \
  --cls_type multilabel \
  --seed 123 \
  --lr 6e-5 \
  --decay 0.02 \
  --warmups 500 \
  --eps 1e-8 \
  --n_epochs 100 \
  --model_class bert \
  --pretrained_model bert-base-uncased or MODEL PATH \
```

## Run hyperparameters optimization
```
python optimization.py \
  --dset data/symptom \
  --model electra \
  --type NER
  --lr 1e-6 1e-4 \
  --decay 0.01 0.1 \
  --warmups 0 3000 \
  --eps 1e-9 1e-7 \
  --init_trials 5 \
  --opt_trials 25 
```

## Output
- prediction
```
[DOC_ID]
[start_char] [token] [true_tag] [pred_tag]
```
```
D1 D        O     O
20 U.N.     I-ORG I-ORG 
25 official O     O 
34 Ekeus    I-PER I-PER 
40 heads    O     O 
46 for      O     O 
50 Baghdad  I-LOC I-LOC 
58 .        O     O 
```
- metrics: acc, f1, ppv, sen, etc
- tensorbaord: loss, lr, acc, f1, ppv, sen, etc
- saved model
- optimization output: best para, contour plot, slice plot, cv plot, tradeoff plot, etc

