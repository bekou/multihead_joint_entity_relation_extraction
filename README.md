# Joint entity recognition and relation extraction as a multi-head selection problem

Implementation of the papers
[Joint entity recognition and relation extraction as a multi-head selection problem](https://arxiv.org/abs/1804.07847) and 
[Adversarial training for multi-context joint entity and relation extraction](https://arxiv.org/abs/1808.06876).

# Requirements
* Ubuntu 16.04
* Anaconda 5.0.1
* Numpy 1.14.1
* Tensorflow 1.5.0

## Task
Given a sequence of tokens (i.e., sentence), (i) give the entity tag of each word (e.g., NER) and (ii) the relations between the entities in the sentence. The following example indicates the accepted input format of our multi-head selection model:


```
0	Marc		B-PER		['N']					[0]		
1	Smith		I-PER 		['lives_in','works_for']		[5,11]
2 	lives		O		['N']					[2]
3	in		O		['N']					[3]
4	New		B-LOC		['N']					[4]
5	Orleans		I-LOC		['N']					[5] 
6	and		O		['N']					[6]
7	is		O		['N']					[7]
8	hired		O		['N']					[8]
9	by		O		['N']					[9]
10	the		O		['N']					[10]
11  government		B-ORG		['N']					[11]
12	.		O		['N']					[12]
```

## Configuration
The model has several parameters such as: 
* EC (entity classification) or BIO (BIO encoding scheme)
* Character embeddings
* Ner loss (softmax or CRF)

that could be specified in the configuration files (see [config](https://github.com/bekou/multihead_joint_entity_relation_extraction/tree/master/configs)).

## Run the model

```
./run.sh
```

## More details
Commands executed in ```./run.sh```:

1. Train on the training set and evaluate on the dev set to obtain early stopping epoch
```python3 -u train_es.py```
2. Train on the concatenated (train + dev) set and evaluate on the test set until either the (1) max epochs or (2) the early stopping limit is exceeded after executing train_es.py
```python3 -u train_eval.py```

## Notes

Please cite our work when using this software.

Giannis Bekoulis, Johannes Deleu, Thomas Demeester, Chris Develder. Joint entity recognition and relation extraction as a multi-head selection problem. Expert Systems with Applications, Volume 114, Pages 34-45, 2018

Giannis Bekoulis, Johannes Deleu, Thomas Demeester, Chris Develder. Adversarial training for multi-context joint entity and relation extraction, In the Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018

