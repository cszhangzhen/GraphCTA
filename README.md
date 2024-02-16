# GraphCTA
Collaborate to Adapt: Source-Free Graph Domain Adaptation via Bi-directional Adaptation (WWW 2024)

![](https://github.com/cszhangzhen/GraphCTA/blob/main/fig/model.png)

This is a PyTorch implementation of the GraphCTA algorithm, which tries to address the domain adaptation problem without accessing the labelled source graph. It performs model adaptation and graph adaptation collaboratively through a series of procedures: (1) conduct model adaptation based on node's neighborhood predictions in target graph considering both local and global information; (2) perform graph adaptation by updating graph structure and node attributes via neighborhood constrastive learning; and (3) the updated graph serves as an input to facilitate the subsequent iteration of model adaptation, thereby establishing a collaborative loop between model adaptation and graph adaptation.


## Requirements
* python3.8
* pytorch==2.0.0
* torch-scatter==2.1.1+pt20cu118
* torch-sparse==0.6.17+pt20cu118
* torch-cluster==1.6.1+pt20cu118
* torch-geometric==2.3.1
* numpy==1.24.3
* scipy==1.10.1
* tqdm==4.65.0

## Datasets
Datasets used in the paper are all publicly available datasets. You can find [Elliptic](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set), [Twitch](https://github.com/benedekrozemberczki/datasets#twitch-social-networks) and [Citation](https://github.com/yuntaodu/ASN/tree/main/data) via the links.

## Quick Start:
Just execuate the following command for source model pre-training:
```
python train_source.py
```
Then, execuate the following command for adaptation:
```
python train_target.py
```
