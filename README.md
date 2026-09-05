[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22374729.svg)](https://doi.org/10.5281/zenodo.22374729)
# Mizo Named Entity Recognition

A silver-standard named entity corpus for Mizo (`lus`), a Tibeto-Burman
language of northeast India, with recognizers trained on it and an
evaluation of entity markup for Mizo--English machine translation.

Accompanies the paper *Named Entity Recognition for Mizo: A
Silver-Standard Corpus via Cross-Lingual Projection and Its Effect on
Machine Translation*.

## What is here

| Resource | Size | Where |
|---|---|---|
| Silver corpus | 441,178 sentences, 590,655 entities, 11 types | [dataset](https://huggingface.co/datasets/haulai/mizo-ner) |
| Gold evaluation set | 300 sentences, 420 entities, kappa 0.798 | [dataset](https://huggingface.co/datasets/haulai/mizo-ner) |
| NER models | XLM-RoBERTa, MizBERT, mBERT | [models](https://huggingface.co/haulai) |
| MT models | MarianMT, plain and entity-marked source | [models](https://huggingface.co/haulai) |
| Code | notebooks reproducing every table | this repository |

Start with [`quickstart.ipynb`](quickstart.ipynb).

## Headline numbers

| | |
|---|---:|
| Projection recovery, exact matching | 54.1% |
| Projection recovery, suffix-aware | 66.7% |
| Best recognizer, silver test set (micro F1) | 0.8810 |
| Best recognizer, human gold set (F1) | 0.6414 |
| **Projection accuracy against human gold** | **0.6146** |
| Entity markup for MT, oracle | +0.33 BLEU |
| Entity markup for MT, recognizer predictions | -0.09 BLEU |

**Read the bold row before using the corpus.** Labels are
silver-standard. Measured against human annotation the projection is
61.5% accurate, so evaluating a model on the corpus itself overstates
accuracy by roughly 0.26 F1. The corpus is large, not clean.

The projection is a good entity *detector* and a poor entity
*classifier*: it locates 88% of gold entities but mistypes 31% of them.

## Reproducing the paper

```bash
git clone [https://github.com/thangkhanhau/mizo-ner](https://github.com/thangkhanhau/mizo-ner)
cd mizo-ner
pip install -r requirements.txt
python -m spacy download en_core_web_sm