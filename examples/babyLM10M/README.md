# Training FAT5 on babyLM10M

## Introduction

This example demonstrates how to train a FAT5 model on the babyLM10M dataset.

## Train a tokenizer and pretokenize the babyLM10M dataset

The following script trains a tokenizer on the babyLM10M dataset.

```bash
python train_tokenizer.py
```

following by the full tokenization of the babyLM10M dataset:

```bash
python pretokenize_babyLM10M.py
```

## Train a FAT5 model

The model can be trained with the following command:

```bash
python train_fat5_babyLM.py config/flash-t5-small-babyLM10M.yaml
```
