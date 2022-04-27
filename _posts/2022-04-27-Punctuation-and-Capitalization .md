---
title: 'Punctuation and Capitalization'
date: 2022-04-27
permalink: /posts/2022/04/Punctuation-and-Capitalization/
tags:
  - Punctuation and Capitalization
  - Catalan Punctuation
  - Catalan Capitalization
  - Catalan Speech
  - Catalan
  - Catalan Speech To Text
  - Catalan ASR
  - Catalan Speech DataSet
  - NeMo Punctuation
  - Catalan Speech To Text
  - Catalan ASR
  - Catalan Tacotron2
image: logan-armstrong-hVhfqhDYciU-unsplash.jpg
---

This Repo Contains Implementation and explanation of Punctuation and Capitalization System for ASR models

![patrick-tomasso-Oaqk7qqNh_c-unsplash](https://user-images.githubusercontent.com/53477752/165603578-ed8d1003-f513-4412-aaf9-f488fa9dabcb.jpg)

## Introduction

Almost all automatic speech recognition(ASR) systems convert speech into text that has no capitalization or puntuation, which can result in miss understanding the generated tex. In this blog I expplain and implement capitalization or puntuation model with [Roberta](https://huggingface.co/PlanTL-GOB-ES/roberta-base-ca) language model for Catalan language. This tutorial is mainly based on Nvidia Nemo tutorial on capitalization or puntuation model [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#training-punctuation-and-capitalization-model).

## Language Model Based Capitalization and Puntuation model
- This model predicts if a sentence needs commas, periods, question marks, ...
- Also model predicts if a given word should be Capitelized.

As in [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#training-punctuation-and-capitalization-model) this model (this method) is a jointly training two token-level classifier on top of a pretrained language model.

## Data Format

The Punctuation and Capitalization model expects the data in the following format:

The training and evaluation data is divided into 2 files:  ``text.txt`` , ``labels.txt``

Each line of the ``text.txt`` file contains text sequences, where words are separated with spaces.

[WORD] [SPACE] [WORD] [SPACE] [WORD], for example:

`` when is the next flight to new york
the next flight is ...
...
``

The `labels.txt` file contains corresponding labels for each word in `text.txt`, the labels are separated with spaces. Each label in `labels.txt` file consists of 2 symbols:

the first symbol of the label indicates what punctuation mark should follow the word (where O means no punctuation needed)

the second symbol determines if a word needs to be capitalized or not (where U indicates that the word should be upper cased, and O - no capitalization needed)

By default, the following punctuation marks are considered: commas, periods, and question marks; the remaining punctuation marks were removed from the data. This can be changed by introducing new labels in the `labels.txt` files.

Each line of the `labels.txt` should follow the format: [LABEL] [SPACE] [LABEL] [SPACE] [LABEL] (for `labels.txt`). For example, labels for the above `text.txt` file should be:

`OU OO OO OO OO OO OU ?U`
`OU OO OO OO ...`
