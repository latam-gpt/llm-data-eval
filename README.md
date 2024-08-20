# Automatic High Quality Data Filtering

We have limited compute, and too much data.

This repo presents a methodogy that allows us to build a smaller but higher quality NLP datasets from a big corpus.

## Filtering data at scale

In an ideal world, we would like to select the samples by hand, but this is impossible given our dataset size. Another approach could be to use an LLM to filter samples given some instructions, but sadly these models are also very expensive to run at scale. Therefore, we need to develop a proxy method:

First, we use an LLM to grade the educational quality of various samples from our original dataset. Then, we use this samples to train a classifier, using a encoder-based model, so that it learns to assign a score from 0 to 5. Since this model is way cheaper to use than an LLM, we can run it in scale over our entire dataset, and thus get the high quality section form it.

## Installation

Using conda:

```terminal
conda create -m llm-data-eval python=3.12
conda activate llm-data-eval
```

## Scripts

- `rate_with_llms.py`: Rates samples of the dataset using an LLM.

- `train_bert_eval.py`: Uses the previous rated samples to train a classifier.

- `run_bert_eval.py`: Runs the classifier over the entire dataset.

```bash
python run_bert_eval.py --model_path /path/to/pre-trained/bert \
    --dataset /path/to/dataset --output_dir /path/to/output
```

## References

- [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
