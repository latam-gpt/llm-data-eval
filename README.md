# Automatic High Quality Data Filtering

We have limited compute, and too much data.

This repo presents a methodogy that allows us to build a smaller but higher quality NLP datasets from a big corpus.

## Filtering data at scale

In an ideal world, we would like to inspect the quality of the samples by hand, but this is impossible given our dataset size. Another approach could be to use an LLM to filter samples given some instructions, but sadly these models are also very expensive to run at scale. Therefore, we need to develop a proxy method:

First, we use an LLM to grade the educational quality of various samples from our original dataset. Then, we use this samples to train a classifier, using a encoder-based model, so that it learns to assign a score from 0 to 5. Since this model is way cheaper to use than an LLM, we can run it in scale over our entire dataset, and thus get the high quality section form it.

## Installation

Using conda:

```terminal
conda create -m llm-data-eval python=3.12
conda activate llm-data-eval
```

## Pipeline Overview

The whole pipeline is a three step process, each contained in its own script.


### GPT-based rating

The entire logic of this step is contained in `rate_with_llms.py`.

The script rates a small subsection of our dataset using an LLM. The dataset is given a base prompt to guide its evaluation, see the reference prompt in `utils/prompts.txt`.

Here is an usage example:

```bash
python score_dataset.py \
  --prompt_path prompts/base_prompt.txt \
  --dataset_path data/es_clean_shuffled/data-00000-of-00512.arrow \
  --model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --download_dir /workspace1/models \
  --output_path results/generated_texts.json \
  --tensor_parallel_size 8 \
  --batch_size 500 \
  --text_column texto
```

### Train encoder

This step is contained in `train_bert_eval.py`.

After running `rate_with_llms.py`, we now have a dataset with prompts, and its quality scores. These samples are now used to train a classifier, using an encoder model as backbone. The model is frozen, and we only add a linear layer over it, wich is used to learn to asses ratings.

In our original experiment, we rated 550K prompts with Llama 3.1, using 500k for training and 50k for validation. Then, we used `Snowflake/snowflake-arctic-embed-m` as the base enconder.

Usage (note that wandb is optional):
```bash
python train_model.py \
  --checkpoint_dir /path/to/checkpoints \
  --base_model Snowflake/snowflake-arctic-embed-m \
  --json_file path/to/generated_texts.json \
  --train_batch_size 2048 \
  --eval_batch_size 1024 \
  --cache_dir ./cache \
  --use_wandb \
  --wandb_project "bert_history_eval" \
  --wandb_entity "your_entity_name"
```

### Evaluate corpus with encoder

The final step is contained in `run_bert_eval.py`.

We have a light quality classifier, so now is time to run it over the entire dataset.

```bash
python run_bert_eval.py \
  --model_path /path/to/pre-trained/bert \
  --dataset /path/to/dataset \
  --output_dir /path/to/output
```

### Filter results

After the rating is done, the corpus will be saved with two new columns `score` and `int_score`, where the first is a continual value, and the second the rounded result, both ranging from 1 to 5. This can be used to filter by quality (5 being the highest quality).

To help in the filtering process, we added the `filter_results.py` script:

```bash
python utils/filter_results.py \
--dataset_path=path/to/my/dataset \
--score=3 \
--num_proc=12 \
--output_path=path/to/store/filtered_dataset
```

## References

- This pipeline is a custom implementation on the approach used in [FineWeb-Edu](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1).
