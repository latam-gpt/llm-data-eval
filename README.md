# Automatic High-Quality Data Filtering

We have limited compute, and too much data.

This repo presents a methodology that allows us to 
- Build a smaller but higher quality NLP datasets from a big corpus using LLMs.
- Perform zero-shot labeling on datasets to ensure coverage of a distribution of domains and use cases where our datasets can effectively contribute to training LLMs tailored for Latin America.

## Index

1. [Installation](#installation)
2. [Filtering data at scale](#filtering-data-at-scale)
3. [Pipeline Overview](#pipeline-overview)
  1. [GPT-based rating](#gpt-based-rating)
  2. [Train encoder](#train-encoder)
  3. [Evaluate corpus with encoder](#evaluate-corpus-with-encoder)
4. [Filter results](#filter-results)
5. [Assessing Data Utility](#assessing-data-utility)
6. [References](#references)


## Installation

Using conda:

```terminal
conda create -m llm-data-eval python=3.12
conda activate llm-data-eval
```

# Filtering data at scale

In an ideal world, we would like to inspect the quality of the samples by hand, but this is impossible given our dataset size. Another approach could be to use an LLM to filter samples given some instructions, but sadly these models are also very expensive to run at scale. Therefore, we need to develop a proxy method:

First, we use an LLM to grade the educational quality of various samples from our original dataset. Then, we use this samples to train a classifier, using a encoder-based model, so that it learns to assign a score from 0 to 5. Since this model is way cheaper to use than an LLM, we can run it in scale over our entire dataset, and thus get the high quality section form it.

## Pipeline Overview

The whole pipeline is a three step process, each contained in its own script.


### GPT-based rating

The entire logic of this step is contained in `rate_with_llms.py`.

The script rates a small subsection of our dataset using an LLM. The dataset is given a base prompt to guide its evaluation, see the reference prompt in `utils/base_prompt.txt`.

Here is an usage example:

```bash
python rate_with_llms.py \
  --prompt_path prompts/base_prompt.txt \
  --dataset_path data/ \
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

# Assessing Data Utility

We aim to evaluate whether the datasets we are collecting will be useful for downstream tasks. By "utility," we mean that people should find these datasets valuable for fine-tuning models pre-trained in the context of Latin American domains and use cases.

To scale this analysis, we leverage an LLM to label a subset of each dataset. This process provides a distribution of domains and use cases, helping us identify areas where we need to collect more data.

## GPT-Based Labeling

We use the script `label_with_llms.py` to label datasets (e.g., in `.arrow` format) using an LLM. The script employs definitions of domains and use cases provided in the prompt file `utils/base_prompt_label_usecases.txt`.

### Example Usage

The following command demonstrates how to label a dataset using the script:

```bash
python label_with_llms.py \
  --prompt_path prompts/base_prompt_label_usecases.txt \
  --dataset_path /path/to/dataset \
  --model_path meta-llama/Meta-Llama-3.1-70B-Instruct \
  --download_dir /path/to/download/models \
  --output_path results/usecasesdomains_dataset.json \
  --tensor_parallel_size 2 \
  --batch_size 500 \
  --text_column texto
```

To label multiple datasets efficiently, we provide a Bash script, `scripts/run_label_datasets.sh`. This script iterates over a list of dataset paths contained in a file and saves the results for each dataset.

### Example Usage

```bash
bash scripts/run_label_datasets.sh \
    -d "/path/to/dataset_paths.txt" \
    -o "/path/to/output_dir" \
    -m "meta-llama/Meta-Llama-3.1-70B-Instruct" \
    -p "/path/to/base_prompt_label_usecases.txt" \
    -c "0,1" \
    -n 2 \
    -l "/path/to/download/models"
```

# References

- This pipeline is a custom implementation on the approach used in [FineWeb-Edu](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1).
