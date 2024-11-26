#!/bin/bash

# Define the list of dataset files
datasets=(
    "/workspace1/gpt/spanish_dataset/conicet/data-00000-of-00010.arrow"
    "/workspace1/gpt/spanish_dataset/datos_bibliotecas_uc/data-00000-of-00004.arrow"
    "/workspace1/gpt/spanish_dataset/datos_colombia/data-00000-of-00037.arrow"
    "/workspace1/gpt/spanish_dataset/datos_uruguay/data-00000-of-00008.arrow"
    "/workspace1/gpt/spanish_dataset/emol/data-00000-of-00007.arrow"
    "/workspace1/gpt/spanish_dataset/hemeroteca/data-00000-of-00129.arrow"
    "/workspace1/gpt/spanish_dataset/inaoe/data-00000-of-00001.arrow"
    "/workspace1/gpt/spanish_dataset/memoria_virtual/data-00000-of-00001.arrow"
    "/workspace1/gpt/spanish_dataset/red_pajama/data-00000-of-18926.arrow"
    "/workspace1/gpt/spanish_dataset/red_pajama_with_toxicity_labels/data-00000-of-00001.arrow"
    "/workspace1/gpt/spanish_dataset/the_stack/data-00000-of-00951.arrow"
    "/workspace1/gpt/spanish_dataset/tweets/data-00000-of-00731.arrow"
    "/workspace1/gpt/spanish_dataset/tweets_with_toxicity_labels/data-00000-of-00001.arrow"
    "/workspace1/gpt/spanish_dataset/uchile/data-00000-of-00015.arrow"
    "/workspace1/gpt/spanish_dataset/uy22/data-00000-of-00006.arrow"
)

# Output directory
output_dir="/workspace1/omarflorez/code/llm-data-eval/results"
model_path="meta-llama/Meta-Llama-3.1-70B-Instruct"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Iterate over the dataset files
for dataset_path in "${datasets[@]}"; do
    # Extract the dataset name from the file path
    dataset_name=$(basename "$(dirname "$dataset_path")") # e.g., "conicet"

    # Replace underscores with hyphens in dataset_name
    dataset_name=$(echo "$dataset_name" | tr '_' '-')

    # Extract model name
    model_name=$(basename $model_path)

    # Define the output file name
    output_path="${output_dir}/usecasesdomains_${dataset_name}_${model_name}.json"

    # Run the Python script with the dataset_path and output_path
    echo "Processing dataset: $dataset_path -> $output_path"
    CUDA_VISIBLE_DEVICES=6,7 python label_with_llms.py \
        --prompt_path /workspace1/omarflorez/code/llm-data-eval/utils/base_prompt_label_usecases.txt \
        --dataset_path "$dataset_path" \
        --model_path "$model_path"  \
        --download_dir /workspace1/omarflorez/models   \
        --output_path "$output_path"  \
        --tensor_parallel_size 2   \
        --batch_size 100   \
        --text_column texto

    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error processing $dataset_path. Skipping..."
        continue
    fi
done

echo "Processing complete."
