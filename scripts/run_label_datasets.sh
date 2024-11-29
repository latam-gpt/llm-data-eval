#!/bin/bash

# Default values for variables
DATASETS_FILE="datasets.txt"
OUTPUT_DIR="./results"
MODEL_PATH="meta-llama/Meta-Llama-3.1-70B-Instruct"
PROMPT_PATH="./base_prompt_label_usecases.txt"
DOWNLOAD_DIR="./models"
CUDA_DEVICES="0,1"
NUM_GPUS="2" # must match the number of CUDA devices
LOG_FILE="error_log.txt"

# Function to display usage information
usage() {
    echo "Usage: $0 [-d DATASETS_FILE] [-o OUTPUT_DIR] [-m MODEL_PATH] [-p PROMPT_PATH] [-c CUDA_DEVICES] [-n NUM_GPUS] [-l DOWNLOAD_DIR]"
    echo
    echo "Options:"
    echo "  -d DATASETS_FILE   Path to the file containing dataset paths to .arrow files (default: datasets.txt)"
    echo "  -o OUTPUT_DIR      Directory to save output files (default: ./results)"
    echo "  -m MODEL_PATH      Path to the model (default: meta-llama/Meta-Llama-3.1-70B-Instruct)"
    echo "  -p PROMPT_PATH     Path to the prompt file (default: ./base_prompt_label_usecases.txt)"
    echo "  -c CUDA_DEVICES    Comma-separated list of CUDA devices (default: 0,1)"
    echo "  -n NUM_GPUS        Degree of tensor parallelism to use when loading and running the model (default: 2)"
    echo "  -l DOWNLOAD_DIR    Directory to download or cache models (default: ./models)"
    exit 1
}

# Parse command-line arguments
while getopts ":d:o:m:p:c:n:l:" opt; do
    case $opt in
        d) DATASETS_FILE="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        m) MODEL_PATH="$OPTARG" ;;
        p) PROMPT_PATH="$OPTARG" ;;
        c) CUDA_DEVICES="$OPTARG" ;;
        n) NUM_GPUS="$OPTARG" ;;
        l) DOWNLOAD_DIR="$OPTARG" ;;
        *) usage ;;
    esac
done

# Ensure the output directory exists
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/error_log.txt"

# Validate the datasets file
if [[ ! -f $DATASETS_FILE ]]; then
    echo "Error: Datasets file '$DATASETS_FILE' not found."
    usage
fi

# Validate NUM_GPUS and CUDA_DEVICES match
IFS=',' read -ra DEVICES <<< "$CUDA_DEVICES"
if [[ ${#DEVICES[@]} -ne $NUM_GPUS ]]; then
    echo "Error: NUM_GPUS ($NUM_GPUS) does not match the number of CUDA devices (${#DEVICES[@]})."
    exit 1
fi

# Read datasets from the input file
mapfile -t datasets < "$DATASETS_FILE"

# Check if datasets were loaded
if [[ ${#datasets[@]} -eq 0 ]]; then
    echo "Error: No datasets found in '$DATASETS_FILE'."
    exit 1
fi

# Initialize counters
processed_count=0
skipped_count=0

# Iterate over the dataset files
for dataset_path in "${datasets[@]}"; do
    # Skip empty lines or comments
    [[ -z "$dataset_path" || "$dataset_path" =~ ^# ]] && continue

    # Extract the dataset name from the file path
    dataset_name=$(basename "$(dirname "$dataset_path")") # e.g., "conicet"

    # Replace underscores with hyphens in dataset_name
    dataset_name=$(echo "$dataset_name" | tr '_' '-')

    # Extract model name
    model_name=$(basename "$MODEL_PATH")

    # Define the output file name
    output_path="${OUTPUT_DIR}/usecasesdomains_${dataset_name}_${model_name}.json"

    # Run the Python script with the dataset_path and output_path
    echo "Processing dataset: $dataset_path -> $output_path"
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python label_with_llms.py \
        --prompt_path "$PROMPT_PATH" \
        --dataset_path "$dataset_path" \
        --model_path "$MODEL_PATH" \
        --download_dir "$DOWNLOAD_DIR" \
        --output_path "$output_path" \
        --tensor_parallel_size "$NUM_GPUS" \
        --batch_size 500 \
        --text_column texto

    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error processing $dataset_path. Skipping..." | tee -a "$LOG_FILE"
        ((skipped_count++))
        continue
    fi

    ((processed_count++))
done

# Summary report
echo "Processing complete."
echo "Processed datasets: $processed_count"
echo "Skipped datasets: $skipped_count"
echo "Errors logged in: $LOG_FILE"
