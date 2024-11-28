#!/bin/bash

# Define usage function
usage() {
    echo "Usage: $0 -b BASE_DIR -o OUTPUT_DIR"
    echo
    echo "Options:"
    echo "  -b BASE_DIR   Base directory where the result files are located"
    echo "  -o OUTPUT_DIR Output directory where the figures will be saved"
    exit 1
}

# Parse command-line arguments
while getopts ":b:o:" opt; do
    case $opt in
        b) base_dir="$OPTARG" ;;
        o) output_dir="$OPTARG" ;;
        *) usage ;;
    esac
done

# Check required arguments
if [[ -z "$base_dir" || -z "$output_dir" ]]; then
    usage
fi

# Find all files that contain results
datasets=($(find "$base_dir" -type f -name "usecasesdomains_*.json"))

# Ensure the output directory exists
mkdir -p "$output_dir"

# Iterate over the dataset files
for dataset_path in "${datasets[@]}"; do
    # Run the Python script with the dataset_path and output_path
    echo "Processing dataset: $dataset_path -> $output_dir"
    python plot_labeled_results.py \
        --result_path "$dataset_path" \
        --output_dir "$output_dir"

    # Check if the script executed successfully
    if [ $? -ne 0 ]; then
        echo "Error processing $dataset_path. Skipping..."
        continue
    fi
done

# Plot all datasets combined
python plot_labeled_results.py  \
    --result_path "$base_dir"  \
    --output_dir "$output_dir"

echo "Processing complete."
