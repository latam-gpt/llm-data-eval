
#!/bin/bash

# Define the base directory where the files are located
base_dir="/workspace1/omarflorez/code/llm-data-eval/results"

# Find all files that contains results
datasets=($(find "$base_dir" -type f -name "usecasesdomains_*.json"))

# Output directory
output_dir="/workspace1/omarflorez/code/llm-data-eval/results/figures"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Iterate over the dataset files
for dataset_path in "${datasets[@]}"; do
    # # Extract the dataset name from the file path
    # dataset_name=$(basename "$(dirname "$dataset_path")") # e.g., "conicet"

    # # Replace underscores with hyphens in dataset_name
    # dataset_name=$(echo "$dataset_name" | tr '_' '-')

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
