#!/bin/bash

# Define the dataset names
datasets=("AIME" "gsm8k" "MATH-500" "commonsenseqa")  # Add more dataset names as needed

# Base data directory
base_dir="data"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    input_file="$base_dir/$dataset/test.jsonl"
    output_file="$base_dir/$dataset/test_incontext_qwen72b_baseline.jsonl"

    # Check if the input file exists before running the command
    if [[ -f "$input_file" ]]; then
        echo "Processing $input_file..."
        python data/get_response.py "$input_file" "$output_file"
    else
        echo "Warning: $input_file does not exist, skipping..."
    fi
done

echo "Processing complete."
