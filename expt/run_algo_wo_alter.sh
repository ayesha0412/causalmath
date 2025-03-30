#!/bin/bash

# Define the dataset names (add more as needed)
datasets=("commonsense_cot_partial_raw_train")

# Base data directory
base_dir="data"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    # Define the input file path.
    input_file="$base_dir/$dataset/response_0.jsonl"
    
    # Check if the input file exists before running the command.
    if [[ -f "$input_file" ]]; then
        echo "Processing dataset '$dataset'..."

        # For non-prompt output file:
        np_output="$base_dir/$dataset/test_results.jsonl"
        if [[ -f "$np_output" ]]; then
            np_lines=$(wc -l < "$np_output")
        else
            np_lines=0
        fi

        # For prompt-based output file:
        pb_output="$base_dir/$dataset/test_results.jsonl"
        if [[ -f "$pb_output" ]]; then
            pb_lines=$(wc -l < "$pb_output")
        else
            pb_lines=0
        fi

        # Run non-prompt version concurrently (direct rollouts)
        python data/run_algo.py \
            --input_file "$input_file" \
            --output_file "$np_output" \
            --batch_size 20 \
            --lines_processed "$np_lines" &

        # Run prompt-based version concurrently
        python data/run_algo.py \
            --input_file "$input_file" \
            --output_file "$pb_output" \
            --batch_size 20 \
            --lines_processed "$pb_lines" \
            --use_prompt &

        # Wait for both background processes to finish before moving to the next dataset.
        wait
    else
        echo "Warning: $input_file does not exist, skipping dataset '$dataset'..."
    fi
done

echo "Processing complete."
