#!/bin/bash

# Define the dataset names (add more as needed)
datasets=("gsm8k")
# Base data directory
input_file="$base_dir/$dataset/test_gsm8k_answered.jsonl"
np_output="$base_dir/$dataset/gsm8k_pns_np.jsonl"

# Set PYTHONPATH
export PYTHONPATH="$(pwd)/../algo"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    input_file="$base_dir/$dataset/test_r1_answered_aime25.jsonl"
    
    if [[ -f "$input_file" ]]; then
        echo "Processing dataset '$dataset'..."

        # Output paths
        np_output="$base_dir/$dataset/r1_aime25_original_pn_np.jsonl"
        pb_output="$base_dir/$dataset/r1_aime25_original_pn_pb.jsonl"

        # Get number of lines processed
        np_lines=0
        pb_lines=0

        if [[ -f "$np_output" ]]; then
            np_lines=$(wc -l < "$np_output")
        fi

        if [[ -f "$pb_output" ]]; then
            pb_lines=$(wc -l < "$pb_output")
        fi

        # Run non-prompt version
        echo "Starting non-prompt version..."
        python run_algo_o.py \
            --input_file "$input_file" \
            --output_file "$np_output" \
            --batch_size 20 \
            --lines_processed "$np_lines" \
            --append &

        np_pid=$!

        # Uncomment to run prompt-based version
        # echo "Starting prompt-based version..."
        # python run_algo.py \
        #     --input_file "$input_file" \
        #     --output_file "$pb_output" \
        #     --batch_size 20 \
        #     --lines_processed "$pb_lines" \
        #     --prompt_based \
        #     --append &

        # pb_pid=$!

        # Wait for jobs
        [[ -n "$np_pid" ]] && wait "$np_pid"
        [[ -n "$pb_pid" ]] && wait "$pb_pid"

    else
        echo "⚠️  Warning: $input_file does not exist, skipping dataset '$dataset'..."
    fi
done

echo "✅ Processing complete."