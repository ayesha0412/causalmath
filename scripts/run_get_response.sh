# #!/bin/bash

# # Define the dataset names
# datasets=("AIME" "gsm8k" "MATH-500" "commonsenseqa")  # Add more dataset names as needed

# # Base data directory
# base_dir="data"

# # Loop through each dataset
# for dataset in "${datasets[@]}"; do
#     input_file="$base_dir/$dataset/test.jsonl"
#     output_file="$base_dir/$dataset/test_incontext_qwen72b_baseline.jsonl"

#     # Check if the input file exists before running the command
#     if [[ -f "$input_file" ]]; then
#         echo "Processing $input_file..."
#         python data/get_response.py "$input_file" "$output_file"
#     else
#         echo "Warning: $input_file does not exist, skipping..."
#     fi
# done

# echo "Processing complete."

#!/bin/bash
dataset="gsm8k"
base_dir="data"
input_file="$base_dir/$dataset/test.jsonl"
output_file="$base_dir/$dataset/test_gsm8k_answered.jsonl"

# 500 samples only — pipe through head
head -500 "$input_file" > /tmp/gsm8k_500.jsonl

python expt/get_response.py /tmp/gsm8k_500.jsonl "$output_file"
echo "Done → $output_file"
