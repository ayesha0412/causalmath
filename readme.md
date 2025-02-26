# In-Context Learning with LLM API

This file demonstrates how to perform in-context learning using an API. It processes multiple datasets by reading JSONL files, querying the API, and writing the results back to disk.

### Changing the LLM Prompt to do in-context learning

Edit `get_response.py` inside `get_answer_for_line()`:

```python
messages = [
    {"role": "user", "content": "Example Question: What is 2 + 2?\nExample Answer: 4\n\nNow answer this: " + user_query}
]
```
Modify the system message as needed.

## Usage

### Running the Bash Script

1. **Make the Script Executable:**

   ```bash
   chmod +x run_get_response.sh
   ```

2. **Run the Script:**

   ```bash
   ./run_get_response.sh
   ```

The script will process datasets under the `data/` directory, like `AIME`, `gsm8k`, `MATH-500`, and `commonsenseqa`. Each dataset should contain a `test.jsonl` file. The output will be saved to a file named like `test_llama33_answered.jsonl`. Modify the output file name inside run\_get\_response.sh as needed.

## Customizing Output and Model

### Customizing Output File Name

In `run_get_response.sh`, modify:

```bash
output_file="$base_dir/$dataset/test_llama33_answered.jsonl"
```

Change `test_llama33_answered.jsonl` to your desired output file name.


### Changing the Model

The API caller function (e.g., `qwen_api_caller()`) is the model inference API. Replace it with your preferred API if you want to switch models.

## Additional Notes

- Ensure each `test.jsonl` file is a valid JSONL file containing `question` or `problem` fields.
- Test with a small dataset before scaling up.
