import json

# Input and output file paths
input_file = 'data/COMMONSENCEQA/qwen_commonsenseqa_after_pn_np.jsonl'
output_file = 'data/COMMONSENCEQA/qwen_commonsenseqa_after_pn_np_indexed.jsonl'

try:
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f_in:
        lines = f_in.readlines()

    # 处理每一行并添加index字段
    new_lines = []
    for index, line in enumerate(lines, start=1):
        try:
            data = json.loads(line)
            data["index"] = index
            new_lines.append(json.dumps(data) + '\n')
        except json.JSONDecodeError:
            print(f"Error decoding line {index}: {line}")

    # Write updated data to output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.writelines(new_lines)

    print(f"Processing complete, results saved to {output_file}")

except FileNotFoundError:
    print(f"Error: File not found {input_file}")
except Exception as e:
    print(f"Unknown error occurred: {e}")
    