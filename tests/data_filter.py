import json

# Define function to filter JSON data according to conditions
def filter_jsonl_data(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as infile:

        filtered_data = []
        
        # Read JSONL file line by line
        for line in infile:
            item = json.loads(line.strip())  # Parse each line of JSON data
            
            original_token_length = item.get('metrics', {}).get('original_token_length', 0)
            token_length = item.get('metrics', {}).get('token_length', 0)
            accuracy = item.get('metrics', {}).get('accuracy', 0)  # 获取accuracy值
            original_accuracy = item.get('metrics', {}).get('original_accuracy', 0)
        
            
            # # 如果 original_token_length 为零或者无效数据，则跳过
            # if original_token_length == 0 or accuracy != 1:
            #     continue
            
            # # 计算 30% 到 70% 的范围
            # lower_bound = original_token_length * 0.30
            # upper_bound = original_token_length * 0.70
            
            # # 检查 token_length 是否在这个范围内
            # if lower_bound <= token_length <= upper_bound:
            #     filtered_data.append(item)
            # 筛选答案只有一个step的数据
            # if token_length == 1:
            #     filtered_data.append(item)
            # 筛选错误回答
            if original_accuracy == 0:
                filtered_data.append(item)
            
    # 将筛选后的数据逐行写入新的文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in filtered_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')  # 每个 JSON 对象占一行

input_file = 'incontext/test_incontext_qwen72b_tune_results_gsm8k.jsonl'  # 输入文件名
output_file = 'incontext_error_gsm8k.jsonl'  # 输出文件名

filter_jsonl_data(input_file, output_file)