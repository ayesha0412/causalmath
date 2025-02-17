import json

# 定义函数来筛选符合条件的 JSON 数据
def filter_jsonl_data(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as infile:

        filtered_data = []
        
        # 逐行读取 JSONL 文件
        for line in infile:
            item = json.loads(line.strip())  # 解析每行 JSON 数据
            
            original_token_length = item.get('metrics', {}).get('original_token_length', 0)
            token_length = item.get('metrics', {}).get('token_length', 0)
            accuracy = item.get('metrics', {}).get('accuracy', 0)  # 获取accuracy值
            
            # 如果 original_token_length 为零或者无效数据，则跳过
            if original_token_length == 0 or accuracy != 1:
                continue
            
            # 计算 30% 到 70% 的范围
            lower_bound = original_token_length * 0.30
            upper_bound = original_token_length * 0.70
            
            # 检查 token_length 是否在这个范围内
            if lower_bound <= token_length <= upper_bound:
                filtered_data.append(item)
    
    # 将筛选后的数据逐行写入新的文件
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for item in filtered_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write('\n')  # 每个 JSON 对象占一行

input_file = 'data/MATH-500/test_qwen_with_qwen_rollouts_results_prompt_based.jsonl'  # 输入文件名
output_file = 'MATH-500_filtered_data.jsonl'  # 输出文件名

filter_jsonl_data(input_file, output_file)