import pprint
from typing import Dict, Any

from base_model import gpt_api_caller, llm_predict_gpt35
from equivalent_ans import is_equivalent_answer

def calculate_ps_pn(
    llm_response: str,
    ground_truth: str,
    threshold: float = 0.3,
    reasoning_attempts: int = 3,  # 干预后推理次数
    do_type: int = 0,
    alter_attempts = 5 # 如果节点一样最大重干预次数
) -> Dict[str, Any]:

    # 记录初始CoT的指标
    original_token_length = len(llm_response.split())  # 通过单词计数近似 token 长度
    original_steps = llm_response.split('\n\n')
    original_steps = [step.strip() for step in original_steps if step.strip()] 
    original_ps = 1 if is_equivalent_answer(original_steps[-1], ground_truth) else 0
    original_accuracy = original_ps
    original_step_length = len(original_steps)

    # 分割节点
    nodes = llm_response.split('\n\n')
    nodes = [node.strip() for node in nodes if node.strip()]
    pprint.pprint(nodes)

    final_answer = nodes[-1] if nodes else ""
    ps = 1 if is_equivalent_answer(final_answer, ground_truth) else 0

    pn_values = []
    i = 1  # 从第二步开始（因为第一步是问题）
    while i < len(nodes):
        current_step = nodes[i]
        context_steps = nodes[:i]  # 当前步骤之前的上下文
        print("context_steps ", context_steps)
        context_message = [
            {"role": "system", "content": "You are a helpful assistant. Please continue reasoning to solve the problem without repeating the previous steps. Only provide mathematical expressions, no text explanations or summaries."},
            {"role": "user", "content": '\n\n'.join(context_steps) + '\n\n'}
        ]
        print(f"\n步骤 {i} 的上下文: {context_steps}")
        print(f"尝试干预步骤 {i}: {current_step}")

        y_values = []
        replacement_steps = []

        # 基于上下文生成替换步骤
        if do_type == 1:
            # 基于prompt干预。skip当前meaning
            skip_message =  [
            {"role": "system", "content": "You are a helpful assistant. Please continue reasoning to solve the problem without repeating the previous steps. Only provide mathematical expressions, no text explanations or summaries.    Ensure the next output node does not match the meaning of "+ current_step},
            {"role": "user", "content": '\n\n'.join(context_steps) + '\n\n'}
            ]
            replacement_text = gpt_api_caller(skip_message)
        else:
            # 直接rollout
            replacement_text = gpt_api_caller(context_message)
        # 判断替换步骤的第一个节点是否与原来相似
        for attempt in range(1, alter_attempts + 1):
            replacement_steps = [step.strip() for step in replacement_text.split('\n\n') if step.strip()]
            
            if not is_equivalent_answer(replacement_steps[0], current_step):
                break  # 如果当前步骤与原来不相似，退出循环
            else:
                # 如果相似，尝试重新生成替代文本
                replacement_text = gpt_api_caller(context_message)
                # 如果是最后一次尝试且依然相似。抛出异常
                if attempt == alter_attempts:
                    print(f"达到最大尝试次数（{alter_attempts}），替换步骤仍与原来相似")
                    break
        print(f"生成的替换步骤: {replacement_steps}")            
        # 用替换步骤再次多次推理直到final answer
        eval_nodes_candidates = []

        for eval_attempt in range(reasoning_attempts):
            print(f"评估替换步骤 {i} 的第 {eval_attempt+1} 次推理")
            eval_context_steps = context_steps + [replacement_steps[0]]
            eval_message = [
                {"role": "system", "content": "You are a helpful assistant. Please continue reasoning to solve the problem without repeating the previous steps. Only provide mathematical expressions, no text explanations or summaries."},
                {"role": "user", "content": '\n\n'.join(eval_context_steps)}
            ]
            evaluation_text = gpt_api_caller(eval_message)            
                
            if not evaluation_text:
                print("没有生成替换步骤。")
                eval_nodes_candidates.append("")  # 保证有个空结果
            else:
                eval_nodes_candidates.append(evaluation_text)

            # 提取评估的最终答案
            eval_nodes = [step.strip() for step in evaluation_text.split('\n\n') if step.strip()]
            eval_final_answer = eval_nodes[-1] if eval_nodes else ""
            is_correct = is_equivalent_answer(eval_final_answer, ground_truth)
            y_values.append(1 if is_correct else 0)
            print(f"评估结果: {'正确' if is_correct else '错误'}")

        # 计算平均 Y 值
        average_y = sum(y_values) / len(y_values) if y_values else 0
        pn = 1 - average_y
        print(f"步骤 {i} 的 PN 值: {pn}")

        pn_values.append(pn)

        # 选择最大 Y 值的替换步骤
        if y_values:
            max_y_index = y_values.index(max(y_values))
            max_y_value = y_values[max_y_index]
            print(f"最大 Y 值: {max_y_value}，对应的替换步骤: {eval_nodes_candidates[max_y_index]}")
        else:
            max_y_index = 0
            max_y_value = 0
            print("没有可用的 Y 值。")

        # 判断是否替换步骤
        if pn < threshold:
            # 替换当前步骤及其后的所有步骤
            eval_nodes = [step.strip() for step in eval_nodes_candidates[max_y_index].split('\n\n') if step.strip()]
            nodes = eval_context_steps + eval_nodes
            print(f"更新后的步骤链: {nodes}")
            # 更新i为当前步骤替换后的长度，避免跳过步骤
            i = len(eval_context_steps) + len(eval_nodes)
        else:
            i += 1  # 继续到下一步

    # 重新计算 PS(chain) 基于最终答案（因为可能已更新）
    final_answer = nodes[-1] if nodes else ""
    ps = 1 if is_equivalent_answer(final_answer, ground_truth) else 0

    # # 检查最后几个步骤是否有重复
    # if len(nodes) >= 3:
    #     last_three_steps = nodes[-3:]
    #     if is_equivalent_answer(last_three_steps[0], last_three_steps[1]):
    #         nodes = nodes[:-2]
    #     if len(nodes) >= 2 and is_equivalent_answer(nodes[-2], nodes[-1]):
    #         nodes = nodes[:-1]

    # 计算其他度量
    final_chain = '\n\n'.join(nodes)
    token_length = len(final_chain.split())  # 通过单词计数近似 token 长度
    accuracy = ps
    step_length = len(nodes)
    average_pn = sum(pn_values) / len(pn_values) if pn_values else 0

    results = {
        "original_token_length": original_token_length,
        "original_accuracy": original_accuracy,
        "original_step_length": original_step_length,
        "PS(chain)": ps,
        "token_length": token_length,
        "accuracy": accuracy,
        "step_length": step_length,
        "average_pn": average_pn,
        "final_chain": nodes
    }

    return results

# 示例用法
if __name__ == "__main__":
    llm_response = """
    Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n\n
Alright, let's see. So, Natalia sold clips to 48 of her friends in April, and then in May, she sold half as many clips. I need to find out how many clips she sold altogether in these two months.\n\n

First, in April, she sold 48 clips. That's straightforward. Now, in May, she sold half as many as in April. So, half of 48. Let's think, half of 40 is 20, and half of 8 is 4, so half of 48 is 24. So, in May, she sold 24 clips.\n\n

To find the total, I need to add the clips sold in April and May together. So, 48 plus 24. Let's see, 48 plus 20 is 68, and then plus 4 more is 72. So, altogether, she sold 72 clips in April and May.\n\n

Wait, is there another way to look at this? Maybe I can think of it as April being one part and May being half a part, so together it's one and a half parts. If one part is 48, then one and a half parts would be 48 plus 24, which again is 72. Yeah, that matches what I got before.\n\n

I don't think I'm missing anything here. It seems pretty straightforward. She sold 48 in April, half that in May, which is 24, and together that's 72 clips.
    """

    ground_truth = "72"

    results = calculate_ps_pn(
        llm_response=llm_response,
        ground_truth=ground_truth
    )

    print(f"\n最终步骤链: {results['final_chain']}")
    metrics = {k: v for k, v in results.items() if k != 'final_chain'}
    print(f"度量指标: {metrics}") 