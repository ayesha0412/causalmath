import re
import os, sys

from lightllm_api.llm_api import qwen_api_caller, gpt_api_caller
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from test_api import query_api
# from lightllm_api.llm_api import gpt_api_caller, qwen_api_caller

llm_api = qwen_api_caller # 替换为你的LLM调用函数

def is_equivalent_step(final_answer, ground_truth): # 两步的含义是否一致
    prompt = f"""
    You are an intelligent assistant. Determine if the following two steps are equivalent in meaning:
    Answer 1: {final_answer}
    Answer 2: {ground_truth}
    Respond with "yes" if the answers convey the same meaning or logic, even if they are written differently. Respond with "no" otherwise.
    """ 
    # print('进行答案对比：')
    print("ground_truth:",ground_truth,"final_answer:",final_answer)
    # client = openai.AzureOpenAI(
    #     azure_endpoint="https://feng-cloud-openai.openai.azure.com/",
    #     api_key="e51a662bf2934ff585b9e53b21b7f6c2",
    #     api_version="2024-02-15-preview"
    # )

    # response = client.chat.completions.create(
    #             model="gpt-35-turbo",
    #             messages=[{"role": "system", "content": "You are a helpful assistant."},
    #                 {"role": "user", "content": prompt}],
    #             temperature=0
    #         )
    
    # answer = response.choices[0].message.content.strip().lower()

    answer =llm_api([{"role": "user", "content": prompt}])
        # Define regex patterns for "yes" and "no"
    yes_pattern = r"\byes\b"
    no_pattern = r"\bno\b"

    # Check if "yes" or "no" is in the answer
    if re.search(yes_pattern, answer, re.IGNORECASE):
        return True
    elif re.search(no_pattern, answer, re.IGNORECASE):
        return False
    else:
        return False
def is_equivalent_answer(final_answer, ground_truth): # 结果是否等价
    prompt = f"""
    You are an intelligent assistant. Determine if the following two answers are equivalent in meaning:
    Answer: {final_answer}
    Reference answer: {ground_truth}
    Respond with "yes" if the answers agree on the value of the results or the choice. Respond with "no" otherwise.
    """ 
    # print('进行答案对比：')
    print("ground_truth:",ground_truth,"final_answer:",final_answer)
    # client = openai.AzureOpenAI(
    #     azure_endpoint="https://feng-cloud-openai.openai.azure.com/",
    #     api_key="e51a662bf2934ff585b9e53b21b7f6c2",
    #     api_version="2024-02-15-preview"
    # )

    # response = client.chat.completions.create(
    #             model="gpt-35-turbo",
    #             messages=[{"role": "system", "content": "You are a helpful assistant."},
    #                 {"role": "user", "content": prompt}],
    #             temperature=0
    #         )
    
    # answer = response.choices[0].message.content.strip().lower()
    
    answer = llm_api([{"role": "user", "content": prompt}])
        # Define regex patterns for "yes" and "no"
    yes_pattern = r"\byes\b"
    no_pattern = r"\bno\b"

    # Check if "yes" or "no" is in the answer
    if re.search(yes_pattern, answer, re.IGNORECASE):
        return True
    elif re.search(no_pattern, answer, re.IGNORECASE):
        return False
    else:
        return False