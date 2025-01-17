# import openai

import re
import os, sys

from base_model import gpt_api_caller
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from lightllm_api.llm_api import gpt_api_caller

def is_equivalent_answer(final_answer, ground_truth):
    prompt = f"""
    You are an intelligent assistant. Determine if the following two answers are equivalent in meaning:
    Answer 1: {final_answer}
    Answer 2: {ground_truth}
    Respond with "yes" if the answers convey the same meaning, even if they are written differently. Respond with "no" otherwise.
    """ 
    print('进行答案对比：')
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
    answer = gpt_api_caller([{"role": "user", "content": prompt}])
    print('是否回答正确:',answer)
    pattern = r"^(yes|no)$"
    return bool(re.match(pattern, answer.strip(), re.IGNORECASE))