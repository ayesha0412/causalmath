from typing import Dict, List
import os, sys
from causalmath.models.ollama_client import cerebras_query      # ← only change from before

def query_api(messages: List[Dict[str, str]], model=None, temperature=0.7) -> str:
    return cerebras_query(messages, model=model, temperature=temperature)


# final_answer = 1
# ground_truth = "answer is 1"
# prompt = f"""
#     You are an intelligent assistant. Determine if the following two answers are equivalent in meaning:
#     Answer 1: {final_answer}
#     Answer 2: {ground_truth}
#     Respond with "yes" if the answers convey the same meaning, even if they are written differently. Respond with "no" otherwise.
#     """ 
# print('进行答案对比：')
# print("ground_truth:",ground_truth,"final_answer:",final_answer)

# answer = query_api([{"role": "user", "content": prompt}])
# print('是否回答正确:',answer)