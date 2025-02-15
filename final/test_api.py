from typing import Dict, List
import openai

openai.api_key = "sk-O8VHM4wmKqyidqqYF7201937F6Ea4fC0Ac1713C99dC9A1B6"
openai.base_url = 'https://api.shubiaobiao.cn/v1/'

def query_api( 
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 1.0):
    try:
        print('query_api')
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )
        # print('get message')
        return response.choices[0].message.content.strip()

    except Exception as e:
        return str(e)


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