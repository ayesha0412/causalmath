import pprint
import openai
from transformers import pipeline


pipe = pipeline("text-generation", 
                model="/home/yxn/causalmath/models--Anjie6--sft-qwen-7b/snapshots/1", 
                torch_dtype="auto", 
                device_map="auto",
                temperature=1)

# def rollout_function(context): # rollout单独的节点
#     """
#     Generate multiple variants of a node.
#     """

#     context_text = " ".join(context) if context else ""

#     prompt = (
#         f"Given the context: '{context_text}', suggest three possible next steps in the reasoning chain.\n"
#         f"Only output one complete reasoning step for each suggestion.\n"
#         f"Ensure each reasoning step is clear, grammatically correct, and ends with a complete statement."
#     )
#     outputs = pipe(prompt, max_new_tokens=100, num_return_sequences=3, do_sample=True)
#     # 修改为明确返回三个one-step节点
#     return [output["generated_text"] for output in outputs]


def llm_predict_function(chain): 
    """
    Predict the final result based on the given chain by continuing the reasoning until the final answer is reached.
    """

    chain_text = " ".join(chain)
    # sys_prompt = """
    #         Please reason carefully and methodically for any user query. Use the format `### Step n: <description>` for each step, incrementing `n` for every logical step. Ensure the logical flow is coherent and avoid skipping critical details.
    #         At the end of your reasoning, present your concise final answer in the format `Final Answer: <your answer>`.

    #         Example:
    #         Question: What is the sum of the first five positive integers?
    #         ### Step 1: Identify the first five positive integers: 1, 2, 3, 4, 5.
    #         ### Step 2: Calculate the sum of these integers: 1 + 2 + 3 + 4 + 5 = 15.
    #         Final Answer: 15"""

    user_prompt = (chain_text)
    

    output = pipe(user_prompt, max_new_tokens=2000, num_return_sequences=1, do_sample=False)
    
    generated_text = output[0]["generated_text"].strip()
    
    steps = generated_text.split('\n')

    print(len(steps))
    return steps


def llm_predict_gpt35(chain): 
    chain_text = " ".join(chain)
    user_prompt = (chain_text)
    
    client = openai.AzureOpenAI(
        azure_endpoint="https://feng-cloud-openai.openai.azure.com/",
        api_key="e51a662bf2934ff585b9e53b21b7f6c2",
        api_version="2024-02-15-preview"
    )

    response = client.chat.completions.create(
                model="gpt-35-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}],
                temperature=1
            )
    
    generated_text = response.choices[0].message.content.strip()
    steps = generated_text.split('\n')
    return steps






if __name__ == "__main__":
    context = "2+4-5+7=?\n ### Step1: 6-5+7=?\n"
    result = llm_predict_gpt35(context)
    pprint.pprint(result)



