from transformers import pipeline


pipe = pipeline("text-generation", 
                model="/home/yxn/causalmath/models--Anjie6--sft-qwen-7b/snapshots/1", 
                torch_dtype="auto", 
                device_map="auto",
                temperature=1)

def rollout_function(context): # rollout单独的节点
    """
    Generate multiple variants of a node.
    """

    context_text = " ".join(context) if context else ""
    prompt = (f"Given the context: '{context_text}', suggest three possible next steps in the reasoning chain.\n"
    f"Only output the next step, do not continue reasoning further.")
    outputs = pipe(prompt, max_new_tokens=100, num_return_sequences=3, do_sample=True)
    # 修改为明确返回三个one-step节点
    return [output["generated_text"] for output in outputs]


def llm_predict_function(chain): # 按照上下文推理直到最终答案
    """
    Predict the final result based on the given chain.
    """

    chain_text = " ".join(chain)
    prompt = (
        f"Continue the reasoning chain starting from:\n'{chain_text}'\n"
        f"Provide the next steps in the reasoning chain only. Do not include any explanations, commentary, or unrelated content.\n"
        f"Output strictly the reasoning steps, one step at a time, and continue until the correct final answer is reached, ensuring all steps are covered.")

    output = pipe(prompt, max_new_tokens=200, num_return_sequences=1, do_sample=False)
    return output[0]["generated_text"].strip()

if __name__ == "__main__":
    context = "2+4-5+7=?\n 6-5+7=?\n"
    result = rollout_function(context)
    print(result)



