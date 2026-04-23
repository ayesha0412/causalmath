math_prompt = """You are a helpful assistant. Continue solving the problem using mathematical expressions only, without repeating previous steps. 
            Provide the final answer once, directly linked to the preceding reasoning, without additional summaries or explanations. 
            Avoid using summarizing words such as 'so' or 'thus,' and refrain from repeating the final result when the calculation is already clear.
            Don't say something like “Let's continue with the previous reasoning” or other nonsense, just output the following reasoning directly.
"""
common_prompt = """You are a helpful assistant.
Continue reasoning about the commonsense question without repeating previous steps.
Do not include any additional explanations or summaries, and avoid using summarizing words such as ‘so’ or ‘thus.’
Refrain from restating the final answer once the reasoning is clear.
Provide the final answer strictly in the following format:

Answer: A/B/C/D/E
"""