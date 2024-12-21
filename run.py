import json
from transformers import pipeline
from openai import AzureOpenAI
from get_dag import  convert_gpt_output_to_json_and_dag, extract_json_from_gpt_output
from prompt import reasoning_prompt, q_prompt, get_dag_prompt
from visualize import plot_causal_graph

# Load the model
pipe = pipeline("text-generation", model="/home/yxn/causalmath/models--Anjie6--sft-qwen-7b/snapshots/1", torch_dtype="auto", device_map="auto")

# Prepare input messages for reasoning step
messages = [
    {"role": "system", "content": reasoning_prompt},
    {"role": "assistant", "content": "Now, I will think step by step, starting by analyzing the problem and breaking it down."},
    {"role": "user", "content": q_prompt}
]

# Get reasoning steps (with adjusted token limit to avoid truncation)
reasoning_steps = (pipe(messages, max_new_tokens=1024)[0]["generated_text"])[3]["content"]
print("Reasoning Steps:\n", reasoning_steps)


client = AzureOpenAI(
        azure_endpoint="https://feng-cloud-openai.openai.azure.com/",
        api_key="e51a662bf2934ff585b9e53b21b7f6c2",
        api_version="2024-02-15-preview"
    )


# Prepare input messages for DAG extraction
messages2 = [
    {"role": "system", "content": get_dag_prompt},
    {"role": "user", "content": reasoning_steps}
]

response = client.chat.completions.create(
            model="gpt-35-turbo",
            messages=messages2,
        )
nodes_and_edges = response.choices[0].message.content

# # Get nodes and edges for the DAG (adjust token limit again)
# nodes_and_edges = pipe(messages2, max_new_tokens=1024)[0]["generated_text"][3]["content"]
print("================================================")
print("Nodes and Edges:\n", nodes_and_edges)

json_str = extract_json_from_gpt_output(nodes_and_edges)

# Get the nodes, edges, and the DAG object
nodes, edges, dag = convert_gpt_output_to_json_and_dag(json_str)


# Print results
print("Nodes:", json.dumps(nodes, indent=2))
print("Edges:", json.dumps(edges, indent=2))
# # Extract the DAG structure
# dag = parse_dag(nodes_and_edges)

print("DAG:\n", dag)

plot_causal_graph(dag, file_name="dag_test")
