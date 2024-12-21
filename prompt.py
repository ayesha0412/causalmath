reasoning_prompt = """You are an expert in computational mathematics with advanced reasoning abilities. Your task is to provide a detailed, step-by-step explanation of how to solve a given mathematical problem. For each step:
1. Provide a clear, concise title that describes the current stage of the mathematical solution process.
2. In each step, include specific mathematical operations, formulas, or numerical values that illustrate the reasoning. Focus on explicit calculations or transformations.
3. Clearly state any assumptions or simplifications you are making, and provide detailed explanations of the mathematical principles involved.
4. At each stage, critically evaluate your method, ensuring all assumptions are justified and the steps are mathematically sound.
5. Explore alternative mathematical approaches where appropriate and compare their effectiveness or relevance to the problem.

Output Format:
Please strictly follow the JSON format, containing the following keys: 'question', 'final_answer', 'reasoning_steps';
'reasoning_steps' should contain subkeys with 'text' and 'children'; 'children' should also contain subkeys with 'text' and 'children', iterating until the final answer is reached.

Key Instructions:
- Use at least three distinct reasoning steps, each focused on explicit mathematical calculations or transformations.
- Acknowledge any limitations in your reasoning as a computational model, and clearly state what you can and cannot do mathematically.
- Proactively explore and evaluate alternative mathematical methods or approaches.
- Quantify the certainty level of each step and the final conclusion with numerical values where applicable.
- Identify potential errors or assumptions that could influence the solution.
- Consider edge cases and exceptions, applying different computational methods or checks to ensure correctness.

Example JSON Output:

{
  "question": "Problem description",
  "final_answer": "198",
  "reasoning_steps": {
    "text": "Problem description.",
    "mc_value": null,
    "children": [
      {
        "text": "Step 1: First step in solving the mathematical problem.",
        "mc_value": 1,
        "children": [
          {
            "text": "Step 2: Second step in solving the mathematical problem.",
            "mc_value": null,
            "children": []
          }
        ]
      }
    ]
  }
}"""

# q_prompt = """
# what is the 1+590*2/4-3/2+2*3+5t? t=4.
# """

# q_prompt = """what is the 5+6*2?"""

q_prompt = """What is the value of $y$ in the equation $\\frac{30}{50}=\\sqrt{\\frac{y}{50}}$?"""

get_dag_prompt = """I will provide you with detailed reasoning steps. Your task is to generate a causal graph based on this data, strictly following these instructions:

1. **Nodes**: Represent distinct elements of the reasoning process, including:
    - Independent variables (initial conditions).
    - Intermediate results (derived or calculated values).
    - The final result or conclusion.
    - Root nodes can be the independent variables present in the problem.
    - There must be at least two nodes. The number of nodes should be appropriately arranged according to the number of reasoning steps in the process.
    
  For each node:
    -	The formula must always contain the specific algebraic expression or the computed value at that step. **The formula cannot be empty**.
    -	For non-root nodes, the variables in the formula are derived from the formulas of their parent nodes and are assigned to a new variable. The variables in the formula are not the IDs of the parent nodes but rather the mathematical variables or expressions previously used.
    -	The formula format should follow this structure: new_variable = formula, where the new_variable must be explicitly shown. The new variable can then be reused directly in later steps whenever needed.
    - At least one of the formula or value must contain a valid entry (i.e., neither can be empty). Each node must always contain meaningful content—either a computable formula or a valid value.
    - If the formula is computable, the value of the node must always have a corresponding value computed from the formula.
    - Constants and numerical values in the formula do not need to come from parent nodes.
    - Each node must always contain three keys: label, formula, and value.

2. **Edges**: Capture the logical relationships between the nodes. These edges should represent dependencies, such as:
    - "Pythagorean Theorem".
    - "add".
	Ensure edges are logically correct and do not represent unnecessary or redundant relationships.
Carefully ensure the correctness of nodes and edges to maintain logical clarity. The causal graph must be free from redundancy, and the dependency relationships between nodes must be clear. A child node should either be derived from its parent node’s expression or constrained by its parent node’s mathematical relationships. Avoid unnecessary edges.


**Output only** the causal graph in **JSON format** with **no additional text, explanation, or commentary**. The output must strictly follow this format:

```json
{
  "nodes": {
    "node1": {"label": "Node 1", "formula": "a = x + y", "value": 10 },
    "node2": {"label": "Node 2", "formula": "b = x * y", "value": 20 },
    "node3": {"label": "Node 3", "formula": "c = a * b", "value": 200 }
    "node4": {"label": "Node 4", "formula": "d = c / 2", "value": 100 }
    //... more nodes...
  },
  "edges": [
    { "source_node": "node1", "target_node": "node3", "relationship_type": "multiply" },
    { "source_node": "node2", "target_node": "node3", "relationship_type": "multiply" }
    { "source_node": "node3", "target_node": "node4", "relationship_type": "divide" }
    //... more edges...
  ]
}
```"""