import json
import re
import networkx as nx

def convert_gpt_output_to_json_and_dag(gpt_output_str):
    if not gpt_output_str or gpt_output_str.strip() == "":
        raise ValueError("Input string is empty or None")

    try:
        # Step 1: Parse the GPT output string into a Python dictionary (JSON format)
        gpt_output_str = gpt_output_str.strip()

        # Try to decode the string into JSON
        graph_data = json.loads(gpt_output_str)

        # Extract nodes and edges from the parsed data
        nodes = graph_data.get('nodes', {})
        edges = graph_data.get('edges', [])

        # Step 2: Construct the DAG using NetworkX (nx.DiGraph)
        dag = nx.DiGraph()

        # Add nodes to the graph
        for node_id, node_data in nodes.items():
            dag.add_node(node_id, **node_data)

        # Add edges to the graph
        for edge in edges:
            dag.add_edge(edge['source_node'], edge['target_node'], relationship_type=edge['relationship_type'])

        # Step 3: Return the nodes and edges as JSON along with the NetworkX DiGraph
        return nodes, edges, dag

    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")
    
def extract_json_from_gpt_output(gpt_reply):
    # 判断是否包含 ```json 字眼
    if '```json' in gpt_reply:
        # 正则表达式匹配被 ```json 和 ``` 包围的内容
        match = re.search(r'```json(.*?)```', gpt_reply, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()  # 提取并去除两端的空白字符
            return json_str
        else:
            raise ValueError("No JSON content found within the ```json...``` block.")
    else:
        # 如果没有 ```json 字眼，直接返回原始文本或其他处理
        return gpt_reply  # 这里可以进行后续处理，比如返回原始字符串