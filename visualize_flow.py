import json
import graphviz

def visualize_flowchart(json_str):
    """
    Visualizes the flowchart from a given JSON string.

    Parameters:
    json_str (str): A JSON string containing 'nodes' and 'edges'. 
                     'nodes' should contain node identifiers as keys and their corresponding values as descriptions.
                     'edges' should contain a list of dictionaries with 'source_node' and 'target_node' keys.
    
    Returns:
    str: The file path to the generated flowchart image.
    """
    # Convert JSON string to a Python dictionary
    data = json.loads(json_str)
    
    # Initialize a new directed graph
    graph = graphviz.Digraph(format='png', engine='dot')

    # Add nodes to the graph
    for node_id, node_value in data['nodes'].items():
        graph.node(node_id, label=node_value)

    # Add edges between nodes
    for edge in data['edges']:
        graph.edge(edge['source_node'], edge['target_node'])

    # Render and save the image
    file_path = 'flow_chart'
    graph.render(file_path)

    return file_path

