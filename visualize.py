from graphviz import Digraph
from IPython.display import Image
from graphviz import Digraph
from IPython.display import Image
import networkx as nx

def plot_causal_graph(causal_graph, file_name):
    """
    Plots the causal graph using the graphviz Digraph format, saves it as a PNG,
    and returns the image for display using IPython's Image.

    :param causal_graph: The causal graph as a networkx DiGraph.
    :param file_name: The file name to save the image.
    :return: The image for display using IPython's Image.
    """
    # Create a directed graph (Digraph)
    dot = Digraph(format="png")
    
    # Add nodes to the graph
    for node_id, node_data in causal_graph.nodes(data=True):
        # Extract the formula and value for labeling
        formula = node_data.get("formula", "")
        value = node_data.get("value", "")
        label = f'{formula}\nValue: {value}'  # Label to include formula and value
        dot.node(str(node_id), label=label)
    
    # Add edges to the graph
    for source, target, edge_data in causal_graph.edges(data=True):
        relationship = edge_data.get("relationship_type", "Unknown")
        dot.edge(str(source), str(target), label=relationship)
    
    # Save the graph as a PNG file
    dot.render(file_name, cleanup=True)
    
    # Return the image for display
    return Image(filename=f"{file_name}.png")