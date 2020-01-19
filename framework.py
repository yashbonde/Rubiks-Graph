'''
framework.py

This holds our graph for us and has util ops relating to it

07.10.2019 - @yashbonde
'''

from collections import namedtuple

NODES = 'nodes'
EDGES = 'edges'
RECEIVERS = 'receivers'
SENDERS = 'senders'

ALL_FEATURES = [NODES, EDGES, RECEIVERS, SENDERS]

class GraphFrame(namedtuple("GraphFrame", ALL_FEATURES)):
    
"""
What is the most usable form of a graph 

Graph
- node_embeddings [num_node_types, embedding_dim] # float
- edge_embeddings [num_edge_types, embedding_dim] # float
- senders [num_edges,] # int
- receivers [num_edges,] # int
- edges [num_edges,] # edge types int
- graph_splits [num_graphs,]  # number of nodes in each graph int

"""


def get_placeholders_from_graph_lists(graph_lists):
    node_embedding_dim = graph_lists[0]["nodes"].shape[-1]
    edge_embedding_dim = graph_lists[0]["edges"].shape[-1]