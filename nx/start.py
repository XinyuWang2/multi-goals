import networkx as nx
import random

# G = nx.DiGraph()
G = nx.Graph() # undirected graphs
G.add_edge("A", "B", weight=4)
G.add_edge("B", "D", weight=2)
G.add_edge("A", "C", weight=3)
G.add_edge("C", "D", weight=4)
p = nx.shortest_path(G, "A", "D", weight="weight") #['A', 'B', 'D']

# G.nodes, G.edges
sampled_nodes = random.sample(G.nodes, 2)
sampled_graph = G.subgraph(sampled_nodes)

x = G["A"]
print(x)

# [n for n in G.neighbors(0)]

# nx.single_source_dijkstra_path_length(tree, starting_state)

