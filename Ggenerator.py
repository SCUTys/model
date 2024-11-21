import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Read the CSV file
csv_net_path = 'data/SF/SF_net.csv'
edge_data = pd.read_csv(csv_net_path, usecols=['init_node', 'term_node'])

# Create a directed graph
G = nx.DiGraph()

# Add edges to the graph
for index, row in edge_data.iterrows():
    G.add_edge(row['init_node'], row['term_node'])

# Draw the graph
pos = nx.spring_layout(G, k=0.5)  # Adjust the k parameter to spread out nodes
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold')

# Show the plot
plt.show()