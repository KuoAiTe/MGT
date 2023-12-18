import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from netgraph import Graph
import networkx as nx
import os 
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# Sample data (replace this with your actual data)

# Replace 'your_file.json' with the path to your JSON file.
with open('data.json', 'r') as json_file:
    data = json.load(json_file)

def plot_heatmap(row, file):
    plt.clf()
    fig = plt.figure(figsize=(8, 4))
    period_length = len(row.user_tweets)
    data = np.array(row.user_temporal_attention_layer_0)
    print(data.shape)
    # plot a heatmap with annotation
    prefix = ['[CLS]', '[SIAM]']
    ax = sns.heatmap(data, annot=True, vmax=1, cmap=sns.color_palette("light:r", as_cmap=True), cbar_kws = {'orientation':'horizontal', 'fraction': 0.06, 'aspect': 50}, annot_kws={"size": 14}, xticklabels = prefix + [f'T{i}' for i in range(period_length)], yticklabels = prefix + [f'T{i}' for i in range(period_length)])
    
    cax = ax.collections[0].colorbar.ax  # get the ax of the colorbar
    pos = cax.get_position()  # get the original position
    cax.set_position([pos.x0, pos.y0 + 0.05, pos.width * 1.6, pos.height * 0.5])  # set a new position
    cax.set_frame_on(True)
    
    for spine in cax.spines.values():  # show the colorbar frame again
        spine.set(visible=True, lw=.1, edgecolor='black')

    np.random.seed(0)
    sns.set()
    plt.tight_layout()#rect=[-0.01, -0.03, 1.12, 1.03]

    # Save the new figure
    plt.savefig(f'{file}_time.pdf', dpi = 320)
    
def plot_gnn_heatmap(row, file):

    plt.clf()
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(nrows=4, ncols=1)
    fig.set_figwidth(7)
    fig.set_figheight(8)
    for i, (period_id, period_data) in enumerate(row.gat_data.items()):
        row = i // 1
        column = i % 1
        
        adjs = []
        yticklabels = []
        for relation, period_relation_data in period_data.items():
            adj = []
            for layer, layer_data in period_relation_data.items():
                adj.append(layer_data['0']['adj'][0])
                break
            print(layer_data['0'])
            adj = np.mean(adj, axis = 0).tolist()
            adjs.append(adj)
            yticklabels.append(relation.capitalize())
        
        xticklabels = ['User'] + [f'Frnd. {i * len(adj[1:]) + j}' for j, _ in enumerate(adj[1:])]
        sns.heatmap(adjs, cmap=sns.color_palette("light:r", as_cmap=True), annot=True, annot_kws={"size": 12}, xticklabels = xticklabels, yticklabels = yticklabels, linewidth=1.5, ax = axs[row])

        axs[row].set(xlabel=f"", ylabel="Relation")
        axs[row].title.set_text(f'T{period_id}')
    plt.tight_layout()

    # Save the new figure
    plt.savefig(f'{file}_gnn.pdf', dpi = 320)
def plot_graph(row, file):
    period_length = len(row.user_tweets)
    fig, ax = plt.subplots()
    node_to_community = dict()
    node = 0
    partition_sizes = [10, 20, 30, 40]
    g = nx.random_partition_graph(partition_sizes, 0.5, 0.1)
    g = nx.Graph()
    for i in range(len(row.user_tweets)):
        g.add_node(f'user_{i}')
    for i in range(len(row.friend_tweets)):
        print('friend',i, row.friend_tweets[i])
        g.add_node(f'friend_{i}')
        g.add_edge(f'user_{i}', f'friend_{i}')
        

    for community_id, size in enumerate(partition_sizes):
        for _ in range(size):
            node_to_community[node] = community_id
            node += 1
    
    community_to_color = {
        0 : 'tab:blue',
        1 : 'tab:orange',
        2 : 'tab:green',
        3 : 'tab:red',
    }
    node_color = {node: community_to_color[community_id] \
                for node, community_id in node_to_community.items()}
    Graph(g,
        node_color=node_color, # indicates the community each belongs to  
        node_edge_width=0,     # no black border around nodes 
        edge_width=0.1,        # use thin edges, as they carry no information in this visualisation
        edge_alpha=0.5,        # low edge alpha values accentuates bundles as they appear darker than single edges
        node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
        ax=ax,
    )
    
df = pd.DataFrame.from_records(data)
directory_path = Path(os.path.abspath(__file__)).parent / 'figures' 
directory_path.mkdir(parents=True, exist_ok=True)
for row in df.itertuples():
    filename = f'{row.user_id}_{row.labels[0]}'
    print(filename,)
    if  row.original_user_id != '123456789': continue
    print("YES")
    file = directory_path / filename
    plot_heatmap(row, file)
    
    plot_gnn_heatmap(row, file)
    # Color nodes according to their community.
    #plot_graph(row, file)