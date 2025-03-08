
import numpy as np
from collections import deque
import math
import matplotlib.cm as cm
import pandas as pd
from os.path import join
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from itertools import combinations
import networkx as nx
import gc
import io
from flask import Flask, render_template, request, send_from_directory, url_for
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import imageio
import os
from pathlib import Path

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "static", "gifs") 
app.config["UPLOAD_PNG"] = os.path.join(app.root_path, "static", "png") 

# Create upload folder if it doesn't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["UPLOAD_PNG"], exist_ok=True)

def get_initial_data(name, data_type="normal"):
  
    # *********************
    # Reading and processing raw data from the file
    # *********************

    MIN_WEIGHT = 1e-6

    # initial data

    # preprocessed data with merged free cycles          

    if data_type=="normal":
        data = pd.read_csv(name, dtype=str)
    elif data_type=="uk":
        data = pd.read_csv(name, dtype=str)
        data = data.rename(columns={'company_number': 'organisation_inn'})
    else:
        data = pd.read_feather(name)
        data = data.rename(columns={'company_number': 'organisation_inn', 'index': 'equity_share'})

    data=data[['organisation_inn','participant_id','equity_share']]
    data = data.astype({'equity_share': float, 'participant_id':str, 'organisation_inn':str})
    data=data[~data['equity_share'].isna()]

    # Primary processing

    # remove self-ownership
    data = data[data.participant_id != data.organisation_inn]

    # calculation of normalized shares
    gdata = data.groupby('organisation_inn').sum().reset_index()
    dict_companies = dict(gdata.values)
    data['equity_share'] = data['equity_share']/np.array([dict_companies[num] for num in data['organisation_inn']])

    # delete anomalously low shares
    data = data[data['equity_share'] > MIN_WEIGHT]

    # calculation of normalized shares
    gdata = data.groupby('organisation_inn').sum().reset_index()
    dict_companies = dict(gdata.values)
    data['equity_share'] = data['equity_share']/np.array([dict_companies[num] for num in data['organisation_inn']])

    # assign super_holders and super_targets
    data['super_holder']=~pd.Series(data.participant_id).isin(data.organisation_inn)
    data['super_target']=~pd.Series(data.organisation_inn).isin(data.participant_id)

    return data

def add_subgraph(G, target):
    global raw_data
    try:
        df = raw_data[raw_data.organisation_inn == target]
        ebunch = [(u, target, int(10**max(3,np.ceil(abs(np.log10(w))))*w)/(10**max(3,np.ceil(abs(np.log10(w)))))) for u,w in zip(df.participant_id.values, df.equity_share.values)]
        G.add_weighted_edges_from(ebunch)
        for new_target in df.participant_id.values:

            new_df = raw_data[raw_data.organisation_inn == new_target]
            set1 = set(new_df.participant_id.values)
            set2 = set(G.nodes)
            if set2.intersection(set1) != set1:
                G = add_subgraph(G, new_target)
            else:
                ebunch = [(u, new_target, int(10**max(3,np.ceil(abs(np.log10(w))))*w)/(10**max(3,np.ceil(abs(np.log10(w)))))) for u,w in zip(new_df.participant_id.values, new_df.equity_share.values)]
                G.add_weighted_edges_from(ebunch)


    except KeyError:
        pass

    return G


def draw_net(target, for_orgs=[], figsize=10, with_labels=0, edge_labels=True, rotate=None, cyclic_holders_list=[], classic_holders_list=[]):
    G = nx.DiGraph()
    G = add_subgraph(G, target)

    Grev = G.reverse()
    path_lengths = nx.single_source_shortest_path_length(Grev, target)
    maxlen = max(path_lengths.values())
    nodelist = [[target]] + [[node for node in list(G.nodes()) if path_lengths[node] == i] for i in range(1, maxlen+1)]
    nodecolor = {}

    for n in G.nodes():
        if n == target:
            nodecolor[n] = 'r'
        elif n in for_orgs:
            nodecolor[n] = 'brown'
        elif n in classic_holders_list:
            nodecolor[n] = 'b'
        elif n in cyclic_holders_list:
            nodecolor[n] = 'c'
        else:
            nodecolor[n] = 'gray'

    pos = nx.shell_layout(G, nlist = nodelist, center = (0,0), rotate=rotate)

    return G, pos

def draw_net2(target, png_path, for_orgs=[], figsize=10, with_labels=0, edge_labels=True, rotate=None, cyclic_holders_list=[], classic_holders_list=[]):

    G = nx.DiGraph()
    G = add_subgraph(G, target)

    Grev = G.reverse()


    import matplotlib.pyplot as plt

    path_lengths = nx.single_source_shortest_path_length(Grev, target)
    maxlen = max(path_lengths.values())
    nodelist = [[target]] + [[node for node in list(G.nodes()) if path_lengths[node] == i] for i in range(1, maxlen+1)]

    nodecolor = {}

    for n in G.nodes():
        if n == target:
            nodecolor[n] = 'r'
        elif n in for_orgs:
            nodecolor[n] = 'brown'
        elif n in classic_holders_list:
            nodecolor[n] = 'b'
        elif n in cyclic_holders_list:
            nodecolor[n] = 'c'
        else:
            nodecolor[n] = 'gray'

    fig, ax = plt.subplots(figsize = (figsize,figsize))

    pos = nx.shell_layout(G, nlist = nodelist, center = (0,0), rotate=rotate)
    labels = nx.get_edge_attributes(G,'weight')

    nx.draw(G, pos = pos, with_labels=with_labels, ax = ax, node_color = list(nodecolor.values()), edgelist=[])
    if edge_labels:
        nx.draw_networkx_edge_labels(G,pos, edge_labels=labels, font_size=10)

    nx.draw_networkx_edges(G, pos, width=1, connectionstyle='arc3, rad=0.2')

    e_weights = dict([((n1, n2), d['weight']) for n1, n2, d in G.edges(data=True)])
    #print()
    #_ = nx.draw_networkx_edge_labels(G, pos = pos, edge_labels = e_weights, ax = ax)

    for r in np.arange(0, 1.0, 1.0/(maxlen+1)):
        circle = plt.Circle((0, 0), r, color='k', fill=False, linewidth = 0.5)
        ax.add_patch(circle)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    # Save frame
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()


verbose=False

def reverse_edges(G):
    G_inv = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        G_inv.add_edge(v, u, **data)
    return G_inv




def simulate_random_walk(G, pos, start_node=None, convergence_threshold=1e-8, max_steps=1000, output_gif="random_walk.gif"):
    """
    Simulates a random walk on a directed weighted graph and creates an animation of the probability distribution evolution.
    G (networkx.DiGraph): Directed weighted graph
    pos (dict): Node positions for visualization
    start_node (node, optional): Starting node for the random walk
    convergence_threshold (float): Threshold for convergence detection
    max_steps (int): Maximum number of steps to simulate
    output_gif (str): Output filename for the GIF animation
    """

    path_lengths = nx.single_source_shortest_path_length(G, start_node)

    
    maxlen = max(path_lengths.values())
    # Validate input
    if not G.is_directed():
        raise ValueError("Graph must be directed.")
    if not pos:
        raise ValueError("Position dictionary is required.")
    
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Create transition matrix
    adj_matrix = nx.adjacency_matrix(G, nodelist=nodes, weight='weight').toarray().astype(float)
    row_sums = adj_matrix.sum(axis=1)
    
    # Handle nodes with no outgoing edges by adding self-loops
    zero_rows = np.where(row_sums == 0)[0]
    if zero_rows.size > 0:
        for i in zero_rows:
            adj_matrix[i, i] = 1.0  # Add self-loop
        row_sums = adj_matrix.sum(axis=1)  # Recalculate row sums
    
    transition_matrix = adj_matrix / row_sums[:, np.newaxis]
    
     #  Set initial state
    if start_node is None:
        start_node = nodes[0]
    start_idx = nodes.index(start_node)
    current_state = np.zeros(n)
    current_state[start_idx] = 1.0
    states = [current_state]
    
    # Simulate random walk
    converged = False
    for step in range(max_steps):
        new_state = current_state @ transition_matrix
        states.append(new_state.copy()/(step+1))
        new_state[start_idx] = 1.0
    
        current_state = new_state
        

    print(f"Converged after {len(states)-1} steps" if converged else f"Max steps ({max_steps}) reached")
    
    first_20 = states[:30]
    remaining_list = states[30:]  
    every_5th = remaining_list[::10] 
    result = first_20 + every_5th
    states = result
    # Create animation frames
    cmap = plt.cm.Blues
    norm = Normalize(vmin=0.05, vmax=0.5)
    filenames = []
    frames = []
            
    # Save to a .npz file
    np.savez('states.npz', states)

    for i, state in enumerate(states):
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, node_color=state, cmap=cmap, 
                               vmin=0.05, vmax=0.5, node_size=800, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), edge_color='gray',
                               width=1.5, arrows=True, arrowsize=20, connectionstyle='arc3, rad=0.2', ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12)


        for r in np.arange(0, 1.0, 1.0/(maxlen+1)):
            circle = plt.Circle((0, 0), r, color='k', fill=False, linewidth = 0.5)
            ax.add_patch(circle)

        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        # Add colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, shrink=0.95, label='Probability')
        
        ax.set_title(f"Step {i}", fontsize=14)
        plt.tight_layout()

        # Save frame
        filename = f"temp_frame_{i}.png"
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        filenames.append(filename)
    
    # Create GIF
    with imageio.get_writer(output_gif, mode='I', duration=0.1, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)


global raw_data
year = 2023


def get_current_file_dir():
    """
    Returns the full path of the directory containing the current Python file.
    The path uses forward slashes (/) for compatibility across platforms.
    """
    # Get the absolute path of the current file
    if '__file__' in globals():
        current_file_path = os.path.abspath(__file__)
    else:
        # If __file__ is not available (e.g., in some interactive environments), use sys.argv[0]
        current_file_path = os.path.abspath(sys.argv[0])
    
    # Get the directory of the current file and convert to a Path object
    current_dir = Path(os.path.dirname(current_file_path))
    
    # Convert the path to use forward slashes
    current_dir = current_dir.as_posix()
    
    return current_dir

# Example usage
current_directory = get_current_file_dir()
print("Current file directory:", current_directory)

name = current_directory + f'/russian_organisations_participants_current_pruned_long_10oct24_{year}.csv.gz' 
trans_data = pd.read_csv(current_directory +f"/ru_ultimate_ownership_{year}_pruned_10oct24_transitive_ownership.csv.gz", dtype={'organisation_inn': str, 'participant_id': str})
cycles_df = pd.read_csv(current_directory +f"/ru_ultimate_ownership_{year}_pruned_10oct24_cycles.csv", dtype={'cycle_inn': str, 'cycle_id': str})



"""
# Get current directory
current_dir = os.getcwd()

# Define expected filenames based on the year
file1 = f'russian_organisations_participants_current_pruned_long_10oct24_{year}.csv.gz'
file2 = f'ru_ultimate_ownership_{year}_pruned_10oct24_transitive_ownership.csv.gz'
file3 = f'ru_ultimate_ownership_{year}_pruned_10oct24_cycles.csv'


# Read the files with full path verification
name = pd.read_csv(
    os.path.join(current_dir, file1),
    # Add dtypes here if needed for the first file
)

trans_data = pd.read_csv(
    os.path.join(current_dir, file2),
    dtype={'organisation_inn': str, 'participant_id': str}
)

cycles_df = pd.read_csv(
    os.path.join(current_dir, file3),
    dtype={'cycle_inn': str, 'cycle_id': str}
)

"""

@app.route('/', methods=['GET', 'POST'])
def index():
    global raw_data
    active_section = 'home'
    
    if request.method == 'POST':
        vertex_name = str(request.form['vertex_name'])
        active_section = 'model'
        participants = list(trans_data[trans_data['organisation_inn'] == vertex_name]['participant_id'])
        shares = list(trans_data[trans_data['organisation_inn'] == vertex_name]['share'])
        if not participants:
            participants = ['N/A']
            shares = ['N/A']
        # Check if files already exist
        png_path = os.path.join(app.config['UPLOAD_PNG'], f'{vertex_name}.png')
        gif_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{vertex_name}.gif')
        
        if not (os.path.exists(png_path) and os.path.exists(gif_path)):
            # Only run simulation if files don't exist
            raw_data = get_initial_data(name)
            target = vertex_name
        
            cyclic_holders_list = list(cycles_df[cycles_df['cycle_id'] == '-1'].cycle_inn)
            classic_holders_list = list(trans_data[trans_data['organisation_inn'] == target].participant_id)
            
            G_inv, pos = draw_net(target, for_orgs=[], figsize=10, with_labels=0, 
                                 edge_labels=False, cyclic_holders_list=cyclic_holders_list, 
                                 classic_holders_list=classic_holders_list)
            G = reverse_edges(G_inv)

            draw_net2(target, png_path, for_orgs=[], figsize=10, with_labels=0, 
                    edge_labels=True, rotate=None, cyclic_holders_list=cyclic_holders_list, 
                    classic_holders_list=classic_holders_list)
            simulate_random_walk(G, pos, start_node=target, max_steps=300, output_gif=gif_path)

        return render_template('index.html', 
                            png_path=url_for('static', filename=f"png/{vertex_name}.png"),  
                            gif_path=url_for('static', filename=f"gifs/{vertex_name}.gif"),
                            active_section=active_section,
                            participants=participants,
                            shares=shares )
    
    return render_template('index.html', active_section=active_section)


from jinja2 import Environment, select_autoescape
@app.template_filter('zip')
def zip_lists(a, b):
    return zip(a, b)
if __name__ == '__main__':
    app.run(debug=True)






