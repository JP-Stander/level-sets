import numpy as np
from networkx.algorithms import isomorphism
from networkx import from_numpy_array
#%%
_reference_subgraphs = {
    "g2": {
        "g2_1": from_numpy_array(np.array([
            [0, 1],
            [1, 0]
        ]))
    },
    "g3": {
        "g3_1": from_numpy_array(np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])),
        "g3_2": from_numpy_array(np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ]))
    },
    "g4": {
        "g4_1": from_numpy_array(np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0]
        ])),
        "g4_2": from_numpy_array(np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 1, 1, 0]
        ])),
        "g4_3": from_numpy_array(np.array([
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0]
        ])),
        "g4_4": from_numpy_array(np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0]
        ])),
        "g4_5": from_numpy_array(np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0]
        ])),
        "g4_6": from_numpy_array(np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ]))
    },
    "g5": {
        "g5_1": from_numpy_array(np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0]])),
        
        "g5_2": from_numpy_array(np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]])),
        
        'g5_3': from_numpy_array(np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 1],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0]])),
        
        'g5_4': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]])),
        
        'g5_5': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0]])),

        'g5_6': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]])),
        
        'g5_7': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]])),
        
        'g5_8': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0]])),
        
        'g5_9': from_numpy_array(np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 0, 1],
            [0, 0, 0, 1, 0]])),
        
        'g5_10': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1],
            [0, 0, 1, 1, 0]])),
        
        'g5_11': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0]])),
        
        'g5_12': from_numpy_array(np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]])),
        
        'g5_13': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0]])),
        
        'g5_14': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0]])),
        
        'g5_15': from_numpy_array(np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 0, 0, 0]])),
        
        'g5_16': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 1, 0]])),
        
        'g5_17': from_numpy_array(np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]])),
        
        'g5_18': from_numpy_array(np.array([
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [0, 1, 1, 1, 0]])),
        
        'g5_19': from_numpy_array(np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 0]])),
        
        'g5_20': from_numpy_array(np.array([
            [0, 1, 1, 1, 0],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 1, 1, 0]])),
        
        'g5_21': from_numpy_array(np.array([
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0]]))
    }
}

def _count_subgraphs(main_graph, subgraphs):
    """Count the number of unique subgraphs present in the main graph"""
    counts = {}
    for key, graph in subgraphs.items():
        GM = isomorphism.GraphMatcher(main_graph, graph)
        counts[key] = sum(1 for _ in GM.subgraph_isomorphisms_iter())
    return counts

def count_unique_subgraphs(main_graph, max_subgraph_size=5):
    result = {}
    for key, subgraph_dict in _reference_subgraphs.items():
        if int(key[1:]) <= max_subgraph_size:
            result[key] = _count_subgraphs(main_graph, subgraph_dict)
    return result

# For plotting
# import networkx as nx
# import matplotlib.pyplot as plt


# for gs in ['g2', 'g3', 'g4', 'g5']:
#     total_graphs = len(_reference_subgraphs[gs].values())

#     # Determine the size of the matrix
#     cols = int(np.ceil(np.sqrt(total_graphs)))
#     rows = int(np.ceil(total_graphs / cols))

#     # Create the plots
#     fig, axarr = plt.subplots(rows, cols, figsize=(15, 15))

#     # Helper function to safely retrieve an axis (useful when there's only one row or column)
#     def get_axis(r, c):
#         if rows == 1 and cols == 1:
#             return axarr
#         if rows == 1:
#             return axarr[c]
#         if cols == 1:
#             return axarr[r]
#         return axarr[r, c]

#     # Loop and plot
#     i = 0
#     for key, graph in _reference_subgraphs[gs].items():
#         r, c = divmod(i, cols)
#         ax = get_axis(r, c)
#         nx.draw(graph, with_labels=False, ax=ax)
#         ax.set_title(key)
#         i += 1

#     # If there are any unused subplots, turn them off
#     for j in range(i, rows*cols):
#         r, c = divmod(j, cols)
#         ax = get_axis(r, c)
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()
# # %%
# # Assuming the existing imports and data structures

# # This will store node positions for each graph size
# pos_dict = {}

# for gs in ['g2', 'g3', 'g4', 'g5']:
#     # Check if the node positions for this graph size have been calculated
#     sample_graph = list(_reference_subgraphs[gs].values())[0]
#     if gs not in pos_dict:
#         pos_dict[gs] = nx.shell_layout(sample_graph)
    
#     total_graphs = len(_reference_subgraphs[gs].values())
#     cols = int(np.ceil(np.sqrt(total_graphs)))
#     rows = int(np.ceil(total_graphs / cols))

#     fig, axarr = plt.subplots(rows, cols, figsize=(15, 15))

#     def get_axis(r, c):
#         if rows == 1 and cols == 1:
#             return axarr
#         if rows == 1:
#             return axarr[c]
#         if cols == 1:
#             return axarr[r]
#         return axarr[r, c]

#     i = 0
#     for key, graph in _reference_subgraphs[gs].items():
#         r, c = divmod(i, cols)
#         ax = get_axis(r, c)
#         nx.draw(graph, pos=pos_dict[gs], with_labels=False, ax=ax)  # Use the pre-defined positions here
#         ax.set_title(key)
#         i += 1

#     for j in range(i, rows*cols):
#         r, c = divmod(j, cols)
#         ax = get_axis(r, c)
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

# %%
