# %%
from tqdm import tqdm
import numpy as np
from os import listdir, cpu_count
from networkx import Graph, read_graphml, write_graphml
from level_sets.metrics import width_to_height, get_angle
from concurrent.futures import ProcessPoolExecutor, as_completed

# %%

def enrich_graph(graph_path):
    g = read_graphml(graph_path)
    for node, data in g.nodes(data=True):
        pixels = [a for a in eval(data.get("pixel_indices"))] if ")," in data.get("pixel_indices") else [eval(data.get("pixel_indices"))]
        img_size = max(max(a[0] for a in pixels), max(a[1] for a in pixels))
        img = np.zeros((img_size+1, img_size+1))
        rows, cols = zip(*pixels)
        img[rows, cols] = 1
        w_t_h = width_to_height(img)
        angl = get_angle(img)
        g.nodes[node]["width_to_height"] = w_t_h
        g.nodes[node]["angle"] = angl
    write_graphml(g, graph_path)

# %%

if __name__ == '__main__':
    all_graphs = [f"../graphical_models/level_sets/{file}" for file in listdir("../graphical_models/level_sets", )]
    all_graphs += [f"../graphical_models/fuzzy_sets_10/{file}" for file in listdir("../graphical_models/fuzzy_sets_10", )]
    all_graphs += [f"../graphical_models/fuzzy_sets_30/{file}" for file in listdir("../graphical_models/fuzzy_sets_30", )]


    n_cores = cpu_count() - 2
    print(f'Running process on {n_cores} cores')
    with ProcessPoolExecutor() as executor:
        # Setup tqdm
        future_to_graph = {executor.submit(enrich_graph, graph_path): graph_path for graph_path in all_graphs}
        for future in tqdm(as_completed(future_to_graph), total=len(all_graphs)):
            try:
                future.result()  # retrieve results if there are any
            except Exception as e:
                print(f"Error processing graph: {future_to_graph[future]}. Error: {e}")


