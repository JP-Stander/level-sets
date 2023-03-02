# Sys.setenv(RETICULATE_PYTHON = "/home/qxz1djt/.local/share/virtualenvs/aip.mlops.terraform.modules-iNbkyG8C/bin/python")
library(reticulate)
py_config()
setwd(paste0(getwd(),"/level-sets"))

source_python("level_sets/utils.py")
source_python("graphical_model/utils.py")
img = load_image("../mnist/img_56.jpg")#, list(10,10)

gm <- graphical_model(img, TRUE)
nodes = gm[[1]]
edges = gm[[2]]
spp = gm[[3]]

edges_matrix = matrix(unlist(edges), ncol = dim(vertices[[1]])[1], byrow = TRUE)
edges_matrix_cut = (edges_matrix > 0.01)*edges_matrix
graph_obj = graph_from_adjacency_matrix(edges_matrix_cut)
g  <- graph.adjacency(edges_matrix_cut, weighted=TRUE, mode="undirected")
plot(g, vertex.size=5)
spp[[1]]