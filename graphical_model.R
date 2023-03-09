# Begin Exclude Linting
# Sys.setenv(RETICULATE_PYTHON = "/home/qxz1djt/.local/share/virtualenvs/aip.mlops.terraform.modules-iNbkyG8C/bin/python")
# End Exclude Linting
library(reticulate)
library(igraph)
py_config()
setwd(paste0(getwd(), "/level-sets"))

source_python("level_sets/utils.py")
source_python("graphical_model/utils.py")
source("graphical_model/utils.R")
img <- load_image("../mnist/img_45.jpg") # , list(10,10))

gm <- graphical_model(
  img,
  TRUE,
  0.8,
  normalize_gray = TRUE
)
nodes <- gm[1]
edges <- gm[2]
spp <- gm[3]

edges_matrix <- matrix(unlist(edges), ncol = dim(nodes[[1]])[1], byrow = TRUE)
edges_matrix_cut <- (edges_matrix > 0.5) * edges_matrix

g <- graph.adjacency(edges_matrix_cut, weighted = TRUE, mode = "undirected")
g <- set_vertex_attr(g, "intensity", index = V(g), spp[[1]][, 4])

coordinates <- as.matrix(rotate(as.data.frame(spp[[1]][, 2:3]), 90))
coordinates[is.nan(coordinates)] <- 0

g <- set.vertex.attribute(g, "x_coordinate", index = V(g), coordinates[, 1])
g <- set.vertex.attribute(g, "y_coordinate", index = V(g), coordinates[, 2])

plot(
  g,
  vertex.size = 5, # replace(spp[[1]][,5],spp[[1]][,5] %in% c(272, 102),20),
  vertex.color = gray(V(g)$intensity),
  layout = cbind(V(g)$x_coordinate, V(g)$y_coordinate)
)

isolated <- which(degree(g) <= 6)
g2 <- delete.vertices(g, isolated)
isolated <- which(degree(g2) <= 6)
g2 <- delete.vertices(g2, isolated)

plot(
  g2,
  vertex.size = 5, # replace(spp[[1]][,5],spp[[1]][,5] %in% c(272, 102),20),
  vertex.color = gray(V(g2)$intensity),
  layout = cbind(V(g2)$x_coordinate, V(g2)$y_coordinate)
)
