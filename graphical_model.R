Sys.setenv(
  RETICULATE_PYTHON = "/home/qxz1djt/.local/share/virtualenvs/aip.mlops.terraform.modules-iNbkyG8C/bin/python"
)

library(reticulate)
library(igraph)
library(stringr)
py_config()
setwd(paste0(getwd(), "/level-sets"))

# for (pkg in c("pandas", "Pillow", "scikit-image", "scikit-learn", "opencv-python")) {
#   py_install(pkg)
# }

source_python("level_sets/utils.py")
source_python("graphical_model/utils.py")
source("graphical_model/utils.R")
source("graphical_model/reference_graphs.R")
img <- load_image("data/img_16.jpg") # , list(10, 10))

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
binary_edges <- edges_matrix > 0.5

keep_idx <- which(colSums(edges_matrix > 0.5) > 1)
edges_matrix <- edges_matrix[keep_idx, keep_idx]
binary_edges <- binary_edges[keep_idx, keep_idx]

edges_matrix_cut <- (binary_edges) * edges_matrix

g <- graph.adjacency(edges_matrix_cut, weighted = TRUE, mode = "undirected")
g <- set_vertex_attr(g, "intensity", index = V(g), spp[[1]][keep_idx, 4])

coordinates <- as.matrix(rotate(as.data.frame(spp[[1]][keep_idx, 2:3]), 90))
coordinates[is.nan(coordinates)] <- 0

g <- set.vertex.attribute(g, "x_coordinate", index = V(g), coordinates[, 1])
g <- set.vertex.attribute(g, "y_coordinate", index = V(g), coordinates[, 2])

plot(
  g,
  vertex.size = 5, # replace(spp[[1]][,5],spp[[1]][,5] %in% c(272, 102),20),
  vertex.color = gray(V(g)$intensity),
  layout = cbind(V(g)$x_coordinate, V(g)$y_coordinate)
)

graphlets <- graphlet_basis(g)
col_names <- list()
for (n in c(2, 3, 4, 5)) {
  for (name in names(eval(parse(text = paste("cliques.size.", n, sep = ""))))) {
    col_names <- append(col_names, str_replace(name, "g", paste("g", n, "_", sep = "")))
  }
}
rect_data <- setNames(data.frame(matrix(ncol = length(col_names), nrow = 0)), col_names)

for (n in c(2, 3, 4, 5)) {
  cliques_size_n <- Filter(function(x) length(x) == n, graphlets$cliques)
  for (cliq in cliques_size_n) {
    cliq_adj <- binary_edges[cliq, cliq]
    graplet <- get_graphlet_num(cliq_adj, cliques_size_n)
    rect_data[1, graplet] <- rect_data[1, graplet] + 1
  }
}
g <- set.vertex.attribute(g, "n_graphlets", index = V(g), 0)
for (c in seq_along(graphlets$cliques)) {
  g <- set.vertex.attribute(g, "color_code_gl", index = graphlets$cliques[[c]], c)
  V(g)[graphlets$cliques[[c]]]$n_graphlets <- V(g)[graphlets$cliques[[c]]]$n_graphlets + 1
}

cliquess <- cliques(g) # graphlet_basis(g)
g <- set.vertex.attribute(g, "n_cliques", index = V(g), 0)
for (c in seq_along(cliquess)) {
  g <- set.vertex.attribute(g, "color_code_cl", index = cliquess[[c]], c)
  V(g)[cliquess[[c]]]$n_cliques <- V(g)[cliquess[[c]]]$n_cliques + 1
}

plot(
  g,
  vertex.size = ceiling(V(g)$n_graphlets / 20),
  # vertex.size = ceiling(V(g)$n_cliques),
  # vertex.color = V(g)$color_code_gl,
  vertex.color = gray(V(g)$intensity),
  # vertex.color = V(g)$color_code_cl,
  # vertex.frame.color = V(g)$color_code_cl,
  # layout = cbind(V(g)$x_coordinate, V(g)$y_coordinate)
)

plot(
  g,
  # vertex.size = ceiling(V(g)$n_graphlets),
  vertex.size = ceiling(V(g)$n_cliques / 500),
  # vertex.color = V(g)$color_code_gl,
  # vertex.color = V(g)$color_code_cl,
  vertex.color = gray(V(g)$intensity),
  # layout = cbind(V(g)$x_coordinate, V(g)$y_coordinate)
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
