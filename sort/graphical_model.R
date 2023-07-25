library(reticulate)
library(igraph)
library(stringr)
library(rlist)
library(Matrix)
library(ScreenClean)
py_config()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

source_python("level_sets/utils.py")
source_python("graphical_model/utils.py")
source("graphical_model/utils.R")
source("graphical_model/reference_graphs.R")

images = list.files("data/", pattern="*.jpg")

col_names = unlist(lapply(names(reference.cliques), function(x){names(reference.cliques[[x]])}))
col_names = col_names[!startsWith(col_names, "g5")]
rect_data <- setNames(data.frame(matrix(0, ncol = length(col_names), nrow = length(images))), col_names)

start.time <- Sys.time()
for(i in 1:length(images)){

  img <- load_image(paste0("data/", images[i], sep=""), list(10, 10))
  gm <- graphical_model(
    img,
    TRUE,
    0.8,
    normalize_gray = TRUE
  )
  nodes <- gm[1]
  edges <- gm[2]
  # spp <- gm[3]

  edges_matrix <- matrix(unlist(edges), ncol = dim(nodes[[1]])[1], byrow = TRUE)
  binary_edges <- edges_matrix > 0.5

  # keep_idx <- which(colSums(edges_matrix > 0.5) > 1)
  # edges_matrix <- edges_matrix[keep_idx, keep_idx]
  # binary_edges <- binary_edges[keep_idx, keep_idx]

  # edges_matrix_cut <- (binary_edges) * edges_matrix
  # sparse_adjacency <- as(edges_matrix_cut, "sparseMatrix")

  # g <- graph.adjacency(edges_matrix_cut, weighted = TRUE, mode = "undirected")
  # g <- set_vertex_attr(g, "intensity", index = V(g), spp[[1]][, 4])

  # coordinates <- as.matrix(rotate(as.data.frame(spp[[1]][, 2:3]), 90))
  # coordinates[is.nan(coordinates)] <- 0

  # g <- set.vertex.attribute(g, "x_coordinate", index = V(g), coordinates[, 1])
  # g <- set.vertex.attribute(g, "y_coordinate", index = V(g), coordinates[, 2])

  # plot(
  #   g,
  #   vertex.size = 5, # replace(spp[[1]][,5],spp[[1]][,5] %in% c(272, 102),20),
  #   vertex.color = gray(V(g)$intensity),
  #   layout = cbind(V(g)$x_coordinate, V(g)$y_coordinate)
  # )
  # break
  # graphlets <- graphlet_basis(g)
  # all_cliques <- cliques(g)
  subgraphs <- FindAllCG(Matrix(binary_edges, sparse = TRUE), 4)
#
  # for(clique in Filter(function(x) length(x) %in% c(5), graphlets$cliques)){
  for(k in 2:length(subgraphs)){
    subgraph = subgraphs[[k]]
    for(j in 1:dim(subgraph)[1]){
      clique = subgraph[j,]
      # if(!all(colSums(binary_edges[clique, clique]) > 0)) next  
      clique_name = get_graphlet_num(binary_edges[clique, clique])
      rect_data[i, clique_name] = rect_data[i, clique_name] + 1
    }
  }
}

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

library(microbenchmark)
microbenchmark(
  a=FindAllCG(Matrix(binary_edges, sparse = TRUE), 4),
  b=FindAllCG(binary_edges, 4),
  times=10
)

# g <- set.vertex.attribute(g, "n_graphlets", index = V(g), 0)
# for (c in seq_along(graphlets$cliques)) {
#   g <- set.vertex.attribute(g, "color_code_gl", index = graphlets$cliques[[c]], c)
#   V(g)[graphlets$cliques[[c]]]$n_graphlets <- V(g)[graphlets$cliques[[c]]]$n_graphlets + 1
# }

# cliquess <- cliques(g) # graphlet_basis(g)
# g <- set.vertex.attribute(g, "n_cliques", index = V(g), 0)
# for (c in seq_along(cliquess)) {
#   g <- set.vertex.attribute(g, "color_code_cl", index = cliquess[[c]], c)
#   V(g)[cliquess[[c]]]$n_cliques <- V(g)[cliquess[[c]]]$n_cliques + 1
# }

# plot(
#   g,
#   vertex.size = ceiling(V(g)$n_graphlets / 20),
#   # vertex.size = ceiling(V(g)$n_cliques),
#   # vertex.color = V(g)$color_code_gl,
#   vertex.color = gray(V(g)$intensity),
#   # vertex.color = V(g)$color_code_cl,
#   # vertex.frame.color = V(g)$color_code_cl,
#   # layout = cbind(V(g)$x_coordinate, V(g)$y_coordinate)
# )

# plot(
#   g,
#   # vertex.size = ceiling(V(g)$n_graphlets),
#   vertex.size = ceiling(V(g)$n_cliques / 500),
#   # vertex.color = V(g)$color_code_gl,
#   # vertex.color = V(g)$color_code_cl,
#   vertex.color = gray(V(g)$intensity),
#   # layout = cbind(V(g)$x_coordinate, V(g)$y_coordinate)
# )

# isolated <- which(degree(g) <= 6)
# g2 <- delete.vertices(g, isolated)
# isolated <- which(degree(g2) <= 6)
# g2 <- delete.vertices(g2, isolated)

# plot(
#   g2,
#   vertex.size = 5, # replace(spp[[1]][,5],spp[[1]][,5] %in% c(272, 102),20),
#   vertex.color = gray(V(g2)$intensity),
#   layout = cbind(V(g2)$x_coordinate, V(g2)$y_coordinate)
# )
