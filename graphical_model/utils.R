library(igraph)

rotate <- function(df, degree) {
  dfr <- df
  degree <- pi * degree / 180
  l <- sqrt(df[, 1]^2 + df[, 2]^2)
  teta <- atan(df[, 2] / df[, 1])
  dfr[, 1] <- round(l * cos(teta - degree))
  dfr[, 2] <- round(l * sin(teta - degree))
  return(dfr)
}

get_graphlet_num <- function(graphlet_adj) {
  # source("reference_graphlets")
  graphlet_adj = graphlet_adj*1
  clique_size = dim(graphlet_adj)[1]
  reference_cliques = reference.cliques[[paste0("g", clique_size, sep="")]]
  for(clique_name in names(reference_cliques)){
    cliq_adj = reference_cliques[[clique_name]]
    num_matching_cols = 0
    for(graphlet_colnum in 1:dim(graphlet_adj)[1]){
      col_sums = colSums(graphlet_adj[,graphlet_colnum]==cliq_adj)
      if(max(col_sums) == clique_size){
        num_matching_cols = num_matching_cols + 1
      }
    }
    if(num_matching_cols == clique_size){
      return(clique_name)
    }
  }
  return("No matching graphlet")
}

# graph_metrics <- function(g) {
#   V(g)$degree <- degree(g)
#   metrics <- data.frame(
#     degree(g), # calculate degree of each node
#     betweenness(g), # calculate betweenness centrality of each node
#     closeness(g), # calculate closeness centrality of each node
#     eigen_centrality(g)$vector, # calculate eigenvector centrality of each node
#     transitivity(g), # calculate clustering coefficient of each node
#     tri_count(g), # calculate number of triangles each node participates in
#     diameter(g), # calculate diameter of the graph
#     clusters(g)$no, # calculate number of connected components in the graph
#     max(clusters(g)$csize), # calculate the size of the largest connected component
#     # assortativity_degree(g) # calculate the assortativity coefficient of the graph
#     edge_density(g) # calculate the graph density
#   )
#   return(metrics)
# }
ms <- 2
adjacency <- (binary_edges != 0)*1
idx_keep <- colSums(binary_edges)>0
adjacency <- adjacency[idx_keep, idx_keep]
sg <- list(
  "1" <- list(),
  "2" <- list()
  )

return_next_nodes <- function(adj, n){
  column = adj[n, ]
  return(which(column == 1))
}

for(n in 1:dim(adjacency)[1]){
  
  adjacency_loop = adjacency[n:dim(adjacency)[1], n:dim(adjacency)[1]]
  next_nodes = return_next_nodes(adjacency_loop, n)
  sg_n = list(rep(n, length(next_nodes)))
}

find_routes <- function(matrix, n, start_node, current_node, route, visited, routes=0) {
  if(routes==0){
    routes <- lapply(1:n, function(i) list())
  }
  route_len <- length(route)
  if (route_len <= n) {
    # Print the complete route
    routes[[route_len]] = append(routes[[route_len]], route)
    # cat("Route:", paste(route, collapse = " -> "), "\n")
  } else {
    for (next_node in 1:nrow(matrix)) {
      if (matrix[current_node, next_node] == 1 && !next_node %in% visited) {
        # Move to the next node and continue the route
        find_routes(matrix, n, start_node, next_node, c(route, next_node), c(visited, next_node), routes)
        
        # Break the loop if we have reached the starting node
        if (next_node == start_node)
          next
      }
    }
  }
  return(routes)
}

adjacency_matrix <- matrix(c(0, 1, 1, 0, 0,
                            1, 0, 1, 1, 0,
                            1, 1, 0, 0, 1,
                            0, 1, 0, 0, 1,
                            0, 0, 1, 1, 0), nrow = 5, ncol = 5)

plot(graph.adjacency(adjacency_matrix, weighted = TRUE, mode = "undirected"))