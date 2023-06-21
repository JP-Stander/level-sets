library(igraph)
library(combinat)

rotate <- function(df, degree) {
  dfr <- df
  degree <- pi * degree / 180
  l <- sqrt(df[, 1]^2 + df[, 2]^2)
  teta <- atan(df[, 2] / df[, 1])
  dfr[, 1] <- round(l * cos(teta - degree))
  dfr[, 2] <- round(l * sin(teta - degree))
  return(dfr)
}

get_graphlet_num <- function(graphlet_adj_ref) {
  # source("reference_graphlets")
  graphlet_adj_ref = graphlet_adj_ref*1
  clique_size = dim(graphlet_adj_ref)[1]
  reference_cliques = reference.cliques[[paste0("g", clique_size, sep="")]]
  for(perm in permn(clique_size)){
    graphlet_adj = graphlet_adj_ref[perm, perm]
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


get_graphlets <- function(adjacency, n, current_node, route, visited, routes) {
  route_len <- length(route)
  if (route_len < n) {
    new_graphlet = matrix(sort(route), ncol=route_len)
    if(!any(apply(routes[[route_len]], 1, function(row) identical(row, new_graphlet)))){
      routes[[route_len]] <- rbind(routes[[route_len]], new_graphlet)
    }
    next_nodes <- setdiff(which(adjacency[current_node, ]==1), visited)
    for (next_node in next_nodes) {#[next_nodes>current_node]
      routes <- get_graphlets(adjacency, n, next_node, c(route, next_node), c(visited, next_node), routes)
    }
  } else if (route_len == n) {
    new_graphlet = matrix(sort(route), ncol=route_len)
    if(!any(apply(routes[[route_len]], 1, function(row) identical(row, new_graphlet)))){
      routes[[route_len]] <- rbind(routes[[route_len]], new_graphlet)
    }
      return(routes)
  } else {
    return(routes)
  }
  return(routes)
}

get_all_graphlets <- function(adjacency_matrix, max_graphlet_size){
  routes <- lapply(1:max_graphlet_size, function(i) matrix(,nrow=0,ncol=i))
  for(node in 1:nrow(adjacency_matrix)){
    routes <- get_graphlets(adjacency_matrix,#[node:nrow(adjacency_matrix), node:ncol(adjacency_matrix)], 
      max_graphlet_size, 
      node, node, node, routes
    )
  }
  return(routes)
}

adjacency_matrix <- matrix(c(0, 1, 1, 0, 0,
                            1, 0, 1, 1, 0,
                            1, 1, 0, 0, 1,
                            0, 1, 0, 0, 1,
                            0, 0, 1, 1, 0), nrow = 5, ncol = 5)



th <- FindAllCG(adjacency_matrix, 4)
mn <- get_all_graphlets(adjacency_matrix, 4)
microbenchmark(
  FindAllCG(adjacency_matrix, 4),
  get_all_graphlets(adjacency_matrix, 4)
)
matrix=adjacency_matrix
n=3
start_node=1
current_node=1
route=1
visited=1
routes=0

current_node=next_node
route=c(route, next_node)
visited=c(visited, next_node)

plot(graph.adjacency(adjacency_matrix, weighted = TRUE, mode = "undirected"))
