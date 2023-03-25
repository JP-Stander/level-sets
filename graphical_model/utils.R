
rotate <- function(df, degree) {
  dfr <- df
  degree <- pi * degree / 180
  l <- sqrt(df[,1]^2 + df[,2]^2)
  teta <- atan(df[,2] / df[,1])
  dfr[,1] <- round(l * cos(teta - degree))
  dfr[,2] <- round(l * sin(teta - degree))
  return(dfr)
}

graph_metrics <- function(g){
  V(g)$degree <- degree(g)
  metrics <- data.frame(
    degree(g), # calculate degree of each node
    betweenness(g), # calculate betweenness centrality of each node
    closeness(g), # calculate closeness centrality of each node
    eigen_centrality(g)$vector, # calculate eigenvector centrality of each node
    transitivity(g), # calculate clustering coefficient of each node
    tri_count(g), # calculate number of triangles each node participates in
    diameter(g), # calculate diameter of the graph
    clusters(g)$no, # calculate number of connected components in the graph
    max(clusters(g)$csize), # calculate the size of the largest connected component
    # assortativity_degree(g) # calculate the assortativity coefficient of the graph
    edge_density(g) # calculate the graph density
  )
  return(metrics)
}