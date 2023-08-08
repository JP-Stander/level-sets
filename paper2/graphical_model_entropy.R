library(jsonlite)
library(igraph)
library(tidyverse)

path_to_files <- "../my_graphical_dataset/img_size_30_spatial_euclidean_attr_cityblock"
all_files <- list.files(path_to_files, pattern = "*.json")

edge_entropy <- function(g) {
    if (!is_weighted(g)) {
        stop("Graph is not weighted")
    }
    w <- E(g)$weight
    p <- table(w) / length(w)
    entropy <- -sum(p * log(p))
    return(entropy)
}

spectral_entropy <- function(g) {
    laplacian <- laplacian_matrix(g)
    eigenvalues <- eigen(laplacian)$values
    p <- eigenvalues / sum(eigenvalues)
    entropy <- -sum(p * log(p))
    return(entropy)
}

attribute_entropy <- function(g, attribute) {
    a <- vertex_attr(g, attribute)
    p <- table(a) / length(a)
    entropy <- -sum(p * log(p))
    return(entropy)
}


for (file in all_files[1]){
    # Read the JSON file
    json_data <- jsonlite::fromJSON(paste0(path_to_files, "/", file))

    # Create an edge dataframe
    edges_df <- as.data.frame(json_data$edges) %>%
        mutate(edge = map_chr(edge, ~paste(.x, collapse = ", "))) %>% 
        separate(edge, into = c("from", "to"), sep = ", ", convert = TRUE)


    # Create a node attributes dataframe
    features_df <- as.data.frame(do.call(rbind, json_data$features))
    colnames(features_df) <- c("intensity", "size", "compactness", "elongation")
    features_df$name <- seq_len(nrow(features_df))
    features_df <- features_df[, c("name", "intensity", "size", "compactness", "elongation")]

    # Create a graph
    g <- graph_from_data_frame(d = edges_df, directed = FALSE, vertices = features_df)

}
