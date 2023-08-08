library(jsonlite)
library(igraph)
library(tidyverse)

path_to_files <- "../my_graphical_dataset/img_size_50/spatial_euclidean_attr_cityblock/centroid_method_mean/cut_image"
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


for (file in all_files[6]){ #1 and 13
    # Read the JSON file
    json_data <- jsonlite::fromJSON(paste0(path_to_files, "/", file))
    cat(json_data$metadata$`file name`)
    cat("\n")

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

image_matrix <- matrix(NA, nrow = 50, ncol = 50)
for (vertex_name in V(g)$name) {
    # Get the corresponding metadata
    coordinates <- json_data$metadata[[vertex_name]] + 1
    # Convert to matrix form
    coordinates <- matrix(coordinates, ncol = 2, byrow = TRUE)
    # Get the intensity value
    intensity <- V(g)[vertex_name]$intensity
    # Fill in the image_matrix
    image_matrix[coordinates] <- intensity
}

img_matrix <- as.matrix(image_matrix)

# Get dimensions of the matrix (image)
height <- dim(img_matrix)[1]
width <- dim(img_matrix)[2]

# Create an empty graph
g2 <- make_empty_graph(n = height * width, directed = FALSE)

# For each pixel...
for (i in 1:height) {
  for (j in 1:width) {
    # ...get its index in the matrix
    index <- (i - 1) * width + j

    # Set the pixel intensity as a node attribute
    g2 <- set_vertex_attr(g2, "intensity", index = index, value = img_matrix[i, j])

    # Determine the indices of the neighbours
    top <- ifelse(i == 1, NA, (i-2) * width + j)
    bottom <- ifelse(i == height, NA, i * width + j)
    left <- ifelse(j == 1, NA, (i-1) * width + j - 1)
    right <- ifelse(j == width, NA, (i-1) * width + j + 1)
    top_left <- ifelse(i == 1 | j == 1, NA, (i-2) * width + j - 1)
    top_right <- ifelse(i == 1 | j == width, NA, (i-2) * width + j + 1)
    bottom_left <- ifelse(i == height | j == 1, NA, i * width + j - 1)
    bottom_right <- ifelse(i == height | j == width, NA, i * width + j + 1)

    neighbours <- c(top, bottom, left, right, top_left, top_right, bottom_left, bottom_right)
    weights <- c(1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5)

    # Remove NA values
    neighbours <- neighbours[!is.na(neighbours)]
    weights <- weights[!is.na(neighbours)]

    # Add edges to the graph
    for (k in 1:length(neighbours)) {
      g2 <- add_edges(g2, c(index, neighbours[k]))
      g2 <- set_edge_attr(g2, "weight", index = E(g2)[index %--% neighbours[k]], value = weights[k])
    }
  }
}
