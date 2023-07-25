# Import needed R libraries
library(reticulate)
library(Matrix)
library(reshape2)
library(ScreenClean)
library(node2vec)
py_config()
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Import needed python functions
source_python("images/utils.py")
source_python("graphical_model/utils.py")

# Import needed R functions
source("graphical_model/utils.R")
source("graphical_model/reference_graphs.R")

# List images in database
n_images <- 100
# images <- paste0("dotted/", sample(list.files("../dtd/images/dotted/", pattern = "*.jpg", recursive = TRUE), n_images))
images <- c(paste0("fibrous/", sample(list.files("../dtd/images/fibrous", pattern = "*.jpg", recursive = TRUE), n_images)))
# images <- c(images,
#     "dotted/dotted_0001.jpg",
#     "dotted/dotted_0103.jpg",

#     "fibrous/fibrous_0216.jpg",
#     "fibrous/fibrous_0110.jpg"
# )
num_images <- length(images)

# Number of embeddings for Node2Vec
num_embeddings <- 10
image_size <- 10
q <- 1
p <- 2
num_walks <- 10
walk_length <- 6

# Get all names of graphlets that will be used
col_names_graphlets <- unlist(lapply(names(reference.cliques), function(x) {
    names(reference.cliques[[x]])
}))
col_names_graphlets <- col_names_graphlets[!startsWith(col_names_graphlets, "g5")]

# Name of Node2Vec embeddings
col_names_embeddings <- paste("embedding", 1:num_embeddings, sep = "_")

start_time <- Sys.time()
foreach (i = 1:length(images)) %do% {
    img <- load_image(
        paste0("../dtd/images/", images[i], sep = ""),
        list(image_size, image_size)
    )

    # Create rectangular datasets to save results into
    # Graphlet counts
    graphlet_counts <- setNames(
        data.frame(matrix(0, ncol = length(col_names_graphlets) + 2, nrow = 1)),
        c("image_name", "image_type", col_names_graphlets)
    )

    # Graphlet locations
    subgraphs_coordinates <- data.frame(
        image_name = character(0),
        image_type = character(0),
        graphlet_name = character(0),
        x = numeric(0),
        y = numeric(0)
    )
    image_name <- tools::file_path_sans_ext(basename(images[i]))
    image_type <- strsplit(image_name, "_")[[1]][1]

    graphlet_counts["image_name"] <- image_name
    graphlet_counts["image_type"] <- image_type

    # image(img, col=grey.colors(n=255))
    img <- load_image(
        paste0("../dtd/images/", images[i], sep = ""),
        list(10, 10)
    )
    gm <- graphical_model(
        img,
        TRUE,
        0.7,
        normalize_gray = TRUE
    )
    nodes <- gm[1]
    edges <- gm[2]
    attr <- gm[3][[1]]

    edges_matrix <- matrix(unlist(edges), ncol = dim(nodes[[1]])[1], byrow = TRUE)
    binary_edges <- edges_matrix > 0.5

    g <- graph_from_adjacency_matrix(binary_edges, mode = "undirected")

    # Plot the graph
    plot(g)

    # Subgraph approach
    subgraphs <- FindAllCG(Matrix(binary_edges, sparse = TRUE), 4)
    for (k in 2:length(subgraphs)){
        subgraph <- subgraphs[[k]]
        for (j in seq_len(dim(subgraph)[1])){
            clique <- subgraph[j, ]
            clique_name <- get_graphlet_num(binary_edges[clique, clique])
            graphlet_counts[i, clique_name] <- graphlet_counts[i, clique_name] + 1

            subgraphs_coordinates[nrow(subgraphs_coordinates) + 1, ] <- c(
                image_name,
                strsplit(image_name, "_")[[1]][1],
                clique_name,
                colMeans(attr[clique, c("x-coor", "y-coor")])
            )
        }
    }

    # Node2vec approach
    edges_cut <- binary_edges * edges_matrix
    edges_list <- which(edges_cut != 0, arr.ind = TRUE)
    edges_list <- cbind(edges_list, edges_cut[edges_list])

    node_2_vec <- node2vecR(
        edges_list,
        p = p,
        q = q,
        num_walks = num_walks,
        walk_length = walk_length,
        dim = num_embeddings
    )

    # Node2Vec Embeddings
    embeddings <- setNames(
        data.frame(matrix(0, ncol = length(col_names_embeddings) + 2, nrow = dim(node_2_vec)[1])),
        c("image_name", "image_type", col_names_embeddings)
    )
    embeddings["image_name"] <- rep(image_name, nrow(embeddings))
    embeddings["image_type"] <- rep(image_type, nrow(embeddings))
    embeddings[, col_names_embeddings] <- node_2_vec
    # embeddings[i, col_names_embeddings] <- apply(node_2_vec, 2, function(x) list(x))

    result_path = paste("../results/datasets/", image_size, "_p", p, "_q", q, "_nw", num_walks, "_wl", walk_length, "/", image_name, sep = "")
    dir.create(result_path, recursive = TRUE, showWarnings = FALSE)
    write.table(graphlet_counts, paste(result_path, "graphlet_counts.csv", sep = "/"), sep = ",")
    write.csv(subgraphs_coordinates, paste(result_path, "subgraphs_coordinates.csv", sep = "/"))
    write.csv(embeddings, paste(result_path, "embeddings.csv", sep = "/"))
}

end_time <- Sys.time()
time_taken <- end_time - start_time
time_taken

# write.csv(rect_data, "rect_data.csv")
