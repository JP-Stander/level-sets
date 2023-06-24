Sys.setenv(
    RETICULATE_PYTHON = "/home/qxz1djt/.local/share/virtualenvs/aip.mlops.terraform.modules-iNbkyG8C/bin/python"
)

library(reticulate)
# library(igraph)
# library(stringr)
# library(stellaR)
# library(rlist)
library(Matrix)
library(ScreenClean)
library(node2vec)
py_config()
setwd(paste0(getwd(), "/level-sets"))

# source_python("level_sets/utils.py")
source_python("images/utils.py")
source_python("graphical_model/utils.py")
source("graphical_model/utils.R")
source("graphical_model/reference_graphs.R")

images <- paste0("dotted/", list.files("../dtd/images/dotted/", pattern = "*.jpg", recursive = TRUE))
images <- c(images, paste0("fibrous/", list.files("../dtd/images/fibrous", pattern = "*.jpg", recursive = TRUE)))
# folders <- dirname(images)
num_images <- 4#length(images)
col_names <- unlist(lapply(names(reference.cliques), function(x) {
    names(reference.cliques[[x]])
}))
col_names <- c("graph_name", col_names[!startsWith(col_names, "g5")])
rect_data <- setNames(data.frame(matrix(0, ncol = length(col_names) + 10, nrow = num_images)), col_names)
subgraphs_coordinates <- data.frame(
    graph_name = character(0),
    graphlet_name = character(0),
    x = numeric(0),
    y = numeric(0)
)
# rotate <- function(x) t(apply(x, 2, rev))
start_time <- Sys.time()
for (i in c(1, 239)){ #seq_along(images[1:3])
    img <- load_image(
        paste0("../dtd/images/", images[i], sep = ""),
        list(10, 10)
    )
    image_name <- tools::file_path_sans_ext(basename(images[i]))
    rect_data[i, "graph_name"] <- image_name
    # image(img, col=grey.colors(n=255))
    # Get the image dimensions
    # height <- dim(img)[1]
    # width <- dim(img)[2]

    # # Calculate the center of the image
    # center_x <- width / 2
    # center_y <- height / 2

    # # Define the size of the block
    # block_size <- 20

    # # Calculate the corners of the block
    # start_x <- center_x - block_size / 2 + 10
    # end_x <- center_x + block_size / 2 + 10
    # start_y <- center_y - block_size / 2 + 10
    # end_y <- center_y + block_size / 2 + 10

    # Extract the block
    # img <- img[start_y:end_y, start_x:end_x]
    # image(img)
    gm <- graphical_model(
        img,
        TRUE,
        0.5,
        normalize_gray = TRUE
    )
    nodes <- gm[1]
    edges <- gm[2]
    attr <- gm[3][[1]]

    edges_matrix <- matrix(unlist(edges), ncol = dim(nodes[[1]])[1], byrow = TRUE)
    binary_edges <- edges_matrix > 0.5

    # Subgraph approach
    subgraphs <- FindAllCG(Matrix(binary_edges, sparse = TRUE), 4)
    for (k in 2:length(subgraphs)){
        subgraph <- subgraphs[[k]]
        for (j in seq_len(dim(subgraph)[1])){
            clique <- subgraph[j, ]
            clique_name <- get_graphlet_num(binary_edges[clique, clique])
            rect_data[i, clique_name] <- rect_data[i, clique_name] + 1

            subgraphs_coordinates[nrow(subgraphs_coordinates) + 1, ] <- c(
                image_name,
                clique_name, 
                colMeans(attr[clique, c("x-coor", "y-coor")])
            )
        }
    }

    # Node2vec approach
    edges_cut <- binary_edges * edges_matrix
    edges_list <- which(edges_cut != 0, arr.ind = TRUE)
    edges_list <- cbind(edges_list, edges_cut[edges_list])

    embeddings <- node2vecR(
        edges_list,
        p = 2,
        q = 1,
        num_walks = 10,
        walk_length = 6,
        dim = 10
    )
    # i_1 <- length(col_names) + 1
    # i_2 <- length(col_names) + 10
    rect_data[i, (length(col_names) + 1):(length(col_names) + 10)] <- colSums(embeddings)
    # GraphSAGE approach
    # graph <- StellarGraph$new(binary_edges)

    # sage <- GraphSAGENodeGenerator(graph, batch_size = 2) %>%
    #     set_base_model("graphsage-mean") %>%
    #     set_target_size(10) %>%
    #     set_num_samples(c(5, 2))

    # num_epochs <- 10
    # history <- sage$fit_generator(num_epochs = num_epochs)
    # node_embeddings <- sage$transform(graph$nodes())
}
end_time <- Sys.time()
time_taken <- end_time - start_time
time_taken

write.csv(rect_data, "rect_data.csv")