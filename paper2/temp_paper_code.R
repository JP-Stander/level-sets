# Import needed R libraries
library(reticulate)
library(jsonlite)
library(foreach)
library(Matrix)
library(parallel)

py_config()
setwd("/home/qxz1djt/projects/phd/level-sets")

# Import needed python functions
source_python("images/utils.py")
source_python("graphical_model/utils.py")

# Import needed R functions
source("graphical_model/utils.R")
source("graphical_model/reference_graphs.R")

# List images in database
n_images <- 10
image_types <- c("dotted", "fibrous")
images <- c(
    "dotted/dotted_0161.jpg",
    "dotted/dotted_0184.jpg",
    "fibrous/fibrous_0116.jpg",
    "fibrous/fibrous_0165.jpg"
)


num_images <- length(images)
image_size <- 50
ls_spatial_dist <- "euclidean"
ls_attr_dist <- "cityblock"
centroid_method <- "mean"
custom_folder <- ""


result_path <- paste(
    "..",
    "paper_results",
    sep = "/"
)

dir.create(result_path, recursive = TRUE, showWarnings = FALSE)


foreach(i = seq_along(images)) %do% {
    #Our method
    cat(images[i])
    cat("Processing image", i, "of", length(images), "\n")
    img <- load_image(
        paste0("../dtd/images/", images[i], sep = "")
        # list(image_size, image_size)
    )
    img <- img[1:image_size, 1:image_size]
    gm <- graphical_model(
        img,
        TRUE,
        0.5,
        normalise_gray = TRUE,
        size_proportion = FALSE,
        ls_spatial_dist = ls_spatial_dist,
        ls_attr_dist = ls_attr_dist,
        centroid_method = centroid_method
    )

    nodes <- gm[1]
    edges <- gm[2]
    attr <- gm[3][[1]]

    edges_matrix <- matrix(unlist(edges), ncol = dim(nodes[[1]])[1], byrow = TRUE)
    binary_edges <- edges_matrix > 0.3
    non_binary_edges <- binary_edges * edges_matrix

    indices <- which(binary_edges, arr.ind = TRUE)

    edges <- unname(lapply(split(indices, seq(nrow(indices))), function(index) {
        c("edge" = list(index), "weight" = edges_matrix[index[1], index[2]])
    }))

    features <- setNames(lapply(seq_len(nrow(attr)), function(j) {
        unlist(attr[j, c("intensity", "size", "compactness", "elongation")])
    }), attr$`level-set`)
    metadata <- setNames(lapply(seq_len(nrow(attr)), function(j) {
        unlist(attr[j, c("pixel_indices")])
    }), attr$`level-set`)
    metadata$`file name` <- images[i]
    edges_matrix <- non_binary_edges

    # Combine edges and features into a list
    data_to_save <- list(
        "edges" = edges,
        "features" = features,
        "metadata" = metadata
    )
    write_json(data_to_save, paste0(result_path, "/", i, ".json"))
}
