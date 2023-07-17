Sys.setenv(
    RETICULATE_PYTHON = "/home/qxz1djt/.local/share/virtualenvs/aip.mlops.terraform.modules-iNbkyG8C/bin/python"
)
# Import needed R libraries
library(reticulate)
library(Matrix)
library(reshape2)
library(jsonlite)
library(foreach)
py_config()
setwd(paste0(getwd(), "/level-sets"))

# Import needed python functions
source_python("images/utils.py")
source_python("graphical_model/utils.py")

# Import needed R functions
source("graphical_model/utils.R")
source("graphical_model/reference_graphs.R")

# List images in database
n_images <- 100
images <- paste0("dotted/", sample(list.files("../dtd/images/dotted/", pattern = "*.jpg", recursive = TRUE), n_images))
images <- c(paste0("fibrous/", sample(list.files("../dtd/images/fibrous", pattern = "*.jpg", recursive = TRUE), n_images)), images)

num_images <- length(images)
image_size <- 10

tracking <- matrix(NA, nrow=length(images), ncol=2)
foreach (i = 1:length(images)) %do% {
    cat('Processing image', i, 'of', length(images), '\n')
    img <- load_image(
        paste0("../dtd/images/", images[i], sep = ""),
        list(image_size, image_size)
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

    indices <- which(binary_edges, arr.ind = TRUE)

    # edges <- split(indices, seq(nrow(indices)))

    features <- setNames(lapply(1:nrow(attr), function(i) {
    unlist(attr[i, c("intensity")])#unlist(attr[i, c("intensity", "size", "compactness", "elongation")])
    }), attr$`level-set`)

    # Combine edges and features into a list
    data_to_save <- list("edges" = indices, "features" = features)
    write_json(data_to_save, paste0("../my_graphical_dataset/", i, ".json"))
    tracking[i, ] <- c(i, images[i])
}
write.csv(tracking, file = paste0("../my_graphical_dataset/tracking.csv"))
