# Import needed R libraries
library(reticulate)
library(jsonlite)
library(Matrix)

py_config()
setwd("/home/qxz1djt/projects/phd/level-sets")

# Import needed python functions
source_python("images/utils.py")
source_python("graphical_model/utils.py")

# Import needed R functions
source("graphical_model/utils.R")
source("graphical_model/reference_graphs.R")

# List images in database
image_types <- c("dotted", "fibrous")
images <- c()
for (image_type in image_types){
    images <- c(images,
        paste(
            image_type,
            sample(list.files(paste0("../dtd/images/", image_type), pattern = "*.jpg", recursive = TRUE), 2),
            # list.files(paste0("../dtd/images/", image_type), pattern = "*.jpg", recursive = TRUE),
            sep = "/"
        )
    )
}

image_size <- 50
ls_spatial_dist <- "euclidean"
ls_attr_dist <- "cityblock"
centroid_method <- "mean"
custom_folder <- ""

result_path <- paste(
    "../my_graphical_dataset/img_size_",
    image_size,
    "/spatial_", ls_spatial_dist,
    "_attr_", ls_attr_dist,
    "/centroid_method_", centroid_method,
    sep = ""
    )
if (custom_folder != "") {
    result_path <- paste(result_path, custom_folder, sep = "/")
}
dir.create(result_path, recursive = TRUE, showWarnings = FALSE)
completed_images <- c()
for (file in list.files(result_path, pattern = "*.json")){
    json_data <- jsonlite::fromJSON(paste0(result_path, "/", file))
    completed_images <- c(completed_images, json_data$metadata$`file name`)
}
completed_images <- unique(completed_images)
all_images_sorted <- sort(images)
images <- images[!(images %in% completed_images)]

img_indexes <- c(
    list(list(0, 50, 0, 50)),
    list(list(0, 50, 150, 200)),
    list(list(150, 200, 0, 50)),
    list(list(150, 200, 150, 200)),
    list(list(75, 125, 75, 125))
)

for (i in seq_along(images)) {
    cat("Processing image", i, "of", length(images), "num: ", which(all_images_sorted == images[i]), "\n")
    img_og <- load_image(
        paste0("../dtd/images/", images[i], sep = ""),
        list(image_size, image_size)
    )
    img_seg <- 1
    # for (idx_set in img_indexes){
        img <- img_og#[idx_set[[1]]:idx_set[[2]], idx_set[[3]]:idx_set[[4]]]

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
        binary_edges[lower.tri(binary_edges)] <- FALSE
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
        write_json(data_to_save, paste0(result_path, "/", which(all_images_sorted == images[i]), "_", img_seg, ".json"))
        break
        # img_seg <- img_seg + 1
  #  }
}

# no_cores <- min(9, detectCores() - 2)
# group_indices <- rep(1:no_cores, each = ceiling(length(images) / no_cores), length.out = length(images))
# image_sets <- split(images, group_indices)
# registerDoParallel(no_cores)
# foreach(k = 1:no_cores, .export = c("load_image", "graphical_model"), .packages = c("jsonlite", "Matrix")) %dopar% {
#     image_set <- image_sets[[k]]
#     for (i in seq_along(image_set)){
#         cat("Processing image", i, "of", length(image_set), "in image set ", k, "\n")
#         img <- load_image(
#             paste0("../dtd/images/", image_set[i], sep = "")
#             # list(image_size, image_size)
#         )
#         img <- img[1:image_size, 1:image_size]
#         gm <- graphical_model(
#             img,
#             TRUE,
#             0.5,
#             normalise_gray = TRUE,
#             size_proportion = FALSE,
#             ls_spatial_dist = ls_spatial_dist,
#             ls_attr_dist = ls_attr_dist,
#             centroid_method = centroid_method
#         )

#         nodes <- gm[1]
#         edges <- gm[2]
#         attr <- gm[3][[1]]

#         edges_matrix <- matrix(unlist(edges), ncol = dim(nodes[[1]])[1], byrow = TRUE)
#         binary_edges <- edges_matrix > 0.3
#         non_binary_edges <- binary_edges * edges_matrix

#         indices <- which(binary_edges, arr.ind = TRUE)

#         edges <- unname(lapply(split(indices, seq(nrow(indices))), function(index) {
#             c("edge" = list(index), "weight" = edges_matrix[index[1], index[2]])
#         }))

#         features <- setNames(lapply(seq_len(nrow(attr)), function(j) {
#             unlist(attr[j, c("intensity", "size", "compactness", "elongation")])
#         }), attr$`level-set`)
#         metadata <- setNames(lapply(seq_len(nrow(attr)), function(j) {
#             unlist(attr[j, c("pixel_indices")])
#         }), attr$`level-set`)
#         metadata$`file name` <- image_set[i]
#         edges_matrix <- non_binary_edges

#         # Combine edges and features into a list
#         data_to_save <- list(
#             "edges" = edges,
#             "features" = features,
#             "metadata" = metadata
#         )
#         write_json(data_to_save, paste0(result_path, "/", which(all_images_sorted == image_set[i]), ".json"))
#     }
# }