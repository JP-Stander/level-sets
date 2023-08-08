library(ggplot2)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

subgraphs_coordinates <- read.csv("../results/datasets/30_p2_q1_nw10_wl6/dotted_0001/subgraphs_coordinates.csv")
embeddings <- read.csv("../results/datasets/30_p2_q1_nw10_wl6/dotted_0001/embeddings.csv")

subgraphs_coordinates$x <-  as.numeric(subgraphs_coordinates$x)
subgraphs_coordinates$y <-  as.numeric(subgraphs_coordinates$y)

ggplot(
    subgraphs_coordinates,
    aes(x = x, y = y, col = image_type)
) +
geom_point() +
facet_grid(graphlet_name ~ image_name)

embedding_metled <- melt(embeddings, id = c("image_name", "image_type"))

color_list <- c("dotted" = "blue", "fibrous" = "#19a77c")

for (emb_num in 2:11){
    plot <- ggplot() +
    ggtitle(paste("Embedding ", emb_num-1))
    for (img_idx in seq_len(dim(embeddings)[1])){
        print(img_idx)
        plot <- plot +
        geom_density(aes(x = unlist(embeddings[img_idx, emb_num]), col = embeddings[img_idx, "image_type"]))
    }
    plot <- plot + scale_color_manual(values = color_list)
    #0000ff#0077ff#1d64b6#1560b6
    # geom_density(aes(x = unlist(embeddings[1, emb_num])), col = "blue") +
    # geom_density(aes(x = unlist(embeddings[2, emb_num])), col = "#19a77c") +
    # geom_density(aes(x = unlist(embeddings[3, emb_num])), col = "blue") +
    # geom_density(aes(x = unlist(embeddings[4, emb_num])), col = "#19a77c") +

    folder <- paste("../results/plots/is", image_size, "_p", p, "_q", q, "_nw", num_walks, "_wl", walk_length, sep = "")
    dir.create(folder, showWarnings = FALSE)
    png(paste(folder, "/Embedding ", emb_num-1, ".png", sep = ""))
    print(plot)
    dev.off()
}
#2,3,6
# emb_num <- 7
# #d vs f
# ks.test(unlist(embeddings[1, emb_num]), unlist(embeddings[2, emb_num]))
# #d vs f
# ks.test(unlist(embeddings[3, emb_num]), unlist(embeddings[4, emb_num]))
# #d vs d
# ks.test(unlist(embeddings[1, emb_num]), unlist(embeddings[3, emb_num]))
# #f vs f
# ks.test(unlist(embeddings[2, emb_num]), unlist(embeddings[4, emb_num]))

# plot_list <- list()

# for (type_val in unique(embeddings["image_type"])[[1]]) {
#     subset_data <- subset(embeddings, image_type == type_val)
#     for (emb_val in 1:5) {
#     # Subset the data for the current 'type' and 'emb' combination
#     # subset_data <- subset_data[, c(1, emb_val+1, 12)]
#     # Create the plot for the current combination
#     assign(paste(type_val, emb_val, sep = "_"), ggplot() +
#         geom_density(aes(x = unlist(subset_data[1, emb_val + 1])), col = "blue") +
#         geom_density(aes(x = unlist(subset_data[2, emb_val + 1])), col = "blue")
#     )
#     print(paste(type_val, emb_val, sep = "_"))
#     # dev.off()
#     # Store the plot in the list
#     # plot_list[[paste(type_val, emb_val, sep = "_")]] <- plot
#   }

# }

# plot_grid(dotted_1, dotted_2, dotted_3, dotted_4, dotted_5,
#         fibrous_1, fibrous_2, fibrous_3, fibrous_4, fibrous_5,
#         ncol = 2, nrow = 5)
# sg_ss <- subset(subgraphs_coordinates, image_name == "fibrous_0216")
# ggplot(
#     sg_ss,
#     aes(x = x, y = y, col = image_name, shape = graphlet_name)
# ) +
# geom_point() +
# facet_wrap(~graphlet_name, ncol=3)

# sg_ss <- subset(subgraphs_coordinates, image_name == "dotted_0001")
# ggplot(
#     sg_ss,
#     aes(x = x, y = y, col = image_name, shape = graphlet_name)
# ) +
# geom_point() +
# facet_wrap(~graphlet_name, ncol=3)

# image(img, col=grey.colors(n=255),)# xlim=c(0,10), ylim=c(0,10))
# points(
#     subgraphs_coordinates[, c('x','y')]/10, 
#     pch=c(16:20)[as.numeric(subgraphs_coordinates$graphlet_name)]
# )