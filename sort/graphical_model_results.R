library(ggplot2)
library(factoextra)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

file_names <- c(
    # "../results/datasets/30_p2_q1_nw10_wl6/fibrous_0123/embeddings.csv"
    "../results/datasets/30_p2_q1_nw10_wl6/dotted_0001/embeddings.csv"

)
embeddings <- do.call(rbind, lapply(file_names, read.csv))
embeddings_only <- embeddings[, paste("embedding", 1:10, sep = "_")]
# pca <- princomp(embeddings_only)

# fviz_eig(pca)

# fviz_pca_ind(pca,
#             col.ind = "cos2", # Color by the quality of representation
#             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
#             repel = TRUE, # Avoid text overlapping
#             geom = "point"
#             )

# fviz_pca_var(pca,
#             col.var = "contrib", # Color by contributions to the PC
#             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
#             repel = TRUE     # Avoid text overlapping
#             )

# groups <- as.factor(embeddings$image_type)

# fviz_pca_ind(pca,
#             col.ind = groups, # color by groups
#             palette = c("#00AFBB",  "#FC4E07"),
#             addEllipses = TRUE, # Concentration ellipses
#             ellipse.type = "confidence",
#             legend.title = "Groups",
#             repel = TRUE,
#             geom = "point"
#             )

embeddings_only_t <- t(embeddings_only)
transposed_pca <- prcomp(embeddings_only_t)
fviz_eig(transposed_pca)
fviz_pca_ind(transposed_pca,
            col.ind = "cos2", # Color by the quality of representation
            gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
            repel = TRUE # Avoid text overlapping
            # geom = "point"
            )

file_names <- c(
    "../results/datasets/30_p2_q1_nw10_wl6/fibrous_0123/embeddings.csv"
    # "../results/datasets/30_p2_q1_nw10_wl6/dotted_0001/embeddings.csv"

)
embeddings2 <- do.call(rbind, lapply(file_names, read.csv))
embeddings_only2 <- embeddings2[, paste("embedding", 1:10, sep = "_")]

embeddings_only_t2 <- t(embeddings_only2)
transposed_pca2 <- prcomp(embeddings_only_t2)
fviz_eig(transposed_pca2)
fviz_pca_ind(transposed_pca2,
            col.ind = "cos2", # Color by the quality of representation
            gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
            repel = TRUE # Avoid text overlapping
            # geom = "point"
            )