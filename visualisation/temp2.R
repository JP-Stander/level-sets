library(ggplot2)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

gc <- paste0(
    "../results/datasets/30_p2_q1_nw10_wl6/", 
    list.files("../results/datasets/30_p2_q1_nw10_wl6/", pattern = "embeddings.csv", recursive = TRUE)
    )

embeddings <- do.call(rbind, lapply(gc, read.csv))

melted_embedding <- melt(
    embeddings[,2:ncol(embeddings)], 
    na.rm=FALSE, 
    value.name="embedding",
    id = c("image_name", "image_type")
)

melted_embedding_ss = melted_embedding[melted_embedding$variable!="embedding_6", c("image_name", "image_type", "embedding")]
ggplot(melted_embedding_ss) +
geom_density(aes(x = embedding, group=image_name, col=image_type))

