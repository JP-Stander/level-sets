library(ggplot2)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

gc <- paste0(
    "../results/datasets/30_p2_q1_nw10_wl6/", 
    list.files("../results/datasets/30_p2_q1_nw10_wl6/", pattern = "subgraphs_coordinates.csv", recursive = TRUE)
    )

graphlet_coors <- do.call(rbind, lapply(gc, read.csv))

library(dplyr)

graphlet_counts <- graphlet_coors %>%
    mutate(count = 1) %>%
    group_by(image_name, image_type, graphlet_name) %>%
    summarise(count = sum(count)) %>%
    mutate(total = sum(count)) %>%
    mutate(proportion = count / total)

ggplot(graphlet_counts) +
geom_point(
    aes(
        x = graphlet_name,
        y = proportion,
        col = image_type
        ),
    position = "jitter"
    ) +
theme_classic() +
theme(
    axis.text = element_text(size = 15),
    axis.title = element_text(size = 15),
    legend.text = element_text(size = 15),
    legend.title = element_blank()
) +
xlab("Graphlet name")

ggsave("../results/plots/ref_graph/graphlet_prop_scatter.png")
