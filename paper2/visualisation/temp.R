library(igraph)
library(ggraph)
library(tidygraph)

source("graphical_model/reference_graphs.R")

g <- graph_from_adjacency_matrix(reference.cliques$g2$g2_1, mode = "undirected")

# Plot the graph
g <- as_tbl_graph(g)

# Plot using ggraph (ggplot2 style)
ggraph(g, layout = "nicely") +
    geom_edge_link() +
    geom_node_point(cex = 10) +
    theme_graph()
ggsave("../results/plots/ref_graph/g2_g2_1.png")
ref_list <- list(
    g2 = c("g2_1"),
    g3 = c("g3_1", "g3_2", "g3_3"),
    g4 = c("g4_1", "g4_2", "g4_3", "g4_4", "g4_5", "g4_6")
)
for (gs in c("g2", "g3", "g4")){
    for (sgs in ref_list[[gs]]){
        g <- graph_from_adjacency_matrix(reference.cliques[[gs]][[sgs[[1]]]], mode = "undirected")
        # Plot the graph
        # g <- as_tbl_graph(g)
        # Plot using ggraph (ggplot2 style)
        ggraph(g, layout = "nicely") +
            geom_edge_link() +
            geom_node_point(cex = 10) +
            theme_graph() +
            theme(
                # panel.background = element_rect(fill = 'transparent') #transparent panel bg
                plot.background = element_rect(fill = 'transparent', color = NA)
            )
        ggsave(paste0("../results/plots/ref_graph/", gs, sgs, ".png", sep=""))
    }
}

toy_example <- t(matrix(c(
    5, 5, 5, 6, 7,
    4, 4, 5, 6, 6,
    4, 3, 5, 5, 6,
    4, 3, 3, 2, 2,
    1, 1, 1, 2, 2),
    ncol = 5,
    byrow = TRUE
))

gm <- graphical_model(
        toy_example,
        TRUE,
        0.5,
        normalize_gray = TRUE
    )
nodes <- gm[1]
edges <- gm[2]
attr <- gm[3][[1]]

to_plot <- cbind(nodes[[1]], attr["intensity"])
colnames(to_plot) <- c("x", "y", "col")

library(viridis)
ggplot() +
geom_hline(yintercept = c(-0.5, 0.5, 1.5, 2.5, 3.5, 4.5)) +
geom_vline(xintercept = c(-0.5, 0.5, 1.5, 2.5, 3.5, 4.5)) +
geom_point(data = to_plot, aes(x = x, y = y, col = col), cex = 10) +
scale_color_viridis() +
theme_void() +
theme(legend.position = "none",
plot.background = element_rect(fill = 'white', color = NA)
) +
coord_fixed()
ggsave(paste0("../results/plots/ref_graph/marked_point_pattern.png", sep = ""))

edges_matrix <- matrix(unlist(edges), ncol = dim(nodes[[1]])[1], byrow = TRUE)
binary_edges <- edges_matrix > 0.1

g <- graph_from_adjacency_matrix(binary_edges, mode = "undirected")
V(g)$name <- 1:vcount(g)
# Plot the graph
g <- as_tbl_graph(g)
node_data <- data.frame(name = V(g)$name,
                        intensity = to_plot["col"])

g <- g %N>%
    left_join(node_data, by = "name")
# Plot using ggraph (ggplot2 style)
ggraph(g, layout = "nicely") +
    geom_edge_link() +
    geom_node_point(aes(col = col), cex = 10) +
    scale_color_viridis() +
    theme_graph() +
    theme(legend.position = "none")
ggsave(paste0("../results/plots/ref_graph/graphical_model.png", sep = ""))

ggplot() +
geom_point(data = to_plot, aes(x = x, y = y, col = col), cex = 10) +
scale_color_viridis() +
theme_void() +
theme(legend.position = "none") +
geom_hline(yintercept = c(-0.5, 0.5, 1.5, 2.5, 3.5, 4.5)) +
geom_vline(xintercept = c(-0.5, 0.5, 1.5, 2.5, 3.5, 4.5)) +
geom_segment(aes(x = , y= , xend= , yend=))
coord_fixed()
ggsave(paste0("../results/plots/ref_graph/marked_point_pattern.png", sep = ""))


ggraph(g, layout = "nicely") +
    geom_edge_link() +
    geom_node_point(cex = 3) +
    scale_color_viridis() +
    theme_graph() +
    theme(legend.position = "none")
ggsave(paste0("../results/plots/ref_graph/gm.png", sep = ""))