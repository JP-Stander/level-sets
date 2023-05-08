reference.cliques <- list(
  g2 = list(
    g2_1 = matrix(c(0, 1, 1, 0), ncol = 2, byrow = TRUE)
  ),
  g3 = list(
    g3_1 = matrix(c(0, 1, 1, 
                    1, 0, 1, 
                    1, 1, 0), ncol = 3, byrow = TRUE),
    g3_2 = matrix(c(0, 1, 0, 
                    1, 0, 1, 
                    0, 1, 0), ncol = 3, byrow = TRUE),
    g3_2 = matrix(c(0, 1, 1, 
                    1, 0, 0, 
                    1, 0, 0), ncol = 3, byrow = TRUE)
  ),
  g4 = list(
    g4_1 = matrix(c(0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0), ncol = 4, byrow = TRUE),
    g4_2 = matrix(c(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0), ncol = 4, byrow = TRUE),
    g4_3 = matrix(c(0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0), ncol = 4, byrow = TRUE),
    g4_4 = matrix(c(0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0), ncol = 4, byrow = TRUE),
    g4_5 = matrix(c(0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0), ncol = 4, byrow = TRUE),
    g4_6 = matrix(c(0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0), ncol = 4, byrow = TRUE)
  ),
  g5 = list(
    g5_1  = matrix(c(0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0), ncol = 5, byrow = TRUE),
    g5_2  = matrix(c(0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0), ncol = 5, byrow = TRUE),
    g5_3  = matrix(c(0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0), ncol = 5, byrow = TRUE),
    g5_4  = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0), ncol = 5, byrow = TRUE),
    g5_5  = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0), ncol = 5, byrow = TRUE),
    g5_6  = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0), ncol = 5, byrow = TRUE),
    g5_7  = matrix(c(0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0), ncol = 5, byrow = TRUE),
    g5_8  = matrix(c(0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0), ncol = 5, byrow = TRUE),
    g5_9  = matrix(c(0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0), ncol = 5, byrow = TRUE),
    g5_10 = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0), ncol = 5, byrow = TRUE),
    g5_11 = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0), ncol = 5, byrow = TRUE),
    g5_12 = matrix(c(0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0), ncol = 5, byrow = TRUE),
    g5_13 = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0), ncol = 5, byrow = TRUE),
    g5_14 = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0), ncol = 5, byrow = TRUE),
    g5_15 = matrix(c(0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0), ncol = 5, byrow = TRUE),
    g5_16 = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0), ncol = 5, byrow = TRUE),
    g5_17 = matrix(c(0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0), ncol = 5, byrow = TRUE),
    g5_18 = matrix(c(0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0), ncol = 5, byrow = TRUE),
    g5_19 = matrix(c(0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0), ncol = 5, byrow = TRUE),
    g5_20 = matrix(c(0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0), ncol = 5, byrow = TRUE),
    g5_21 = matrix(c(0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0), ncol = 5, byrow = TRUE)
  )
)

# health checks
# for (num in c("2", "3", "4", "5")) {
#   healthy <- TRUE
#   print(paste0("Health checks for Cliques of size ", num))
#   print("Check symmetry")
#   for (g in names(eval(parse(text = paste("cliques.size.", num, sep = ""))))) {
#     if (!(isSymmetric(eval(parse(text = paste("cliques.size.", num, sep = "")))[[g]]))) {
#       print(paste("Graph ", g, " is not symmetric!"))
#       healthy <- FALSE
#     }
#   }
#   print("Check all diagonals are zero")
#   for (g in names(eval(parse(text = paste("cliques.size.", num, sep = ""))))) {
#     if (!(all(diag(eval(parse(text = paste("cliques.size.", num, sep = "")))[[g]]) == 0))) {
#       print(paste("Graph ", g, " has non-zero elements on it's diagonal!"))
#       healthy <- FALSE
#     }
#   }
#   if (healthy == TRUE) {
#     print("Health checks Passed")
#   } else {
#     print("Health checks Failed")
#   }
# }
