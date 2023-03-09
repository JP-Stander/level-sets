
rotate <- function(df, degree) {
  dfr <- df
  degree <- pi * degree / 180
  l <- sqrt(df[,1]^2 + df[,2]^2)
  teta <- atan(df[,2] / df[,1])
  dfr[,1] <- round(l * cos(teta - degree))
  dfr[,2] <- round(l * sin(teta - degree))
  return(dfr)
}
