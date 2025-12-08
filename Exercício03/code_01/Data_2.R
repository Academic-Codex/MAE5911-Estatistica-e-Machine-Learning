set.seed(32)

n <- 1000
x <- sort(runif(n, -4, 4))
y <- 3/(3 + 2*abs(x)^3) + exp(-x^2) + cos(x)*sin(x) + rnorm(n)*0.3

# Plot da nuvem de pontos
plot(x, y, pch=16, col=rgb(0,0,0,0.3), main="Quantis condicionais do ruído")

# Definir número de janelas ao longo de x
num_bins <- 80

# Quebrar x em janelas
breaks <- seq(min(x), max(x), length.out = num_bins)

# Vetores onde vamos guardar os quantis
qx <- c()
qy75 <- c()
qy25 <- c()

# Calcular quantis dentro de cada janela
for (i in 1:(length(breaks)-1)) {
  idx <- which(x >= breaks[i] & x < breaks[i+1])
  if (length(idx) > 0) {
    qx <- c(qx, mean(x[idx]))
    qy75 <- c(qy75, quantile(y[idx], 0.75))
    qy25 <- c(qy25, quantile(y[idx], 0.25))
  }
}

# Destacar o quantil 75% como pontos azuis
points(qx, qy75, col="blue", pch=19, cex=1.2)

# Destacar o quantil 25% como pontos vermelhos
points(qx, qy25, col="red", pch=19, cex=1.2)

legend("topright",
       legend=c("Quantil 75%","Quantil 25%"),
       col=c("blue","red"),
       pch=19)