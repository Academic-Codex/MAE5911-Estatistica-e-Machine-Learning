yq_hat <- as.numeric(model(X)$squeeze())

plot(x, y, pch = 16, col = rgb(0,0,0,0.3), main="Quantil 75% condicional")
lines(x, yq_hat, col = "blue", lwd = 3)