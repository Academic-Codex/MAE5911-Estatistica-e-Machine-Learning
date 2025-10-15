# Gerar e visualizar uma amostra exponencial
n <- 1000          # tamanho da amostra
theta0 <- 2        # média verdadeira (λ = 1/theta0)

x <- rexp(n, rate = 1/theta0)

# Histograma + densidade teórica
hist(x,
     prob = TRUE, col = "lightblue",
     main = "Distribuição Exponencial",
     xlab = "x", ylab = "densidade")

curve(dexp(x, rate = 1/theta0),
      add = TRUE, col = "red", lwd = 2)
rug(x)