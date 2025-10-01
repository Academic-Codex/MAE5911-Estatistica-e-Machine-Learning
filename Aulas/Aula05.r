# Parâmetro verdadeiro
theta <- 4   # na Poisson, média = variância = theta

# Tamanho da amostra
n <- 30

# Gerar amostra da Poisson(theta)
set.seed(123)
amostra <- rpois(n, lambda = theta)

# Estimador: média amostral
theta_hat <- mean(amostra)

# Variância amostral
var_amostral <- var(amostra)

cat("Valor verdadeiro (theta):", theta, "\n")
cat("Estimativa (theta_hat):", theta_hat, "\n")
cat("Variância amostral:", var_amostral, "\n")

# Histograma com as linhas de theta e theta_hat
hist(amostra, breaks = seq(min(amostra)-0.5, max(amostra)+0.5, 1),
     col = "skyblue", freq = FALSE,
     main = paste("Amostra de Poisson(theta =", theta, "), n =", n),
     xlab = "Valores observados")
abline(v = theta, col = "red", lwd = 2, lty = 2)       # theta verdadeiro
abline(v = theta_hat, col = "darkgreen", lwd = 2)      # média amostral
legend("topright", legend = c(paste("theta =", theta),
                              paste("theta_hat =", round(theta_hat, 2))),
       col = c("red", "darkgreen"), lty = c(2, 1), lwd = 2)