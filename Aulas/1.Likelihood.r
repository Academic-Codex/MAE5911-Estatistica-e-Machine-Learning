rm(list=ls()); set.seed(42)

n <- 100
theta_true <- runif(1, 0, 1)
x <- rbinom(n, size=1, prob=theta_true)
thetahat <- mean(x)

L  <- function(theta) theta^sum(x) * (1 - theta)^(n - sum(x))
LL <- function(theta) L(theta) / L(thetahat)

# plota a verossimilhança
plot(LL, xlim = c(0, 1),
     main = "Função de Verossimilhança Bernoulli",
     xlab = expression(theta),
     ylab = expression(L(theta)/L(hat(theta))),
     lwd = 2)

# adiciona linhas verticais
abline(v = theta_true, col = "red",  lty = 2, lwd = 2)   # θ verdadeiro
abline(v = thetahat,   col = "blue", lty = 1, lwd = 2)   # θ̂ (EMV)

# legenda
legend("topright",
       legend = c(expression(theta["verdadeiro"]),
                  expression(hat(theta)~"(EMV)")),
       col = c("red", "blue"),
       lwd = 2, lty = c(2,1), bty = "n")

# imprime valores numéricos
cat("θ verdadeiro  =", round(theta_true, 4), "\n")
cat("θ̂ (estimado) =", round(thetahat, 4), "\n")
cat("Diferença     =", round(thetahat - theta_true, 4), "\n")

##### PROF. #######
n = 10
# theta = runif(1, 0, 1)
theta = 0.9
x = rbinom(n, size = 1, prob = theta)

L = function(theta) theta^sum(x) * (1 - theta)^(n - sum(x))
LL = function(theta) L(theta) / L(mean(x))

plot(LL, xlim = c(0, 1))
abline(v = mean(x))
abline(v = theta, col = "red",  lty = 2, lwd = 2)   # θ verdadeiro
mean(x)