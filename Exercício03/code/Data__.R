source("Config.R")
set.seed(config$seed)

n <- config$n
x <- sort(runif(n, config$x_min, config$x_max))

y <- 3/(3 + 2*abs(x)^3) +
  exp(-x^2) +
  cos(x)*sin(x) +
  rnorm(n) * config$noise_sd

# Tensores
X <- torch_tensor(matrix(x, ncol = 1), dtype = torch_float())
Y <- torch_tensor(matrix(y, ncol = 1), dtype = torch_float())









##################################################################################
head(x, 10)
head(y, 10)

head(data.frame(x = x, y = y), 10)

noise <- rnorm(n) * 0.3
head(noise, 10)

y_deterministic <- 3/(3 + 2*abs(x)^3) + exp(-x^2) + cos(x)*sin(x)

head(y_deterministic, 10)
head(y, 10)

head(y - y_deterministic, 10)  # deve ser igual ao noise

#estatísticas
summary(x)
summary(y)
summary(noise)

plot(x, y, pch = 16, col = rgb(0,0,0,0.4), main = "Dados simulados")
lines(x, y_deterministic, col = "red", lwd = 3)

# Sem ruido

y_deterministic <- 3/(3 + 2*abs(x)^3) + exp(-x^2) + cos(x)*sin(x)
plot(x, y_deterministic, type='l', col='red', lwd=3)

