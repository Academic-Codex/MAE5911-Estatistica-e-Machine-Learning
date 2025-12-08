# Data.R  -------------------------------------------------------------

source("Config.R")
library(torch)

set.seed(config$seed)

n  <- config$n
x  <- sort(runif(n, config$x_min, config$x_max))

# ---- média verdadeira f(x) (como antes) ----
f_x <- 3/(3 + 2*abs(x)^3) +
  exp(-x^2) +
  cos(x)*sin(x)

# ---- variância verdadeira σ^2(x) (exemplo heteroscedástico) ----
# algo ondulado, parecido com o que o prof faz
f_sigma2 <- function(x) {
  0.3 + 0.8 * (1 + sin(1 + 2*x) + 0.2*cos(1 + 2*x))^2
}
sigma2_true <- f_sigma2(x)
sigma_true  <- sqrt(sigma2_true)

# ---- gera dados: Y = f(x) + ε, ε ~ N(0, σ^2(x)) ----
y <- f_x + rnorm(n) * sigma_true

# tensors “full” para fazer previsões/plots em toda a grade
X_full <- torch_tensor(matrix(x, ncol = 1), dtype = torch_float())
Y_full <- torch_tensor(matrix(y, ncol = 1), dtype = torch_float())

# ---- split treino / teste ----
n_train <- round(config$p_train * n)
idx <- sample(1:n)

idx_train <- idx[1:n_train]
idx_test  <- idx[(n_train + 1):n]

x_train <- x[idx_train]
y_train <- y[idx_train]

x_test <- x[idx_test]
y_test <- y[idx_test]

X_train <- torch_tensor(matrix(x_train, ncol = 1), dtype = torch_float())
Y_train <- torch_tensor(matrix(y_train, ncol = 1), dtype = torch_float())

X_test <- torch_tensor(matrix(x_test, ncol = 1), dtype = torch_float())
Y_test <- torch_tensor(matrix(y_test, ncol = 1), dtype = torch_float())