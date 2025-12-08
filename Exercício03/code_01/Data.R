# RN.R
source("Config.R")
library(torch)

set.seed(config$seed)

n <- config$n
x <- sort(runif(n, config$x_min, config$x_max))

# função determinística (sem ruído)
f_x <- 3/(3 + 2*abs(x)^3) +
  exp(-x^2) +
  cos(x)*sin(x)

# adiciona ruído Normal
y <- f_x + rnorm(n) * config$noise_sd

# tensores completos (grid todo, para plot)
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