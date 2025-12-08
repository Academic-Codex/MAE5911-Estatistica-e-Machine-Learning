source("Config.R")
library(torch)

set.seed(config$seed)

# ------------------------------------------------------
# 1) Geração do grid x e da média verdadeira f(x)
# ------------------------------------------------------
n  <- config$n
x  <- sort(runif(n, config$x_min, config$x_max))

f_x <- 3/(3 + 2*abs(x)^3) +
  exp(-x^2) +
  cos(x)*sin(x)

# ------------------------------------------------------
# 2) Variância verdadeira constante:
#       sigma(x) = noise_sd
#       sigma^2(x) = noise_sd^2
# ------------------------------------------------------
sigma_true  <- rep(config$noise_sd,  length(x))
sigma2_true <- rep(config$noise_sd^2, length(x))

# ------------------------------------------------------
# 3) Geração dos dados
#       y = f(x) + sigma * eps
#       eps ~ N(0,1)
# ------------------------------------------------------
y <- f_x + rnorm(n, mean = 0, sd = sigma_true)

# ------------------------------------------------------
# 4) Tensors completos para plot/predição
# ------------------------------------------------------
X_full <- torch_tensor(matrix(x, ncol = 1), dtype = torch_float())
Y_full <- torch_tensor(matrix(y, ncol = 1), dtype = torch_float())

# ------------------------------------------------------
# 5) Split treino/teste
# ------------------------------------------------------
n_train <- round(config$p_train * n)
idx <- sample(1:n)

idx_train <- idx[1:n_train]
idx_test  <- idx[(n_train + 1):n]

x_train <- x[idx_train]
y_train <- y[idx_train]

x_test  <- x[idx_test]
y_test  <- y[idx_test]

X_train <- torch_tensor(matrix(x_train, ncol = 1), dtype = torch_float())
Y_train <- torch_tensor(matrix(y_train, ncol = 1), dtype = torch_float())

X_test  <- torch_tensor(matrix(x_test,  ncol = 1), dtype = torch_float())
Y_test  <- torch_tensor(matrix(y_test, dtype = torch_float()))