# Data.R
source("Config.R")
library(torch)

set.seed(config$seed)

n <- config$n
x <- sort(runif(n, config$x_min, config$x_max))

# --------------------------------------------------
# função determinística f(x) (sem ruído)
# --------------------------------------------------
f_x <- 3/(3 + 2*abs(x)^3) +
  exp(-x^2) +
  cos(x) * sin(x)

# --------------------------------------------------
# função da variância verdadeira σ(x)
#  - se config$hetero = FALSE -> σ(x) = constante
#  - se config$hetero = TRUE  -> σ(x) varia com x
# --------------------------------------------------
sigma_fun <- function(x) {
  if (isTRUE(config$hetero)) {
    base  <- config$noise_sd
    alpha <- config$hetero_alpha  # quão forte varia com |x|
    base * (1 + alpha * abs(x))   # ex: cresce com |x|
  } else {
    rep(config$noise_sd, length(x))
  }
}

sigma_true <- sigma_fun(x)   # vetor σ(x) no grid completo

# --------------------------------------------------
# gera y com ruído Normal(0, σ(x)^2)
# rnorm em R aceita vetor de desvios-padrão
# --------------------------------------------------
y <- f_x + rnorm(n, mean = 0, sd = sigma_true)

# --------------------------------------------------
# tensores completos (grid todo, para plot)
# --------------------------------------------------
X_full    <- torch_tensor(matrix(x,          ncol = 1), dtype = torch_float())
Y_full    <- torch_tensor(matrix(y,          ncol = 1), dtype = torch_float())
Sigma_full<- torch_tensor(matrix(sigma_true, ncol = 1), dtype = torch_float())

# --------------------------------------------------
# split treino / teste
# --------------------------------------------------
n_train <- round(config$p_train * n)
idx     <- sample(1:n)

idx_train <- idx[1:n_train]
idx_test  <- idx[(n_train + 1):n]

x_train <- x[idx_train]
y_train <- y[idx_train]

x_test  <- x[idx_test]
y_test  <- y[idx_test]

# (opcional) também guardar σ(x) no treino/teste, só para diagnóstico
sigma_train <- sigma_true[idx_train]
sigma_test  <- sigma_true[idx_test]

# tensores de treino/teste
X_train <- torch_tensor(matrix(x_train, ncol = 1), dtype = torch_float())
Y_train <- torch_tensor(matrix(y_train, ncol = 1), dtype = torch_float())

X_test  <- torch_tensor(matrix(x_test,  ncol = 1), dtype = torch_float())
Y_test  <- torch_tensor(matrix(y_test,  ncol = 1), dtype = torch_float())