# Model.R
# (pressupõe que library(torch) já foi chamado em Train.R)

# -------------------------------
# Rede neural para o quantil Q_q(x)
# -------------------------------
QuantileNet <- nn_module(
  initialize = function(n_hidden = 64, n_layers = 2) {
    # n_layers = número de camadas ocultas "linear+gelu"
    layers <- list()
    
    # 1ª camada: da entrada (1D) para o 1º hidden
    layers <- append(layers, list(
      nn_linear(1, n_hidden),
      nn_gelu()
    ))
    
    # camadas ocultas intermediárias (hidden -> hidden)
    if (n_layers > 1) {
      for (i in 2:n_layers) {
        layers <- append(layers, list(
          nn_linear(n_hidden, n_hidden),
          nn_gelu()
        ))
      }
    }
    
    # camada de saída: hidden -> 1 (quantil)
    layers <- append(layers, list(
      nn_linear(n_hidden, 1)
    ))
    
    # guarda tudo em um nn_sequential, como o prof faz
    self$net <- do.call(nn_sequential, layers)
  },
  
  forward = function(x) {
    self$net(x)
  }
)

quantile_loss <- function(y_pred, y_true, q) {
  u <- y_true - y_pred
  q_tensor <- torch_tensor(q, dtype = torch_float(), device = y_pred$device)
  loss_elem <- torch_maximum(q_tensor * u, (q_tensor - 1) * u)
  torch_mean(loss_elem)
}

normal_quantile_loss <- function(y_pred_q, y_true, q, sigma) {
  # quantil da Normal padrão
  z_q <- qnorm(q)
  
  # transformar q e sigma em tensores no mesmo device que y_pred_q
  z_tensor     <- torch_tensor(z_q,    dtype = torch_float(), device = y_pred_q$device)
  sigma_tensor <- torch_tensor(sigma,  dtype = torch_float(), device = y_pred_q$device)
  
  # variável transformada Z = Y + sigma * z_q
  Z <- y_true + sigma_tensor * z_tensor
  
  # MSE entre Z e Q_q(x)
  torch_mean((Z - y_pred_q)$pow(2))
}

# QuantileNet <- nn_module(
#   initialize = function(n_hidden = 32) {
#     self$l1 <- nn_linear(1, n_hidden)
#     self$l2 <- nn_linear(n_hidden, n_hidden)
#     self$l3 <- nn_linear(n_hidden, 1)
#   },
#   forward = function(x) {
#     # x: tensor (n, 1)
#     x <- self$l1(x)
#     x <- nnf_relu(x)
#     x <- self$l2(x)
#     x <- nnf_relu(x)
#     x <- self$l3(x)
#     x
#   }
# )

# -------------------------------
# Pinball loss (quantile loss)
# -------------------------------
# quantile_loss <- function(y_pred, y_true, q) {
#   # y_pred, y_true: tensores de mesmo shape (n, 1)
#   # q: escalar numérico em (0, 1), ex: 0.75
# 
#   u <- y_true - y_pred
# 
#   # transforma q em tensor no mesmo device do modelo
#   q_tensor <- torch_tensor(q, dtype = torch_float(), device = y_pred$device)
# 
#   # máximo elemento a elemento: torch_maximum (evita conflito com arg dim)
#   loss_elem <- torch_maximum(q_tensor * u, (q_tensor - 1) * u)
# 
#   torch_mean(loss_elem)
# }