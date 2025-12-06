# Model.R
# (pressupõe que library(torch) já foi chamado em Train.R)

# -------------------------------
# Rede neural para o quantil Q_q(x)
# -------------------------------
QuantileNet <- nn_module(
  initialize = function(n_hidden = 64) {
    self$l1 <- nn_linear(1, n_hidden)
    self$l2 <- nn_linear(n_hidden, n_hidden)
    self$l3 <- nn_linear(n_hidden, 1)
  },
  forward = function(x) {
    x <- self$l1(x); x <- nnf_tanh(x)
    x <- self$l2(x); x <- nnf_tanh(x)
    x <- self$l3(x)
    x
  }
)
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
quantile_loss <- function(y_pred, y_true, q) {
  # y_pred, y_true: tensores de mesmo shape (n, 1)
  # q: escalar numérico em (0, 1), ex: 0.75
  
  u <- y_true - y_pred
  
  # transforma q em tensor no mesmo device do modelo
  q_tensor <- torch_tensor(q, dtype = torch_float(), device = y_pred$device)
  
  # máximo elemento a elemento: torch_maximum (evita conflito com arg dim)
  loss_elem <- torch_maximum(q_tensor * u, (q_tensor - 1) * u)
  
  torch_mean(loss_elem)
}