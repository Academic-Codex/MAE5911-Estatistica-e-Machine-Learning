# Model.R
library(torch)

library(torch)

HeteroNormalNet <- nn_module(
  "HeteroNormalNet",
  
  initialize = function(input_dim,
                        hidden_mu    = 64,
                        hidden_sigma = 64) {
    
    # rede da média μ(x)
    self$mu_net <- nn_sequential(
      nn_linear(input_dim, hidden_mu),
      nn_relu(),
      nn_linear(hidden_mu, hidden_mu),
      nn_relu(),
      nn_linear(hidden_mu, 1)
    )
    
    # rede da variância (na verdade algo que vira σ(x))
    self$sigma_net <- nn_sequential(
      nn_linear(input_dim, hidden_sigma),
      nn_relu(),
      nn_linear(hidden_sigma, hidden_sigma),
      nn_relu(),
      nn_linear(hidden_sigma, 1)
    )
  },
  
  forward = function(x) {
    # x: [batch, 1]
    
    mu <- self$mu_net(x)               # tensor [batch, 1]
    
    sigma_raw <- self$sigma_net(x)     # qualquer real
    sigma     <- nnf_softplus(sigma_raw) + 1e-4  # garante σ(x) > 0
    
    sigma2 <- sigma$pow(2)             # variância
    
    # IMPORTANTE: devolver com esse nome
    list(
      mu     = mu,
      sigma2 = sigma2
    )
  }
)

# helper: dado um modelo e um nível q, calcula o quantil Q_q(x)
predict_quantile <- function(model, x, q) {
  z_q <- qnorm(q)
  out <- model(x)
  out$mu + z_q * out$sigma
}

# NLL para Normal heteroscedástico,
# recebendo mu(x) e sigma2(x) (variância) como tensores
nll_normal_hetero <- function(mu, sigma2, y) {
  
  # Garantir que são tensores (se vier numeric, convertemos)
  if (!inherits(mu, "torch_tensor")) {
    mu <- torch_tensor(mu, dtype = torch_float())
  }
  if (!inherits(sigma2, "torch_tensor")) {
    sigma2 <- torch_tensor(sigma2, dtype = torch_float())
  }
  if (!inherits(y, "torch_tensor")) {
    y <- torch_tensor(y, dtype = torch_float())
  }
  
  # Segurança numérica: variância >= eps
  eps <- 1e-6
  sigma2_safe <- sigma2$clamp_min(eps)
  
  # Desvio-padrão
  sigma <- sigma2_safe$sqrt()
  
  # Termos da NLL da Normal
  term1 <- torch_log(sigma)                     # log σ(x)
  term2 <- (y - mu)$pow(2) / (2 * sigma$pow(2)) # (y - μ)^2 / 2σ^2
  
  loss <- (term1 + term2)$mean()
  loss
}
# nll_normal_hetero <- function(mu, sigma, y_true, eps = 1e-6) {
#   # mu, sigma, y_true: tensores de mesma forma
#   sigma <- sigma + eps
#   
#   term1 <- torch_log(sigma)
#   term2 <- (y_true - mu)$pow(2) / (2 * sigma$pow(2))
#   
#   torch_mean(term1 + term2)
# }