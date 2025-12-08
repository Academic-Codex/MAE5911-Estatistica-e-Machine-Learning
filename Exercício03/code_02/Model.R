# Model.R
library(torch)

library(torch)
QuantileNormalNet <- nn_module(
  "QuantileNormalNet",
  
  initialize = function(input_dim,
                        hidden_q     = 64,
                        hidden_sigma = 64) {
    
    # rede do quantil Q_q(x)
    self$q_net <- nn_sequential(
      nn_linear(input_dim, hidden_q),
      nn_gelu(),
      nn_linear(hidden_q, 1)
    )
    
    # rede da variância (igual à anterior)
    self$sigma_net <- nn_sequential(
      nn_linear(input_dim, hidden_sigma),
      nn_gelu(),
      nn_linear(hidden_sigma, 1)
    )
  },
  
  forward = function(x) {
    # x: [batch, 1]
    
    q_pred <- self$q_net(x)              # Q_q(x) previsto
    
    sigma_raw <- self$sigma_net(x)
    sigma     <- nnf_softplus(sigma_raw) + 1e-4
    sigma2    <- sigma$pow(2)
    
    list(
      q      = q_pred,   # quantil
      sigma2 = sigma2    # variância
    )
  }
)

# helper: dado um modelo e um nível q, calcula o quantil Q_q(x)
predict_quantile <- function(model, x, q) {
  z_q <- qnorm(q)
  out <- model(x)
  out$mu + z_q * out$sigma
}

# NLL da Normal escrita em função do quantil Q_q(x) e de σ^2(x)
# q_level é o quantil alvo (por ex., 0.75)
nll_normal_from_quantile <- function(Qq, sigma2, y, q_level, eps = 1e-6) {
  
  # Garantir tensores
  if (!inherits(Qq, "torch_tensor"))
    Qq <- torch_tensor(Qq, dtype = torch_float())
  if (!inherits(sigma2, "torch_tensor"))
    sigma2 <- torch_tensor(sigma2, dtype = torch_float())
  if (!inherits(y, "torch_tensor"))
    y <- torch_tensor(y, dtype = torch_float())
  
  # Segurança numérica
  sigma2_safe <- sigma2$clamp_min(eps)
  sigma <- sigma2_safe$sqrt()
  
  # z_q da Normal padrão
  z_q <- qnorm(q_level)
  z_q_t <- torch_tensor(z_q, dtype = torch_float(), device = Qq$device)
  
  # μ(x) = Q_q(x) - σ(x) z_q
  mu <- Qq - sigma * z_q_t
  
  # Mesma NLL de antes, agora em termos de μ(x) reconstruído
  term1 <- torch_log(sigma)                     # log σ(x)
  term2 <- (y - mu)$pow(2) / (2 * sigma$pow(2)) # (y - μ)^2 / 2σ^2
  
  (term1 + term2)$mean()
}