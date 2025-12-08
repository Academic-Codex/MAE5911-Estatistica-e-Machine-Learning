# Model.R
library(torch)

HeteroNormalNet <- nn_module(
  "HeteroNormalNet",
  
  initialize = function(input_dim,
                        hidden_mu   = config$hidden_mu,
                        hidden_sigma= config$hidden_sigma) {
    
    # Rede da média μ(x)
    self$mu_net <- nn_sequential(
      nn_linear(input_dim, hidden_mu),
      nn_relu(),
      nn_linear(hidden_mu, hidden_mu),
      nn_relu(),
      nn_linear(hidden_mu, 1)
    )
    
    # Rede da variância (na verdade log-σ ou algo antes do softplus)
    self$sigma_net <- nn_sequential(
      nn_linear(input_dim, hidden_sigma),
      nn_relu(),
      nn_linear(hidden_sigma, hidden_sigma),
      nn_relu(),
      nn_linear(hidden_sigma, 1)
    )
  },
  
  forward = function(x) {
    # x: tensor [batch, 1]
    
    mu_raw    <- self$mu_net(x)        # pode ser qualquer real
    sigma_raw <- self$sigma_net(x)     # pode ser qualquer real
    
    # Garantir σ(x) > 0
    sigma <- nnf_softplus(sigma_raw) + 1e-4
    
    # Se quiser, pode devolver a lista:
    # list(mu = mu_raw, sigma = sigma)
    # mas fica bem confortável guardar nomes:
    out <- list(
      mu    = mu_raw,
      sigma = sigma
    )
    out
  }
)

# helper: dado um modelo e um nível q, calcula o quantil Q_q(x)
predict_quantile <- function(model, x, q) {
  z_q <- qnorm(q)
  out <- model(x)
  out$mu + z_q * out$sigma
}

nll_normal_hetero <- function(mu, sigma, y_true, eps = 1e-6) {
  # mu, sigma, y_true: tensores de mesma forma
  sigma <- sigma + eps
  
  term1 <- torch_log(sigma)
  term2 <- (y_true - mu)$pow(2) / (2 * sigma$pow(2))
  
  torch_mean(term1 + term2)
}