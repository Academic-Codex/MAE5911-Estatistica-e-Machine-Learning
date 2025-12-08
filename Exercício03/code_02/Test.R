source("Config.R")
source("Train.R")
source("Model.R")
source("Data.R")

model <- HeteroNormalNet(
  input_dim    = config$input_dim,
  hidden_mu    = config$hidden_mu,
  hidden_sigma = config$hidden_sigma
)

res <- treinar_normal_hetero(model)