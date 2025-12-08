

source("Config.R")
library(torch)
source("Data.R")
source("Model.R")  
source("Train.R")

# cria o modelo
model <- HeteroNormalNet(input_dim = 1,
                         hidden_mu = 64,
                         hidden_sigma = 64)

# treina
res <- treinar_normal_hetero(
  model,
  epochs      = 3000,      # por ex.
  lr          = 5e-3
)