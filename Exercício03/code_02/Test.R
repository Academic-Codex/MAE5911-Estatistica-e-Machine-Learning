rm(list = ls())
graphics.off()

source("Config.R")
library(torch)
source("Data.R")
source("Model.R")   # aqui já estão HeteroNormalNet, QuantileNormalNet, etc.
source("Train.R")

# ------------------------------------------------------------------
# escolhe o nível de quantil (pego do Config, mas pode ser fixo)
q_level <- config$q  # por ex., 0.75

# cria o modelo de quantil + variância heteroscedástica
model <- QuantileNormalNet(
  input_dim    = 1,
  hidden_q     = 10,
  hidden_sigma = 10
)

# treina
res <- treinar_normal_quantil(
  model      = model,
  q_level    = q_level,
  epochs     = 500,
  lr         = 5e-3
)