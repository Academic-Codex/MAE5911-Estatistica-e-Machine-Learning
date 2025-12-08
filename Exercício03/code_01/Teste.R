source("Model.R")
source("Train.R")

# instancia o modelo
model <- QuantileNet(n_hidden = 10)

res <- treinar_quantil(
  model,
  q      = 0.10,      # quantil desejado
  epochs = 10000,      # número de épocas
  lr     = 0.0005     # learning rate
)
