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
  epochs     = 10000,
  lr         = 2e-3
)


# 5) (Opcional) salvar o modelo treinado ---------------------------
torch_save(model, "modelo_quantil_075.pt")

# 6) Exemplo de uso do modelo treinado -----------------------------

model$eval()

# novos pontos onde você quer o quantil
x_new <- seq(config$x_min, config$x_max, length.out =1000)

# quantil estimado (vetor R)
q_hat_vec <- predict_quantile_vec(model, x_new)

# se quiser visualizar rápido:
plot(x, y, pch = 16, col = rgb(0,0,0,0.1),
     xlab = "x", ylab = "y", main = "Quantil 75% - modelo treinado")
lines(x, f_x, col = "black", lwd = 2)
lines(x_new, q_hat_vec, col = "blue", lwd = 2)

sigma2_hat_vec <- predict_sigma2_vec(model, x_new)

plot(x, sigma2_true, type = "l", lwd = 2,
     xlab = "x", ylab = expression(sigma^2),
     main = expression(sigma(x)^2 ~ ": verdadeiro vs aprendido"))
lines(x_new, sigma2_hat_vec, col = "blue", lwd = 2)

legend("topleft",
       legend = c(expression(sigma(x)^2),
                  expression(hat(sigma)(x)^2)),
       col = c("black", "blue"),
       lwd = 2, bty = "n")