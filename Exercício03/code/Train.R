# Train.R
source("Config.R")
library(torch)
source("Data.R")
source("Model.R")


# ---- função auxiliar: quantil condicional empírico por janelas ----
plot_conditional_quantile <- function(x, y, q = 0.75, num_bins = 80,
                                      col = "darkgreen", pch = 19, cex = 1.1) {
  breaks <- seq(min(x), max(x), length.out = num_bins)
  qx <- c()
  qy <- c()
  
  for (i in 1:(length(breaks) - 1)) {
    idx <- which(x >= breaks[i] & x < breaks[i + 1])
    if (length(idx) > 0) {
      qx <- c(qx, mean(x[idx]))
      qy <- c(qy, quantile(y[idx], q))
    }
  }
  
  points(qx, qy, col = col, pch = pch, cex = cex)
  
  invisible(list(x = qx, y = qy))
}
# --------------------------------------------------------------------

sigma0 <- config$noise_sd
q <- config$q

model <- QuantileNet(n_hidden = config$n_hidden)
optimizer <- optim_adam(model$parameters, lr = config$lr)

loss_history_train <- numeric(config$num_epochs)
loss_history_test  <- numeric(config$num_epochs)

par(mfrow = c(1, 2))

for (epoch in 1:config$num_epochs) {
  optimizer$zero_grad()
  
  # ---- forward treino ----
  # para o modelo "Normal-likelihood"
  # y_pred_train <- model(X_train)
  # loss_train   <- normal_quantile_loss(y_pred_train, Y_train, q, sigma0)
  y_pred_train <- model(X_train)
  loss_train   <- quantile_loss(y_pred_train, Y_train, q)
  
  loss_train$backward()
  optimizer$step()
  
  # ---- avaliação teste ----
  # y_pred_test <- model(X_test)
  # loss_test   <- normal_quantile_loss(y_pred_test, Y_test, q, sigma0)
  y_pred_test <- model(X_test)
  loss_test   <- quantile_loss(y_pred_test, Y_test, q)
  
  loss_history_train[epoch] <- loss_train$item()
  loss_history_test[epoch]  <- loss_test$item()
  
  if (epoch %% config$print_every == 0) {
    cat("Epoch:", epoch,
        "- Loss train:", loss_train$item(),
        "- Loss test:",  loss_test$item(), "\n")
  }
  
  # ---- gráficos durante o treino ----
  if (epoch %% config$plot_every == 0 || epoch == config$num_epochs) {
    # predição no grid completo
    yq_hat_full <- as.numeric(model(X_full)$squeeze())
    
    # 1) Dados + função verdadeira + quantil aprendido
    plot(x, y, pch = 16, col = rgb(0,0,0,0.3),
         main = paste0("Quantil ", q*100, "% - época ", epoch),
         xlab = "x", ylab = "y")
    
    # função determinística sem ruído
    lines(x, f_x, col = "black", lwd = 2)
    
    # quantil condicional empírico
    plot_conditional_quantile(x, y, q = q,
                              num_bins = 80,
                              col = "tomato", pch = 19, cex = 1.0)
    
    # quantil estimado pela rede
    lines(x, yq_hat_full, col = "blue", lwd = 3)
    
    # (opcional) quantil teórico 75%: f(x) + z_q * sigma
    z_q <- qnorm(q)
    lines(x, f_x + z_q * config$noise_sd,
          col = "orange", lwd = 2, lty = 2)

    # 2) Loss treino x teste
    plot(1:epoch, loss_history_train[1:epoch], type = "l",
         xlab = "Época", ylab = "Loss",
         ylim = range(c(loss_history_train[1:epoch],
                        loss_history_test[1:epoch])),
         main = "Evolução da perda")
    lines(1:epoch, loss_history_test[1:epoch], col = "red")
    legend("topright", legend = c("Train", "Test"),
           col = c("black", "red"), lty = 1, bty = "n")
    
    Sys.sleep(0.05) # animação
  }
}