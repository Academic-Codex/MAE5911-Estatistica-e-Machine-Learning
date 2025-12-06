source("Config.R")
source("Data.R")
source("Model.R")   # aqui está QuantileNet e quantile_loss

q <- config$q

model <- QuantileNet(n_hidden = config$n_hidden)
optimizer <- optim_adam(model$parameters, lr = config$lr)

loss_history_train <- numeric(config$num_epochs)
loss_history_test  <- numeric(config$num_epochs)

# abre janela de gráfico só uma vez
par(mfrow = c(1, 2))

for (epoch in 1:config$num_epochs) {
  optimizer$zero_grad()
  
  # ---- forward no treino ----
  y_pred_train <- model(X_train)
  loss_train   <- quantile_loss(y_pred_train, Y_train, q)
  
  loss_train$backward()
  optimizer$step()
  
  # ---- avalia no teste (sem gradiente) ----
  y_pred_test <- model(X_test)
  loss_test   <- quantile_loss(y_pred_test, Y_test, q)
  
  loss_history_train[epoch] <- loss_train$item()
  loss_history_test[epoch]  <- loss_test$item()
  
  if (epoch %% config$print_every == 0) {
    cat("Epoch:", epoch,
        "- Loss train:", loss_train$item(),
        "- Loss test:",  loss_test$item(), "\n")
  }
  
  # ---- atualizar gráficos durante o treino ----
  if (epoch %% config$plot_every == 0 || epoch == config$num_epochs) {
    # predição no grid completo (pra curva ficar suave)
    yq_hat_full <- as.numeric(model(X_full)$squeeze())
    
    # painel 1: dados + curva do quantil
    plot(x, y, pch = 16, col = rgb(0,0,0,0.3),
         main = paste0("Quantil ", q*100, "% - época ", epoch),
         xlab = "x", ylab = "y")
    lines(x, yq_hat_full, col = "blue", lwd = 3)
    
    # painel 2: loss treino x teste
    plot(1:epoch, loss_history_train[1:epoch], type = "l",
         xlab = "Época", ylab = "Loss",
         ylim = range(c(loss_history_train[1:epoch],
                        loss_history_test[1:epoch])),
         main = "Evolução da loss")
    lines(1:epoch, loss_history_test[1:epoch], col = "red")
    legend("topright", legend = c("Train", "Test"),
           col = c("black", "red"), lty = 1, bty = "n")
    
    Sys.sleep(0.05)  # opcional, só pra você ver a animação
  }
}