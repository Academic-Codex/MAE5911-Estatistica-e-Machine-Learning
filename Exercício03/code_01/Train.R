source("Config.R")
library(torch)
source("Data.R")
source("Model.R")

# ---- função auxiliar: quantil condicional empírico por janelas ----
plot_conditional_quantile <- function(x, y, q, num_bins = 80,
                                      col = "darkgreen", pch = 19, cex = 1.1) {
  breaks <- seq(min(x), max(x), length.out = num_bins)
  qx <- c(); qy <- c()
  
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
# -------------------------------------------------------------------
treinar_quantil <- function(model,
                            q      = config$q,
                            epochs = config$num_epochs,
                            lr     = config$lr) {
  
  sigma      <- config$noise_sd
  print_every <- config$print_every
  plot_every  <- config$plot_every
  num_bins    <- 80
  
  optimizer <- optim_adamw(
    model$parameters,
    lr = lr,
    weight_decay = 1e-3   # valor típico
  )
  
  loss_history_train <- numeric(epochs)
  loss_history_test  <- numeric(epochs)
  
  par(mfrow = c(1, 2))
  
  for (epoch in 1:epochs) {
    
    optimizer$zero_grad()
    
    # ---- forward treino ----
    y_pred_train <- model(X_train)
    loss_train   <- quantile_loss(y_pred_train, Y_train, q)
    
    loss_train$backward()
    optimizer$step()
    
    # ---- avaliação teste ----
    y_pred_test <- model(X_test)
    loss_test   <- quantile_loss(y_pred_test, Y_test, q)
    
    loss_history_train[epoch] <- loss_train$item()
    loss_history_test[epoch]  <- loss_test$item()
    
    if (epoch %% print_every == 0) {
      cat("Epoch:", epoch,
          "- Loss train:", loss_train$item(),
          "- Loss test:",  loss_test$item(), "\n")
    }
    
    # ---- gráficos durante o treino ----
    if (epoch %% plot_every == 0 || epoch == epochs) {
      
      yq_hat_full <- as.numeric(model(X_full)$squeeze())
      
      # limites de y
      ylim_all <- range(
        f_x,
        f_x + qnorm(q) * sigma,
        yq_hat_full
      )
      
      # 1) Gráfico principal: f(x), quantil empírico, NN e teórico
      plot(x, f_x,
           type = "n",
           ylim = ylim_all,
           main = paste0("Quantil ", q*100, "% - época ", epoch),
           xlab = "x", ylab = "y")
      
      # f(x) sem ruído
      lines(x, f_x, col = "black", lwd = 2)
      
      # quantil condicionado empírico
      plot_conditional_quantile(x, y, q = q,
                                num_bins = num_bins,
                                col = "tomato", pch = 19, cex = 1.1)
      
      # quantil estimado pela rede
      lines(x, yq_hat_full, col = "blue", lwd = 3)
      
      # quantil teórico: f(x) + z_q * sigma
      z_q <- qnorm(q)
      lines(x, f_x + z_q * sigma,
            col = "orange", lwd = 2, lty = 2)
      
      legend("topleft",
             legend = c("Y(x)", "Quantil empírico","Quantil teórico", "Quantil NN"),
             # cores
             col = c("black", "tomato", "orange", "blue"),
             
             # espessura da linha (NA para itens que são pontos)
             lwd = c(2, NA, 2, 3),
             
             # tipo de linha (lty=2 é pontilhado)
             lty = c(1, NA, 2, 1),
             
             # símbolo dos pontos (NA para itens que são linhas)
             pch = c(NA, 19, NA, NA),
             
             bty = "n")
      
      # 2) Evolução da loss
      plot(1:epoch, loss_history_train[1:epoch], type = "l",
           xlab = "Época", ylab = "Loss",
           ylim = range(c(loss_history_train[1:epoch],
                          loss_history_test[1:epoch])),
           main = "Evolução da perda")
      lines(1:epoch, loss_history_test[1:epoch], col = "tomato")
      legend("topright", legend = c("Train", "Test"),
             col = c("black", "tomato"), lty = 1, bty = "n")
      
      Sys.sleep(0.05)
    }
  }
  
  invisible(list(
    model      = model,
    loss_train = loss_history_train,
    loss_test  = loss_history_test
  ))
}