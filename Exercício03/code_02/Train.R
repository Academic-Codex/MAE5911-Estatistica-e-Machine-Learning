# Train.R
source("Config.R")
library(torch)
source("Data.R")
source("Model.R")
# se a nll_normal_hetero estiver em outro arquivo, algo como:
# source("Loss.R")

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

treinar_normal_hetero <- function(model,
                                  epochs = config$num_epochs,
                                  lr     = config$lr,
                                  q_plot = config$q) {
  
  print_every <- config$print_every
  plot_every  <- config$plot_every
  num_bins    <- 80
  
  optimizer <- optim_adamw(
    model$parameters,
    lr = lr,
    weight_decay = 1e-3
  )
  
  loss_history_train <- numeric(epochs)
  loss_history_test  <- numeric(epochs)
  
  par(mfrow = c(1, 2))
  
  for (epoch in 1:epochs) {
    
    # ------- TREINO -------
    model$train()
    optimizer$zero_grad()
    
    out_train <- model(X_train)
    mu_train    <- out_train$mu
    sigma_train <- out_train$sigma
    
    loss_train <- nll_normal_hetero(mu_train, sigma_train, Y_train)
    
    loss_train$backward()
    optimizer$step()
    
    # ------- TESTE -------
    model$eval()
    out_test <- model(X_test)
    mu_test    <- out_test$mu
    sigma_test <- out_test$sigma
    
    loss_test <- nll_normal_hetero(mu_test, sigma_test, Y_test)
    
    loss_history_train[epoch] <- loss_train$item()
    loss_history_test[epoch]  <- loss_test$item()
    
    if (epoch %% print_every == 0) {
      cat("Epoch:", epoch,
          "- Loss train:", loss_train$item(),
          "- Loss test:",  loss_test$item(), "\n")
    }
    
    # ------- GRÁFICOS -------
    if (epoch %% plot_every == 0 || epoch == epochs) {
      
      # predição no grid completo
      out_full <- model(X_full)
      mu_full    <- as.numeric(out_full$mu$squeeze())
      sigma_full <- as.numeric(out_full$sigma$squeeze())
      
      z_q <- qnorm(q_plot)
      
      # >>> alteração 1: quantil teórico usando a variância VERDADEIRA do simulador
      # (sigma_true vem do Data.R)
      q_theo <- f_x + z_q * sigma_true
      
      # quantil estimado pelo modelo (usa mu_full e sigma_full da rede)
      q_hat <- mu_full + z_q * sigma_full
      
      # >>> alteração 2: incluir q_theo nos limites do gráfico
      ylim_all <- range(
        f_x,
        q_theo,
        q_hat,
        y
      )
      
      # 1) Gráfico principal
      plot(x, f_x,
           type = "n",
           ylim = ylim_all,
           main = paste0("NLL Normal (q=", q_plot*100, "%) - época ", epoch),
           xlab = "x", ylab = "y")
      
      # f(x) sem ruído
      lines(x, f_x, col = "black", lwd = 2)
      
      # quantil empírico com janelas (sobre os dados observados)
      plot_conditional_quantile(x, y, q = q_plot,
                                num_bins = num_bins,
                                col = "tomato", pch = 19, cex = 1.1)
      
      # quantil teórico (f(x) + z_q * sigma_true)
      lines(x, q_theo, col = "orange", lwd = 2, lty = 2)
      
      # quantil estimado pela rede
      lines(x, q_hat, col = "blue", lwd = 3)
      
      legend("topleft",
             legend = c("f(x)", "Quantil empírico",
                        "Quantil teórico (f+zσ_true)", "Quantil NN"),
             col = c("black", "tomato", "orange", "blue"),
             lwd = c(2, NA, 2, 3),
             lty = c(1, NA, 2, 1),
             pch = c(NA, 19, NA, NA),
             bty = "n")
      
      # 2) Evolução da loss
      plot(1:epoch, loss_history_train[1:epoch], type = "l",
           xlab = "Época", ylab = "NLL",
           ylim = range(c(loss_history_train[1:epoch],
                          loss_history_test[1:epoch])),
           main = "Evolução da NLL")
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