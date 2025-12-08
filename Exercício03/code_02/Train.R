# Train.R
library(torch)

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

treinar_normal_quantil <- function(model,
                                   q_level = config$q,
                                   epochs  = config$num_epochs,
                                   lr      = config$lr) {
  
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
  
  # 3 painéis: média/quantil, sigma2, NLL
  par(mfrow = c(1, 3))
  
  for (epoch in 1:epochs) {
    
    ## ---------- TREINO ----------
    model$train()
    optimizer$zero_grad()
    
    out_train <- model(X_train)
    Qq_train  <- out_train$q
    sigma2_tr <- out_train$sigma2
    
    loss_train <- nll_normal_from_quantile(
      Qq      = Qq_train,
      sigma2  = sigma2_tr,
      y       = Y_train,
      q_level = q_level
    )
    
    loss_train$backward()
    optimizer$step()
    
    ## ---------- TESTE ----------
    model$eval()
    out_test  <- model(X_test)
    Qq_test   <- out_test$q
    sigma2_te <- out_test$sigma2
    
    loss_test <- nll_normal_from_quantile(
      Qq      = Qq_test,
      sigma2  = sigma2_te,
      y       = Y_test,
      q_level = q_level
    )
    
    loss_history_train[epoch] <- loss_train$item()
    loss_history_test[epoch]  <- loss_test$item()
    
    if (epoch %% print_every == 0) {
      cat("Época:", epoch,
          "- NLL train:", loss_train$item(),
          "- NLL test:",  loss_test$item(), "\n")
    }
    
    ## ---------- GRÁFICOS ----------
    if (epoch %% plot_every == 0 || epoch == epochs) {
      
      # predição no grid completo
      out_full   <- model(X_full)
      Qq_full    <- as.numeric(out_full$q$squeeze())
      sigma2_hat <- as.numeric(out_full$sigma2$squeeze())
      sigma_hat  <- sqrt(sigma2_hat)
      
      # --- quantil teórico usando a variância verdadeira heteroscedástica ---
      z_q        <- qnorm(q_level)
      # sigma_true e sigma2_true vêm do Data.R
      q_theo     <- f_x + z_q * sigma_true
      
      ylim_all <- range(
        y,
        f_x,
        q_theo,
        Qq_full
      )
      
      ## Painel 1: f(x), dados, quantil empírico, quantil teórico, quantil NN
      plot(x, y,
           pch  = 16,
           col  = rgb(0,0,0,0.15),
           ylim = ylim_all,
           xlab = "x", ylab = "y",
           main = paste0("Média/Quantil condicional - época ", epoch))
      
      # f(x) sem ruído
      lines(x, f_x, col = "black", lwd = 2)
      
      # quantil condicionado empírico
      plot_conditional_quantile(x, y,
                                q        = q_level,
                                num_bins = num_bins,
                                col      = "tomato", pch = 19, cex = 1.1)
      
      # quantil teórico (f(x) + z_q * sigma_true(x))
      lines(x, q_theo, col = "orange", lwd = 2, lty = 2)
      
      # quantil estimado pela rede
      lines(x, Qq_full, col = "blue", lwd = 3)
      
      legend("topleft",
             legend = c("dados", "f(x)", "Quantil empírico",
                        "Quantil teórico", "Quantil NN"),
             col    = c(rgb(0,0,0,0.3), "black", "tomato", "orange", "blue"),
             lwd    = c(NA, 2, NA, 2, 3),
             lty    = c(NA, 1, NA, 2, 1),
             pch    = c(16, NA, 19, NA, NA),
             bty    = "n")
      
      ## Painel 2: σ^2(x) verdadeira vs estimada (heteroscedástico)
      plot(x, sigma2_true,
           type = "l", lwd = 2,
           xlab = "x", ylab = expression(sigma^2),
           main = expression(sigma^2(x)~": verdadeira vs estimada"))
      lines(x, sigma2_hat, col = "blue", lwd = 2)
      legend("topleft",
             legend = c(expression(sigma[true]^2(x)),
                        expression(hat(sigma)^2(x))),
             col = c("black", "blue"),
             lwd = 2, bty = "n")
      
      ## Painel 3: evolução da NLL
      plot(1:epoch, loss_history_train[1:epoch], type = "l",
           xlab = "Época", ylab = "NLL",
           ylim = range(c(loss_history_train[1:epoch],
                          loss_history_test[1:epoch])),
           main = "Evolução da NLL")
      lines(1:epoch, loss_history_test[1:epoch], col = "tomato")
      legend("topright", legend = c("Train", "Test"),
             col = c("black", "tomato"),
             lty = 1, bty = "n")
      
      Sys.sleep(0.05)
    }
  }
  
  invisible(list(
    model      = model,
    loss_train = loss_history_train,
    loss_test  = loss_history_test
  ))
}