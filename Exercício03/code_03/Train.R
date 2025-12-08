# Train.R  ------------------------------------------------------------

source("Config.R")
library(torch)
source("Data.R")
source("Model.R")   # aqui está o HeteroNormalNet
# e a nll_normal_hetero(mu, sigma2, y)

# ------- função de treino estilo professor --------------------------

treinar_normal_hetero <- function(model,
                                  epochs = config$num_epochs,
                                  lr     = config$lr,
                                  print_every = config$print_every,
                                  plot_every  = config$plot_every) {
  
  optimizer <- optim_adamw(model$parameters, lr = lr)
  
  loss_store_train <- numeric(epochs)
  loss_store_test  <- numeric(epochs)
  
  par(mfrow = c(1, 3))  # 3 painéis: y vs x, sigma2 vs x, loss
  
  for (epoch in 1:epochs) {
    
    ## --------- TREINO ---------
    model$train()
    optimizer$zero_grad()
    
    out_train  <- model(X_train)
    mu_train   <- out_train$mu
    sigma2_tr  <- out_train$sigma2
    
    loss_train <- nll_normal_hetero(mu_train, sigma2_tr, Y_train)
    loss_train$backward()
    optimizer$step()
    
    ## --------- TESTE ---------
    model$eval()
    out_test  <- model(X_test)
    mu_test   <- out_test$mu
    sigma2_te <- out_test$sigma2
    
    loss_test <- nll_normal_hetero(mu_test, sigma2_te, Y_test)
    
    loss_store_train[epoch] <- as.numeric(loss_train$item())
    loss_store_test[epoch]  <- as.numeric(loss_test$item())
    
    if (epoch %% print_every == 0) {
      cat("Época:", epoch,
          "- NLL train:", loss_store_train[epoch],
          "- NLL test:",  loss_store_test[epoch], "\n")
    }
    
    ## --------- GRÁFICOS (debug visual) ---------
    if (epoch %% plot_every == 0 || epoch == epochs) {
      
      # previsão em toda a grade x (para curvas lisas)
      out_full   <- model(X_full)
      mu_full    <- as.numeric(out_full$mu$squeeze())
      sigma2_hat <- as.numeric(out_full$sigma2$squeeze())
      
      # 1) painel 1: y vs x, f(x) e μ_hat(x)
      plot(x, y,
           pch = 1, col = "grey40",
           main = paste0("Média condicional - época ", epoch),
           xlab = "x", ylab = "y",
           cex = 0.7)
      lines(x, f_x, col = "black", lwd = 2)          # f(x) verdadeiro
      lines(x, mu_full, col = "red", lwd = 2)        # μ_hat(x)
      legend("topleft",
             legend = c("dados", "f(x) verdadeiro", "μ_hat(x)"),
             col    = c("grey40", "black", "red"),
             lwd    = c(NA, 2, 2),
             pch    = c(1, NA, NA),
             bty    = "n", cex = 0.8)
      
      # 2) painel 2: σ²(x) verdadeiro vs estimado
      plot(x, sigma2_true,
           type = "l", lwd = 2, col = "black",
           main = expression(sigma^2(x) * " - verdadeira vs estimada"),
           xlab = "x", ylab = expression(sigma^2))
      lines(x, sigma2_hat, col = "blue", lwd = 2)
      legend("topright",
             legend = c(expression(sigma[true]^2(x)),
                        expression(hat(sigma)^2(x))),
             col    = c("black", "blue"),
             lwd    = 2, bty = "n", cex = 0.8)
      
      # 3) painel 3: evolução da NLL (train/test)
      plot(1:epoch, loss_store_train[1:epoch],
           type = "l", lwd = 2,
           xlab = "Época", ylab = "NLL",
           main = "Evolução da NLL",
           ylim = range(c(loss_store_train[1:epoch],
                          loss_store_test[1:epoch])))
      lines(1:epoch, loss_store_test[1:epoch],
            col = "tomato", lwd = 2)
      legend("topright",
             legend = c("train", "test"),
             col    = c("black", "tomato"),
             lwd    = 2, bty = "n", cex = 0.8)
      
      Sys.sleep(0.05)  # só para conseguir ver a animação
    }
  }
  
  invisible(list(
    model      = model,
    loss_train = loss_store_train,
    loss_test  = loss_store_test
  ))
}