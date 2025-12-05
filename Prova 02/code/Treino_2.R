# ============================
# Train.R  —  Treinamento GPT
# ============================

source("Config.R")
source("GPT.R")
source("Generator.R")

library(torch)
library(cli)

# ------------------------------------------
# 1) Carregar corpus e preparar vocabulário
# ------------------------------------------

file0 <- base::readChar(
  config$file_name,
  file.info(config$file_name)$size
)

voc     <- c("<PAD>", sort(unique(unlist(strsplit(file0, "")))))
Encoded <- Encoder(file0, voc)
nvoc    <- length(voc)

n       <- length(Encoded)
p_train <- config$p_train

# ----- Separar treino e teste -----
n_train <- round(p_train * n)

BD.train <- torch_tensor(Encoded[1:n_train], dtype = torch_int())
BD.test  <- torch_tensor(Encoded[(n_train + 1):n], dtype = torch_int())

n_test_total <- BD.test$size()[1]


# ------------------------------------------
# 2) Instanciar modelo e estruturas de treino
# ------------------------------------------

Model <- GPT(
  block_size = config$block_size,
  n_embd     = config$n_embd,
  N_Layers   = config$N_Layers,
  nvoc       = nvoc,
  Head       = config$Head,
  p0         = config$p0
)

optimizer <- torch::optim_adamw(Model$parameters, lr = config$lr)
loss_fn   <- torch::nn_cross_entropy_loss()

loss_store      <- numeric(config$epochs)
loss_store_test <- numeric(config$epochs)

batch_train <- config$batch_size
batch_test  <- config$batch_size  # pode criar outro no Config.R se quiser


# ===========================
# 3) LOOP DE TREINAMENTO
# ===========================
for (ep in 1:config$epochs) {
  
  ## -------------------------
  ## 3.1  MINIBATCH DE TREINO
  ## -------------------------
  idx <- sample(
    1:(n_train - config$block_size - 1),
    batch_train
  )
  
  idx2 <- as.integer(c(
    t(outer(as.integer(idx), 0:config$block_size, `+`))
  ))
  
  Z <- BD.train[idx2, drop = FALSE]$view(
    c(length(idx), config$block_size + 1)
  )
  X <- Z[, 1:config$block_size]
  Y <- Z[, 2:(config$block_size + 1)]
  
  # forward/backward treino
  FIT  <- Model$train()(X)
  loss <- loss_fn(FIT$flatten(end_dim = 2), Y$flatten())
  
  optimizer$zero_grad()
  loss$backward()
  optimizer$step()
  
  loss_store[ep] <- loss$item()
  
  
  ## -------------------------
  ## 3.2  MINIBATCH DE TESTE
  ## -------------------------
  with_no_grad({
    idx_t <- sample(
      1:(n_test_total - config$block_size - 1),
      batch_test
    )
    
    idx2_t <- as.integer(c(
      t(outer(as.integer(idx_t), 0:config$block_size, `+`))
    ))
    
    Zt <- BD.test[idx2_t, drop = FALSE]$view(
      c(length(idx_t), config$block_size + 1)
    )
    
    X_test <- Zt[, 1:config$block_size]
    Y_test <- Zt[, 2:(config$block_size + 1)]
    
    logits_test <- Model$eval()(X_test)
    Lte <- loss_fn(
      logits_test$flatten(end_dim = 2),
      Y_test$flatten()
    )
  })
  
  loss_store_test[ep] <- Lte$item()
  
  
  ## -------------------------
  ## 3.3  PRINT BONITO
  ## -------------------------
  cli::cli_progress_message(
    paste0(
      "Época: ", ep,
      " | Train loss: ", round(loss_store[ep], 4),
      " | Test loss: ",  round(loss_store_test[ep], 4)
    )
  )
  
  
  ## -------------------------
  ## 3.4  GRÁFICO (a cada 10 épocas)
  ## -------------------------
  if (ep %% 10 == 0) {
    
    ylim_range <- range(
      c(loss_store[1:ep], loss_store_test[1:ep]),
      na.rm = TRUE
    )
    
    plot(
      1:ep, loss_store[1:ep],
      type = "l",
      lwd  = 2,
      col  = "black",
      xlab = "Época",
      ylab = "Loss",
      main = "Evolução da perda (Train vs Test)",
      ylim = ylim_range
    )
    
    lines(
      1:ep, loss_store_test[1:ep],
      col  = "tomato",
      lwd  = 2
    )
  }
}

cat("\nTreinamento finalizado!\n")