# Train.R

source("Config.R")
source("GPT.R")
source("Generator.R")

# ----- Carregar corpus e preparar vocabulário -----
file0 <- base::readChar(config$file_name,
                        file.info(config$file_name)$size)

voc     <- c("<PAD>", sort(unique(unlist(strsplit(file0, "")))))
Encoded <- Encoder(file0, voc)
nvoc    <- length(voc)

n <- length(Encoded)
p_train <- config$p_train

BD.train <- torch_tensor(Encoded[1:round(p_train * n)],
                         dtype = torch_int())
BD.test  <- torch_tensor(Encoded[(round(p_train * n) + 1):n],
                         dtype = torch_int())

# ----- Instanciar modelo -----
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
loss_store <- numeric(config$epochs)

# ----- Loop de treino -----
for (ep in 1:config$epochs) {
  
  idx  <- sample(
    1:(round(p_train * n) - config$batch_size),
    config$batch_size
  )
  
  idx2 <- as.integer(c(
    t(pmin(outer(as.integer(idx), 0:config$block_size, `+`), n))
  ))
  
  Z <- BD.train[idx2, drop = FALSE]$view(
    c(length(idx), config$block_size + 1)
  )
  X <- Z[, 1:config$block_size]
  Y <- Z[, 2:(config$block_size + 1)]
  
  FIT  <- Model$train()(X)
  loss <- loss_fn(FIT$flatten(end_dim = 2), Y$flatten())
  
  optimizer$zero_grad()
  loss$backward()
  optimizer$step()
  
  loss_store[ep] <- loss$item()
  cli::cli_progress_message(
    paste("Época:", ep, "- Train loss:", round(loss_store[ep], 4))
  )
}

# depois do treino, você pode testar:
cat(generate("A"), "\n")
# ou
cat(generate_topk("A"), "\n")