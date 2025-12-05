library(torch)

GPT <- torch::nn_module(
  initialize = function(block_size, n_embd, N_Layers, nvoc, N_Head, p0 = 0.1) {
    
    self$N   <- N_Layers
    self$wpe <- torch::nn_embedding(block_size, n_embd)
    self$wte <- torch::nn_embedding(nvoc, n_embd, padding_idx = 1)
    
    self$MM  <- torch::nn_module_list(lapply(1:N_Layers,
      function(x) torch::nn_multihead_attention(
        n_embd, N_Head, dropout = p0, batch_first = TRUE)))
    
    self$scale1 <- torch::nn_module_list(lapply(1:N_Layers,
      function(x) torch::nn_layer_norm(n_embd)))
    
    self$scale2 <- torch::nn_module_list(lapply(1:N_Layers,
      function(x) torch::nn_layer_norm(n_embd)))
    
    self$scale3 <- torch::nn_layer_norm(n_embd, elementwise_affine = TRUE)
    
    self$FFN <- torch::nn_module_list(lapply(1:N_Layers,
      function(x) {
        torch::nn_sequential(
          torch::nn_linear(n_embd, 4 * n_embd),
          torch::nn_gelu(),
          torch::nn_linear(4 * n_embd, n_embd),
          torch::nn_dropout(p0))}))
    
    # cabeça linear final → logits
    self$ln_f  <- torch::nn_linear(n_embd, nvoc, bias = FALSE)
    self$drop0 <- torch::nn_dropout(p = p0)
  },
  
  forward = function(x, return_intermediates = FALSE) {
    # x: [B, T]
    B <- x$size(1)
    T <- x$size(2)
    
    # posições 1..T
    x1 <- torch::torch_arange(
      1, T, dtype = torch::torch_long(), device = x$device
    )
    
    # máscara 
    wei <- torch::torch_triu(
      torch::torch_ones(T, T, device = x$device),
      diagonal = 1)$to(dtype = torch::torch_bool())
    
    # embeddings
    output <- self$wte(x) + self$wpe(x1)$unsqueeze(1)  # [B, T, E]
    output <- self$drop0(output)
    
    for (j in 1:self$N) {
      # pré-norm + atenção multi-cabeças
      # QKV <- self$scale1[[j]](output)
      # attn_out <- self$MM[[j]](
      #   query = QKV, key = QKV, value = QKV,
      #   attn_mask = wei, need_weights = FALSE
      # )[[1]]
      attn_out <- torch_zeros_like(QKV)
      output <- output + attn_out
      
      # feed-forward com pré-norm
      output <- output + self$FFN[[j]](self$scale2[[j]](output))
    }
    
    # norm final + cabeça linear → logits [B, T, nvoc]
    output <- self$scale3(output)
    logits <- self$ln_f(output)
    
    if (return_intermediates) {
      return(list(
        x1     = x1$cpu(),
        wei    = wei$to(dtype = torch_int())$cpu(),
        out    = output$cpu(),
        logits = logits$cpu()))}
    logits
  }
)