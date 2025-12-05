# Generator.R

# ---- Encoder / Decoder de caracteres ----
Encoder <- function(file, vocabulary) {
  file  <- unlist(strsplit(file, ""))
  filex <- numeric(length(file))
  for (i in seq_along(vocabulary)) {
    filex[file == vocabulary[i]] <- i
  }
  filex
}

Decoder <- function(file, vocabulary) {
  filex <- file
  for (i in seq_along(vocabulary)) {
    filex[file == i] <- vocabulary[i]
  }
  filex
}

# ---- Geração greedy (argmax) ----
generate <- function(prompt, max_new = config$max_new_tokens) {
  Model$eval()
  with_no_grad({
    x <- torch_tensor(Encoder(prompt, voc), dtype = torch_int())$unsqueeze(1)  # [1,T]
    
    for (i in 1:max_new) {
      # contexto (janela)
      if (x$size(2) <= config$block_size) {
        ctx <- x
      } else {
        T <- x$size(2)
        ctx <- x[, (T - config$block_size + 1):T]
      }
      
      # forward
      logits <- Model(ctx)              # [1, T_ctx, V]
      last_logits <- logits[, -1, ]     # [1, V]
      
      # próximo token (argmax)
      next_token <- torch_argmax(last_logits, dim = -1)$unsqueeze(2)  # [1,1]
      x <- torch_cat(list(x, next_token), dim = 2)
    }
    
    generated_idx <- as.integer(as_array(x$squeeze(1)))
    paste(voc[generated_idx], collapse = "")
  })
}

# ---- Geração com top-k sampling ----
generate_topk <- function(prompt,
                          k_top   = config$k_top,
                          max_new = config$max_new_tokens) {
  Model$eval()
  with_no_grad({
    x <- torch_tensor(Encoder(prompt, voc), dtype = torch_int())$unsqueeze(1)

    for (i in 1:max_new) {
      if (x$size(2) <= config$block_size) {
        logits <- Model$eval()(x)[, -1, ]
      } else {
        xx     <- x[, (x$size(2) - config$block_size + 1):x$size(2)]
        logits <- Model$eval()(xx)[, -1, ]
      }

      top <- logits$topk(k_top)
      vals  <- top[[1]]$to(dtype = torch_float())
      probs <- torch::nnf_softmax(vals, dim = -1)
      selected   <- torch_multinomial(probs, num_samples = 1)
      next_token <- top[[2]][, selected$item()]$unsqueeze(1)
      x <- torch_cat(list(x, next_token), dim = 2)
    }

    generated_idx <- as.integer(as_array(x$squeeze(1)))
    paste(voc[generated_idx], collapse = "")
  })
}

# ---- Geração com top-k sampling (versão da aula) ----
# generate_topk <- function(prompt,
#                           k_top = config$k_top) {
#   Model$eval()
#   with_no_grad({
#     
#     # prompt -> índices
#     x <- torch_tensor(
#       Encoder(prompt, voc),
#       dtype = torch_int()
#     )$unsqueeze(1)  # [1 x T]
#     
#     for (i in 1:config$max_new_tokens) {
#       
#       if (x$size(2) <= config$block_size) {
#         logits <- Model$eval()(x)[, -1, ]          # 1) forward no contexto inteiro
#         logits <- logits$topk(k_top)               # 2) top-k
#         probs    <- torch::nnf_softmax(logits[[1]], dim = -1)  # 3) softmax nos k
#         selected <- torch::torch_multinomial(probs, num_samples = 1)  # 4) amostra 1
#         next_token <- logits[[2]][, selected$item()]$unsqueeze(1)     # 5) pega índice k
#         # linha extra que o prof tinha (greedy) — se quiser, descomenta:
#         # next_token <- torch_argmax(Model$eval()(x)[, -1, ], -1)$unsqueeze(1)  # 6)
#         x <- torch_cat(list(x, next_token), dim = 2)  # 7) concatena
#       } else {
#         xx <- x[, (x$size(2) - config$block_size + 1):x$size(2)]  # 1) recorte janela
#         logits <- Model$eval()(xx)[, -1, ]                        # 2) forward na janela
#         logits <- logits$topk(k_top)                              # 3) top-k
#         probs    <- torch::nnf_softmax(logits[[1]], dim = -1)     # 4) softmax
#         selected <- torch::torch_multinomial(probs, num_samples = 1)  # 5) amostra
#         next_token <- logits[[2]][, selected$item()]$unsqueeze(1)     # 6) índice k
#         x <- torch_cat(list(x, next_token), dim = 2)              # 7) concatena
#       }
#     }
#     
#     # decodifica sequência inteira (não só o último token)
#     idx <- as.integer(as_array(x$squeeze(1)))
#     paste(Decoder(idx, voc), collapse = "")
#   })
# }


# for (i in 1:config$max_new_tokens) {
#   if (x$size(2)<-config$block_size) {
#     logits = Model$eval()(x)[,-1,]
#     logits = logits$topk(2)
#     probs = torch::nnf_softmax(logits[[1]],-1)
#     selected = torch::torch_multinomial(probs, num_samples=1)
#     next_token <- logits[[2]][,selected$item()]$unsqueeze(1)
#     next_token = torch_argmax(Model$eval()(x)[,-1,],-1)$unsqueeze(1)
#     x <- torch_cat(list(x, next_token), -1)
#   } else {
#     xx = x[, (x$size(2)*config$block_size+1):x$size(2)]
#     logits = Model$eval()(xx)[,-1,]
#     logits = logits$topk(k_top)
#     probs = torch::nn_softmax(logits[[1]],-1)
#     selected = torch::torch_multinomial(probs, num_samples=1)
#     next_token <- logits[[2]][,selected$item()]$unsqueeze(1)
#     x <- torch_cat(list(x, next_token), -1)
#   }
#   cat(Decoder(as.number(next_token)))
# }