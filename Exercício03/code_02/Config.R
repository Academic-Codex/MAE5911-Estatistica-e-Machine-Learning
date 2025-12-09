  # Config.R
  
  config <- list(
    # quantil que você quer olhar nos gráficos (não entra na NLL)
    q = 0.75,
    
    # treino
    num_epochs  = 2500,
    lr          = 1e-3,
    print_every = 10L,
    plot_every  = 10L,
    
    # arquitetura
    input_dim   = 1L,
    hidden_mu   = 10L,
    hidden_sigma= 10L,
    
    
    seed        = 32,
    n           = 1000,
    x_min       = -4,
    x_max       =  4,
    p_train     = 0.7,
    
    # base da variância
    noise_sd    = 0.3,
    
    # se TRUE, faz variância heteroscedástica
    hetero      = FALSE,        # FALSE => variância constante
    hetero_alpha= 0.5          # força da heteroscedasticidade
    
  
  )