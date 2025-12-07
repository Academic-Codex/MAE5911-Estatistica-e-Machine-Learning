# Config.R
config <- list(
  seed        = 32L,
  n           = 1000L,
  x_min       = -4,
  x_max       = 4,
  noise_sd    = 0.3,
  
  q           = 0.75,
  
  # modelo
  n_hidden    = 30,

  # treino
  p_train     = 0.8,
  lr          = 1e-3,
  num_epochs  = 1000L,
  print_every = 200L,
  plot_every  = 500L

)