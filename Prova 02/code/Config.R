config <- list(
  #Corpus for training (global)
  file_name = "Shakespeare.txt",
  train = !TRUE,
  run = TRUE,
  read_weights = !TRUE,
  p_train = 0.8,
  k_top=2,
  
  #gpt parameters (global)
  block_size = 16,   #Maximum context
  n_embd = 128,      #Embedding dimension
  N_Layers = 2,      #Number of layers
  Head = 2,          #Number of heads
  
  #Training parameters (global)
  lr = 0.003,        #Learning rate
  batch_size = 64,   #Batch size
  p0 = 0.2,          #Dropout proportion
  epochs = 1200,        #Number of epochs
  num_workers = 6,  #Number of cpu workers
  
  max_new_tokens = 700
)