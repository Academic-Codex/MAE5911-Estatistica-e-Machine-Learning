config <- list(
  #Corpus for training
  file_name = "Shakespeare.txt",
  train = !TRUE,
  run = TRUE,
  read_weights = !TRUE,
  p_train = 0.8,
  k_top=2,
  
  #gpt parameters
  block_size = 50,   #Maximum context
  n_embd = 128,      #Embedding dimension
  N_Layers = 2,      #Number of layers
  N_Head = 2,        #Number of heads
  
  #Training parameters 
  lr = 0.003,        #Learning rate
  batch_size = 64,   #Batch size
  p0 = 0.2,          #Dropout proportion
  epochs = 1200,     #Number of epochs
  num_workers = 6,   #Number of cpu workers
  
  max_new_tokens = 1500
)