source("Config.R")
source("GPT.R")
source("Generator.R")

prompt <- "A"
# cat(generate(prompt), "\n")
cat(generate_topk(prompt), "\n")
