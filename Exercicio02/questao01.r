## =======================================
## Distribuição de Bernoulli: forma da PMF
## =======================================
library(ggplot2)
library(dplyr)

# Valores de theta a visualizar
thetas <- c(0.1, 0.3, 0.5, 0.7, 0.9)

# Constrói a tabela com P(Z=0)=1-theta e P(Z=1)=theta
bern_data <- expand.grid(Z = c(0,1), theta = thetas) %>%
  mutate(prob = ifelse(Z == 1, theta, 1 - theta))

# Gráfico de barras da PMF
p <- ggplot(bern_data, aes(factor(Z), prob, fill = factor(Z))) +
  geom_col(width = 0.6, alpha = 0.8) +
  geom_text(aes(label = round(prob, 2)), vjust = -0.3, size = 4) +
  facet_wrap(~theta, nrow = 1) +
  scale_fill_manual(values = c("#7aa6f9", "#f57b7b"), labels = c("Z=0", "Z=1")) +
  labs(title = "Distribuição Bernoulli(θ)",
       subtitle = "Cada painel mostra P(Z=0)=1-θ e P(Z=1)=θ",
       x = "Valor de Z", y = "Probabilidade",
       fill = "Evento") +
  theme_minimal(base_size = 14)

print(p)

