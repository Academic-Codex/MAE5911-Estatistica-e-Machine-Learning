n <- 10
theta <- 0.3
k <- 0:n
binom <- data.frame(
  k = k,
  P = dbinom(k, size=n, prob=theta)
)

ggplot(binom, aes(k, P)) +
  geom_segment(aes(xend=k, yend=0), linewidth=1.2, color="#3c78d8") +
  geom_point(size=3, color="#f44336") +
  labs(title=paste0("Distribuição Binomial(n=", n, ", θ=", theta, ")"),
       x="Número de sucessos (k)", y="Probabilidade") +
  theme_minimal(base_size=14)