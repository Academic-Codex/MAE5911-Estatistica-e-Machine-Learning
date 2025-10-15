library(ggplot2)

theta <- 0.3
bern <- data.frame(
  Z = c(0, 1),
  P = c(1 - theta, theta)
)

ggplot(bern, aes(x=factor(Z), y=P, fill=factor(Z))) +
  geom_col(width=0.5) +
  geom_text(aes(label=round(P, 2)), vjust=-0.3, size=5) +
  scale_fill_manual(values=c("#6fa8dc","#f78181"), labels=c("Z=0","Z=1")) +
  labs(title=paste0("Distribuição Bernoulli(θ=", theta, ")"),
       x="Z", y="Probabilidade") +
  theme_minimal(base_size=14)