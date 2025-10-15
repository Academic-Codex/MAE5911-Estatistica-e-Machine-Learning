## --- parâmetros ---
n      = 100      # tamanho da amostra em cada simulação
M      = 10000    # número de simulações (experimentos)
theta0 = 1        # média verdadeira da Exponencial (pode trocar)

## Se quiser θ0 aleatório em cada execução:
# theta0 <- runif(1, 1, 100)

## --- simulação ---
t = numeric(M)

for (i in 1:M) {
  z = rexp(n, rate = 1/theta0)                 # Exp(com média θ0)
  t[i] = sqrt(n) * (mean(z) - theta0) / theta0 # padronização → N(0,1)
}

## --- gráfico ---
hist(t, prob = TRUE,
     main = "TCL: média da Exponencial (θ0 = média)",
     xlab = "t = sqrt(n) * (mean(Z) - θ0) / θ0")
curve(dnorm(x), add = TRUE, col = "tomato", lwd = 2)

## checagem numérica
cat("média(t) ≈", mean(t), "\n")
cat("dp(t)    ≈", sd(t),   " (esperado ~ 1)\n")