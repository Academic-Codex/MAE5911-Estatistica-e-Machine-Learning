set.seed(123)

theta0 <- 2        # valor verdadeiro
n      <- 2200       # tamanho amostral inicial
B      <- 10000    # número de repetições

cover_a <- cover_b <- cover_c <- 0

for (b in 1:B) {
  # 1) Amostra exponencial
  z  <- rexp(n, rate = theta0)   # atenção: rate = theta
  zbar <- mean(z)
  theta_hat <- 1 / zbar
  
  # 2) Variância assintótica de theta_hat
  var_theta_hat <- theta_hat^2 / n
  
  ## --- (a) g(theta) = P(Z > 1) = exp(-theta)
  g_a_hat  <- exp(-theta_hat)
  g_a_der  <- -exp(-theta_hat)
  var_g_a  <- (g_a_der^2) * var_theta_hat
  sd_g_a   <- sqrt(var_g_a)
  IC_a     <- c(g_a_hat - 1.96*sd_g_a,
                g_a_hat + 1.96*sd_g_a)
  g_a_true <- exp(-theta0)
  cover_a  <- cover_a + (g_a_true >= IC_a[1] &&
                           g_a_true <= IC_a[2])
  
  ## --- (b) g(theta) = P(0.1 < Z < 1)
  g_b_hat  <- exp(-0.1*theta_hat) - exp(-theta_hat)
  g_b_der  <- -0.1*exp(-0.1*theta_hat) + exp(-theta_hat)
  var_g_b  <- (g_b_der^2) * var_theta_hat
  sd_g_b   <- sqrt(var_g_b)
  IC_b     <- c(g_b_hat - 1.96*sd_g_b,
                g_b_hat + 1.96*sd_g_b)
  g_b_true <- exp(-0.1*theta0) - exp(-theta0)
  cover_b  <- cover_b + (g_b_true >= IC_b[1] &&
                           g_b_true <= IC_b[2])
  
  ## --- (c) g(theta) = Var(Z) = 1/theta^2
  g_c_hat  <- 1 / theta_hat^2
  g_c_der  <- -2 / theta_hat^3
  var_g_c  <- (g_c_der^2) * var_theta_hat
  sd_g_c   <- sqrt(var_g_c)
  IC_c     <- c(g_c_hat - 1.96*sd_g_c,
                g_c_hat + 1.96*sd_g_c)
  g_c_true <- 1 / theta0^2
  cover_c  <- cover_c + (g_c_true >= IC_c[1] &&
                           g_c_true <= IC_c[2])
}

cover_a <- cover_a / B
cover_b <- cover_b / B
cover_c <- cover_c / B
cover_a; cover_b; cover_c