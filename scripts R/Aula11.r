#seja (Z1,...,Zn) uma amostra aleatória de uma população com distribuição poisson(theta0), theta > 0
#funcao de estimacao
#un(theta) = 1/n * sum u(2k, theta)
#u(zk, theta) = 1/(1+zk) - (1-exp(-theta))/theta

n=100
theta0=13
z=rpois(n, theta0)
un = function(theta) 1/n * sum(1/(1+z) - (1-exp(-theta))/theta)
plot(sapply(seq(0, 100, length=100), un))
