n=1
theta0=runif(1,0,100)
Z = rpois(n, theta0)
M=10000
t=numeric(M)
for (i in 1: M){
  Z = rpois(n, theta0)
  t[i]=sqrt(n)*(mean(Z)-theta0)/sqrt(theta0)
}
hist(t, prob=TRUE)
curve(dnorm,  add=TRUE, col="tomato", lwd=2)

