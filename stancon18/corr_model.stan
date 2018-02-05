data{
 int<lower=0> N;
 int<lower=0> Kb;
 int<lower=0> Kc;
 int<lower=0, upper=1> yb[N,Kb];
 real yc[N,Kc];
}

transformed data{
  int<lower=1> K = Kb + Kc;
}
parameters {
  vector[Kb] mu_b;
  vector[Kc] mu_c;

  cholesky_factor_corr[K] L_R;
  matrix[N, K] Z_raw;
  vector<lower=0>[Kc] sigma;
}

model{
  matrix[N,K] Z = Z_raw * transpose(L_R);
  mu_b ~ normal(0,1);
  mu_c ~ normal(0,10);
  sigma ~ normal(0, 5);
  L_R ~ lkj_corr_cholesky(1);
  to_vector(Z_raw) ~ normal(0,1);
  for (k in 1:Kb) yb[,k] ~ bernoulli_logit(mu_b[k] + Z[,k]);
  for (k in 1:Kc) yc[,k] ~ normal(mu_c[k] + Z[,k + Kb], sigma[k]);
}

generated quantities{
  corr_matrix[K] R = multiply_lower_tri_self_transpose(L_R);
}
