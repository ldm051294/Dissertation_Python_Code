# -*- coding: utf-8 -*-
"""
@author: Luigi Di Mauro
"""

import numpy as np
import matplotlib.pyplot as plt

# Given parameters
k = 2.0
sigma = 1.0
Wt = 1.0
tau = 0.0001

# Function to calculate lambda1, lambda2, and D(tau)
def calculate_lambdas_and_D(gamma):
    A = 2 * sigma**2 * (1 - 1 / (gamma * (gamma - 1)))
    B = 2 * k * (-1 + 1 / (gamma * (gamma - 1)))
    C = -k**2 / (2 * gamma * (gamma - 1) * sigma**2)
    
    discriminant = np.sqrt(B**2 - 4*A*C)
    lambda1 = (-B - discriminant) / 2
    lambda2 = (-B + discriminant) / 2
    D_tau = np.exp(discriminant * tau)
    
    return lambda1, lambda2, D_tau

# Function to calculate beta(tau)
def beta_tau(C, lambda1, lambda2, D_tau):
    return C * (1 - D_tau) / (lambda1 - lambda2 * D_tau)

# Function to calculate J_t
def J_t(gamma, Xt, lambda1, lambda2, D_tau, C):
    exponent1 = sigma**2 * ((C * (lambda2 - lambda1) / lambda2) * np.log(abs((lambda1 - lambda2 * D_tau) / (lambda1 - lambda2))) + (C * (1 - D_tau))) 
    exponent2 = Xt**2 * C * (1 - D_tau) / (lambda1 - lambda2 * D_tau)
    return (1 / gamma) * Wt**gamma * np.exp(exponent1 + exponent2)

# Function to calculate Cov(dJ, dX)
def Cov_dJ_dX(gamma, Xt):
    lambda1, lambda2, D_tau = calculate_lambdas_and_D(gamma)
    C = -k**2 / (2 * gamma * (gamma - 1) * sigma**2)
    beta_t = beta_tau(C, lambda1, lambda2, D_tau)
    
    Jt = J_t(gamma, Xt, lambda1, lambda2, D_tau, C)
    
    first_term = (gamma / (gamma - 1)) * (k/sigma**2 - 2 * beta_t)
    second_term = (2 * C * (1 - D_tau)) / (lambda1 - lambda2 * D_tau)
    
    return Jt * Xt * sigma**2 * (first_term + second_term)

# Range of Xt values
Xt_values = np.linspace(-1, 1, 400)

# Gamma values to consider
gamma_values = [-16, -8, -2, 0.2, 0.8]

# Plotting
plt.figure(figsize=(12, 8))

for gamma in gamma_values:
    Cov_values = [Cov_dJ_dX(gamma, Xt) for Xt in Xt_values]
    plt.plot(Xt_values, Cov_values, label=f'$\gamma$={gamma}')

plt.xlabel('$X_t$')
plt.ylabel('Cov$(dJ, dX)$')
plt.title('Cov$(dJ, dX)$ vs $X_t$ for different gamma values')
plt.legend()
plt.show()
