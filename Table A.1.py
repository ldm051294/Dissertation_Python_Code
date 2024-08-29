# -*- coding: utf-8 -*-
"""
@author: Luigi Di Mauro
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
k = 2
sigma = 1

# Values of gamma to consider
gamma_values = [-16, -8, -2, 0.2, 0.8]

# Time values from 0 to 1
tau = np.linspace(0, 1, 500)

def calculate_beta(gamma, tau):
    A = 2 * sigma**2 * (1 - 1 / (gamma * (gamma - 1)))
    B = 2 * k * (-1 + 1 / (gamma * (gamma - 1)))
    C = -k**2 / (2 * gamma * (gamma - 1) * sigma**2)

    discriminant = np.sqrt(B**2 - 4 * A * C)
    lambda_1 = (-B - discriminant) / 2 
    lambda_2 = (-B + discriminant) / 2 

    D_tau = np.exp(discriminant * tau)
    
    beta_tau = C * (1 - D_tau) / (lambda_1 - lambda_2 * D_tau)
    
    return beta_tau

# Plotting the curves for various levels of gamma
plt.figure(figsize=(10, 6))

for gamma in gamma_values:
    beta_tau = calculate_beta(gamma, tau)
    expression = np.sqrt((-(gamma - 1)) / (k / sigma**2 - 2 * beta_tau))
    
    plt.plot(tau, expression, label=f'$\gamma$ = {gamma}')

# Plot customization
plt.xlabel(r'$\tau$')
plt.ylabel(r'$\sqrt{\frac{-(\gamma-1)}{\frac{k}{\sigma^2} - 2\beta(\tau)}}$')
plt.title(r'Plot of $X_t$ threshold vs $\tau$ for various $\gamma$ values')
plt.legend()
plt.show()
