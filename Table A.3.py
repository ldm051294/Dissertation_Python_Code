# -*- coding: utf-8 -*-
"""
@author: Luigi Di Mauro
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
k = 2.0
sigma = 1.0
Wt = 1.0

# Define the range for tau
tau_range = np.linspace(0, 1, 400)

# Define the different gamma values
gamma_values = [-16, -8, -2, 0.2, 0.8]

# Function to calculate A, B, C
def compute_ABC(gamma):
    A = 2 * sigma**2 * (1 - 1 / (gamma * (gamma - 1)))
    B = 2 * k * (-1 + 1 / (gamma * (gamma - 1)))
    C = -k**2 / (2 * gamma * (gamma - 1) * sigma**2)
    return A, B, C

# Function to calculate lambda_1, lambda_2
def compute_lambdas(A, B, C):
    discriminant = np.sqrt(B**2 - 4 * A * C)
    lambda_1 = (-B - discriminant) / 2
    lambda_2 = (-B + discriminant) / 2
    return lambda_1, lambda_2

# Function to calculate D(tau)
def compute_D(tau, B, C, A):
    discriminant = np.sqrt(B**2 - 4 * A * C)
    return np.exp(discriminant * tau)

# Function to calculate beta(tau)
def compute_beta(tau, A, B, C):
    lambda_1, lambda_2 = compute_lambdas(A, B, C)
    D_tau = compute_D(tau, B, C, A)
    beta_tau = C * (1 - D_tau) / (lambda_1 - lambda_2 * D_tau)
    return beta_tau

# Function to calculate Cov(dW, dX) as a function of tau
def cov_dW_dX(tau, gamma, X_t):
    A, B, C = compute_ABC(gamma)
    beta_tau = compute_beta(tau, A, B, C)
    cov = (sigma**2 / (gamma - 1)) * Wt * X_t * (k / sigma**2 - 2 * beta_tau)
    return cov

# First figure for X_t = 0.5
X_t = 0.5
plt.figure(figsize=(8, 6))
for gamma in gamma_values:
    cov_values = [cov_dW_dX(tau, gamma, X_t) for tau in tau_range]
    plt.plot(tau_range, cov_values, label=f'γ = {gamma}')
plt.title(r'Cov$(dW, dX)$ vs $\tau$ for $X_t = 0.5$')
plt.xlabel(r'$\tau$')
plt.ylabel(r'Cov$(dW, dX)$')
plt.legend()
plt.show()

# Second figure for X_t = -0.5
X_t = -0.5
plt.figure(figsize=(8, 6))
for gamma in gamma_values:
    cov_values = [cov_dW_dX(tau, gamma, X_t) for tau in tau_range]
    plt.plot(tau_range, cov_values, label=f'γ = {gamma}')
plt.title(r'Cov$(dW, dX)$ vs $\tau$ for $X_t = -0.5$')
plt.xlabel(r'$\tau$')
plt.ylabel(r'Cov$(dW, dX)$')
plt.legend()
plt.show()
