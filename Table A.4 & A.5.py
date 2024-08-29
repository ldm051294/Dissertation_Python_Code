# -*- coding: utf-8 -*-
"""
@author: Luigi Di Mauro
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)

# Parameters
k = 2.0      # Mean reversion rate
sigma = 1.0  # Volatility
X0 = 0.0     # Initial value for X_t
W0 = 1.0     # Initial value for W_t
gamma = -1.0 # Given gamma
T = 1.0      # Total time in 1 year
dt = 1/252   # Time step (1 trading day)
N = int(T / dt)  # Number of time steps
M = 10000     # Number of simulations


# Constants A, B, C
A = 2 * sigma**2 * (1 - 1 / (gamma * (gamma - 1)))
B = 2 * k * (-1 + 1 / (gamma * (gamma - 1)))
C = -k**2 / (2 * gamma * (gamma - 1) * sigma**2)

# Calculate lambda1, lambda2
lambda1 = (-B - np.sqrt(B**2 - 4 * A * C)) / 2
lambda2 = (-B + np.sqrt(B**2 - 4 * A * C)) / 2

# Time array
t = np.linspace(0, T, N+1)
tau = T - t

# D(tau) array
D_tau = np.exp(np.sqrt(B**2 - 4 * A * C) * tau)


# Initialize arrays to store the simulations
X = np.zeros((N+1, M))
W = np.zeros((N+1, M))
alpha_star = np.zeros((N+1, M))
X[0, :] = X0
W[0, :] = W0

# Monte Carlo simulation for X_t
for i in range(N):
    dBt = np.sqrt(dt) * np.random.randn(M)
    X[i+1, :] = X[i, :] - k * X[i, :] * dt + sigma * dBt

# Simulation for W^*_t and alpha^*_t
for i in range(N):
    beta_tau = C * (1 - D_tau[i]) / (lambda1 - lambda2 * D_tau[i])
    
    term1 = 1 / (gamma - 1) * X[i, :] * (k / sigma**2 - 2 * beta_tau) * (X[i+1, :] - X[i, :])
    W[i+1, :] = W[i, :] + (W[i, :] * term1)
    
    
    alpha_star[i, :] = 1 / (gamma - 1) * W[i, :] * X[i, :] * (k / sigma**2 - 2 * beta_tau)


# Calculate averages at time T
X_T_avg = np.mean(X[-1, :])
W_T_avg = np.mean(W[-1, :])
alpha_star_T_avg = np.mean(alpha_star[-1, :])

print(f"Average of X_T: {X_T_avg}")
print(f"Average of W_T: {W_T_avg}")
print(f"Average of alpha_star_T: {alpha_star_T_avg}")


# Function to generate a random functional form for alpha_t
def generate_random_alpha_func():
    operations = ['+', '-', '*', '/']
    
    def alpha_func(W, X, tau, k, gamma, sigma):
        # Ensure that each expression involves all the parameters except W and X, which will be handled separately
        terms = ['tau', 'k', 'gamma', 'sigma']
        random.shuffle(terms)  # Shuffle to get diverse combinations
        
        # Start with the first term and combine all others using random operations
        expression = terms[0]
        for i in range(1, len(terms)):
            operation = random.choice(operations)
            expression = f"({expression}) {operation} ({terms[i]})"
        
        # Include X linearly in the expression
        expression_with_X = f"{expression} + X"
        
        # Modify the expression to include W linearly
        base_expression = f"W + ({expression_with_X})"
                        
        
        try:
            # Safely evaluate the expression
            result = eval(base_expression, {"np": np, "W": W, "X": X, "tau": tau, "k": k, "gamma": gamma, "sigma": sigma})
            if np.isnan(result).any() or np.isinf(result).any():
                return np.zeros_like(W)
            return result
        except:
            return np.zeros_like(W)  

    return alpha_func


# Function to simulate non-optimal W_t for a given alpha_t
def simulate_W_t(alpha_func):
    W_nostar = np.zeros((N + 1, M))
    W_nostar[0, :] = W0


    # Simulate W_t with given alpha_func
    for i in range(N):
        tau_i = tau[i]
        alpha = alpha_func(W_nostar[i, :], X[i, :], tau_i, k, gamma, sigma)
        W_nostar[i+1, :] = W_nostar[i, :] + alpha * (X[i+1, :] - X[i, :])

    return W_nostar[-1, :]

# List to store the averages of W_T
W_T_avg_list = []

# Generate 100 random functional forms and calculate W_T average for each
for i in range(100):
    alpha_func = generate_random_alpha_func()  
    W_T_nostar = simulate_W_t(alpha_func)
    W_T_avg_nostar = np.mean(W_T_nostar)
    W_T_avg_list.append(W_T_avg_nostar)  
    print(f"Average of W_T for function {i + 1}: {W_T_avg_nostar}")


# Plotting all sample paths for X_t
plt.figure(figsize=(10, 6))
for i in range(M):  
    plt.plot(t, X[:, i], lw=0.5)
plt.xlabel('Time')
plt.ylabel('$X_t$')
plt.title('Sample Paths of the Ornstein-Uhlenbeck Process')
plt.show()


# Plotting a sample path for X_t and alpha^*_t
plt.figure(figsize=(10, 6))
plt.plot(t, X[:, 0], label='$X_t$', lw=0.5)
plt.plot(t, alpha_star[:, 0], label='$\\alpha^*_t$', lw=0.5)
plt.xlabel('Time')
plt.ylabel('$X_t$ and $\\alpha^*_t$')
plt.title('Sample Path of $X_t$ and $\\alpha^*_t$')
plt.legend()
plt.show()

# Plotting a sample path for W_t
plt.figure(figsize=(10, 6))
plt.plot(t, W[:, 0], label='$W_t$', lw=0.5)
plt.xlabel('Time')
plt.ylabel('$W_t$')
plt.title('Sample Path of $W_t$')
plt.legend()
plt.show()

# Plot the averages of non-optimal W_T vs optimal W_T
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), W_T_avg_list, marker='o', label='Non-optimal wealth $W_T$')
plt.axhline(y=W_T_avg, color='red', linestyle='--', label='Optimal wealth $W_T^*$')
plt.xlabel('Functional Form Index')
plt.ylabel('Averages of terminal wealth $W_T$')
plt.title('Averages of $W_T$ for 100 Random Functional Forms of $\\alpha_t$')
plt.xlim(0, 101)
plt.legend()
plt.show()
