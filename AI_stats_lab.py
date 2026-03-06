"""
AI Stats Lab
Random Variables and Distributions
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

# =========================================================
# QUESTION 1 — CDF Probabilities
# =========================================================

def cdf_probabilities():
    """
    Compute analytical probabilities and Monte Carlo verification for
    X ~ Exp(1) with CDF F_X(x) = (1 - exp(-x)) u(x)
    """
    # Analytical probabilities
    analytic_gt5 = math.exp(-5)              # P(X > 5)
    analytic_lt5 = 1 - math.exp(-5)         # P(X < 5)
    analytic_interval = math.exp(-3) - math.exp(-7)  # P(3 < X < 7)

    # Monte Carlo simulation
    N = 100_000
    X_samples = np.random.exponential(scale=1.0, size=N)
    simulated_gt5 = np.mean(X_samples > 5)

    return analytic_gt5, analytic_lt5, analytic_interval, simulated_gt5

# =========================================================
# QUESTION 2 — PDF Validation and Plot
# =========================================================

def pdf_validation_plot():
    """
    Candidate PDF: f(x) = 2x * exp(-x^2), x >= 0
    """
    # Non-negativity check
    x_vals = np.linspace(0, 3, 300)
    f_vals = 2 * x_vals * np.exp(-x_vals**2)
    non_negative = np.all(f_vals >= 0)

    # Compute integral from 0 to ∞
    integral_value, _ = quad(lambda x: 2*x*np.exp(-x**2), 0, np.inf)

    # Determine if valid PDF
    is_valid_pdf = non_negative and math.isclose(integral_value, 1.0, rel_tol=1e-6)

    # Plot PDF on [0,3]
    plt.figure(figsize=(6,4))
    plt.plot(x_vals, f_vals, label='f(x) = 2x e^{-x²}')
    plt.title("Candidate PDF")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

    return integral_value, is_valid_pdf

# =========================================================
# QUESTION 3 — Exponential Distribution
# =========================================================

def exponential_probabilities():
    """
    X ~ Exp(λ=1)
    """
    # Analytical probabilities
    analytic_gt5 = math.exp(-5)
    analytic_interval = math.exp(-1) - math.exp(-3)  # P(1 < X < 3)

    # Monte Carlo simulation
    N = 100_000
    X_samples = np.random.exponential(scale=1.0, size=N)
    simulated_gt5 = np.mean(X_samples > 5)
    simulated_interval = np.mean((X_samples > 1) & (X_samples < 3))

    return analytic_gt5, analytic_interval, simulated_gt5, simulated_interval

# =========================================================
# QUESTION 4 — Gaussian Distribution
# =========================================================

def gaussian_probabilities():
    """
    X ~ N(10, 2^2)
    """
    mu = 10
    sigma = 2

    # Standardize variable
    # Z = (X - mu)/sigma
    z_12 = (12 - mu)/sigma
    z_8 = (8 - mu)/sigma

    # Analytical probabilities using CDF
    analytic_le12 = norm.cdf(z_12)            # P(X <= 12)
    analytic_interval = norm.cdf(z_12) - norm.cdf(z_8)  # P(8 < X < 12)

    # Monte Carlo simulation
    N = 100_000
    X_samples = np.random.normal(loc=mu, scale=sigma, size=N)
    simulated_le12 = np.mean(X_samples <= 12)
    simulated_interval = np.mean((X_samples > 8) & (X_samples < 12))

    return analytic_le12, analytic_interval, simulated_le12, simulated_interval