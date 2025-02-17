import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type="Call"):
    """
    Calculate the Black-Scholes option price and Greeks.

    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free rate
    - sigma: Volatility of the stock
    - option_type: "call" or "put"

    Returns:
    - A dictionary with the option price and Greeks (Delta, Gamma, Theta, Vega, Rho)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "Put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = -norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2))

    return {
        "Price": price,
        "Delta": delta,
        "Gamma": gamma,
        "Theta": theta,
        "Vega": vega,
        "Rho": rho
    }
