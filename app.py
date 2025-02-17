import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import yfinance as yf
from black_scholes import black_scholes

st.set_page_config(layout="wide")

# App Title
st.title("Black-Scholes Option Pricer with Greeks")

# Sidebar for Ticker input
st.sidebar.header("Ticker Name")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
period = st.sidebar.selectbox('Input Volatility Time Period', ('6mo', '1y', '2y', '5y', '10y', 'ytd'))
stock_data = yf.Ticker(ticker)

# Fetch Current Price
try:
    current_price = stock_data.history(period="1d")['Close'].iloc[-1]
    st.sidebar.write(f"Current Price for {ticker}: ${current_price:.2f}")
except:
    st.sidebar.write(f"Could not fetch data for ticker: {ticker}")
    current_price = 100.0  # Default fallback value

# Fetch Volatility
try: 
    hist = stock_data.history(period=period)
    daily_returns = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
    st.sidebar.write(f"Historical volatility over {period} for {ticker}: {volatility:.2f}")
except:
    st.sidebar.write(f"Could not calculate volatility for ticker: {ticker}")
    volatility = 0.2  # Default fallback value

# Fetch Risk-Free Rate
treasury_ticker = "SHY"
treasury_data = yf.Ticker(treasury_ticker)
try:
    risk_free_rate = treasury_data.info.get('dividendYield', None)
    if risk_free_rate is not None:
        st.sidebar.write(f"Approximate Risk-Free Rate: {risk_free_rate * 100:.2f}%")
    else:
        st.sidebar.write("Risk-free rate data unavailable.")
except:
    st.sidebar.write("Could not fetch risk-free rate.")
    risk_free_rate = 0.01  # Default fallback value

# Sidebar for user inputs
st.sidebar.markdown("---")
st.sidebar.header("Option Parameters")
S = st.sidebar.number_input("Stock Price (S)", min_value=0.0, value=current_price)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=current_price)
T = st.sidebar.number_input("Time to Expiration (T, in years)", min_value=0.01, value=1.0)
r = st.sidebar.number_input("Risk-Free Rate (r)", min_value=0.0, value=0.05)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, value=volatility)

# Calculate results for both call and put options
call_results = black_scholes(S, K, T, r, sigma, option_type="Call")
put_results = black_scholes(S, K, T, r, sigma, option_type="Put")

st.sidebar.markdown("---")  # Horizontal line for separation
st.sidebar.subheader("Purchase Prices")
call_purchase_price = st.sidebar.number_input("Call Option Purchase Price", min_value=0.0, value=np.round(call_results['Price']))
put_purchase_price = st.sidebar.number_input("Put Option Purchase Price", min_value=0.0, value=np.round(put_results['Price']))

st.sidebar.markdown("---")  # Horizontal line for separation
st.sidebar.subheader("Heatmap Parameters")
sigma_range = st.sidebar.slider("Volatility Range (σ)", 0.0, 1.0, (volatility-0.05, volatility+0.05))
S_range = st.sidebar.slider("Stock Price Range (S)", 0.0, current_price+100.0, (current_price-10.0, current_price+10.0))



# Display Call and Put Option Prices Side by Side
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div style="
            background-color: #d4edda;
            border-left: 5px solid #28a745;
            padding: 10px;
            border-radius: 5px;">
            <p style="color: #155724; font-size: 16px; font-weight: bold;">
                Call Option Price: ${call_results['Price']:.2f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div style="
            background-color: #f8d7da;
            border-left: 5px solid #dc3545;
            padding: 10px;
            border-radius: 5px;">
            <p style="color: #721c24; font-size: 16px; font-weight: bold;">
                Put Option Price: ${put_results['Price']:.2f}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display Greeks Side by Side
st.subheader("Option Greeks")
greek_col1, greek_col2 = st.columns(2)

with greek_col1:
    st.write("### Call Option Greeks")
    st.write(
        f"""
        **Delta**: {call_results['Delta']:.4f}  
        **Gamma**: {call_results['Gamma']:.4f}  
        **Theta**: {call_results['Theta']:.4f}  
        **Vega**: {call_results['Vega']:.4f}  
        **Rho**: {call_results['Rho']:.4f}
        """
    )

with greek_col2:
    st.write("### Put Option Greeks")
    st.write(
        f"""
        **Delta**: {put_results['Delta']:.4f}  
        **Gamma**: {put_results['Gamma']:.4f}  
        **Theta**: {put_results['Theta']:.4f}  
        **Vega**: {put_results['Vega']:.4f}  
        **Rho**: {put_results['Rho']:.4f}
        """
    )

# Generate stock price and volatility ranges
stock_prices = np.linspace(S_range[0], S_range[1], 50)
volatilities = np.linspace(sigma_range[0], sigma_range[1], 50)

# Initialize matrices for call and put profits/losses
call_profitability = np.zeros((len(stock_prices), len(volatilities)))
put_profitability = np.zeros((len(stock_prices), len(volatilities)))

# Compute profitability for each combination of stock price and volatility
for i, S in enumerate(stock_prices):
    for j, sigma in enumerate(volatilities):
        call_price = black_scholes(S, K, T, r, sigma, "Call")["Price"]
        put_price = black_scholes(S, K, T, r, sigma, "Put")["Price"]
        call_profitability[i, j] = call_price - call_purchase_price
        put_profitability[i, j] = put_price - put_purchase_price

# Make columns for heatmaps
col1, col2 = st.columns(2)

# Plot call option profitability heatmap
with col1:
    st.subheader("Call Option Profitability Heatmap")
    fig1 = go.Figure(data=go.Heatmap(
        z=call_profitability,
        x=np.round(volatilities, 2),
        y=np.round(stock_prices, 2),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Profit/Loss")
    ))
    fig1.update_layout(
        xaxis_title="Volatility (σ)",
        yaxis_title="Stock Price (S)",
        title="Call Option Profitability",
    )
    st.plotly_chart(fig1, use_container_width=True)

# Put Option Heatmap with Plotly
with col2:
    st.subheader("Put Option Profitability Heatmap")
    fig2 = go.Figure(data=go.Heatmap(
        z=put_profitability,
        x=np.round(volatilities, 2),
        y=np.round(stock_prices, 2),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Profit/Loss")
    ))
    fig2.update_layout(
        xaxis_title="Volatility (σ)",
        yaxis_title="Stock Price (S)",
        title="Put Option Profitability",
    )
    st.plotly_chart(fig2, use_container_width=True)