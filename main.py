# app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from openai import OpenAI

# Set page configuration
st.set_page_config(page_title="AI-Powered Candlestick Pattern Recognizer", layout="wide")

# Initialize OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            st.error("No data found for the given ticker and date range.")
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

# Base class for candlestick patterns
class CandlestickPattern:
    def __init__(self, name):
        self.name = name

    def detect(self, data):
        pass  # To be implemented in subclasses

# Hammer Pattern class
class HammerPattern(CandlestickPattern):
    def __init__(self):
        super().__init__("Hammer")

    def detect(self, data):
        pattern_indices = []
        for i in range(1, len(data)):
            open_price = data['Open'][i]
            close_price = data['Close'][i]
            high_price = data['High'][i]
            low_price = data['Low'][i]

            body = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)

            if lower_shadow > 2 * body and upper_shadow < body:
                pattern_indices.append(i)
        return pattern_indices

# Function to generate explanations using OpenAI GPT-3.5
def generate_explanation(pattern_name, ticker):
    prompt = f"Explain the significance of a {pattern_name} pattern in the stock {ticker} and predict potential market movements."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a stock market expert specialized in technical analysis and candlestick patterns."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        st.error(f"Error generating explanation: {e}")
        return None

# Main Streamlit app
def main():
    st.title("AI-Powered Candlestick Pattern Recognizer")

    # User inputs
    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.text_input("Enter Stock Ticker", value="AAPL")
    with col2:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
    with col3:
        end_date = st.date_input("End Date", value=datetime.now())

    if st.button("Analyze", type="primary"):
        with st.spinner("Fetching and analyzing data..."):
            data = fetch_stock_data(ticker, start_date, end_date)
            if data is not None:
                st.success(f"Data fetched for {ticker}")

                # Detect Hammer Pattern
                hammer = HammerPattern()
                hammer_indices = hammer.detect(data)

                st.subheader("Detected Hammer Patterns")
                if hammer_indices:
                    st.write(f"Found {len(hammer_indices)} Hammer patterns.")

                    # Plotting
                    fig, ax = plt.subplots(figsize=(12, 6))
                    data['Close'].plot(ax=ax, label='Close Price')
                    ax.scatter(data.index[hammer_indices], data['Close'].iloc[hammer_indices], 
                             color='red', label='Hammer Pattern', marker='^', s=100)
                    ax.set_title(f"{ticker} Stock Price with Hammer Patterns")
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)

                    # Generate AI explanation
                    with st.spinner("Generating AI analysis..."):
                        explanation = generate_explanation(hammer.name, ticker)
                        if explanation:
                            st.subheader("AI-Generated Analysis")
                            st.info(explanation)

                    # Save results and provide download option
                    result_df = data.copy()
                    result_df['Hammer_Pattern'] = [1 if i in hammer_indices else 0 for i in range(len(data))]
                    csv = result_df.to_csv().encode('utf-8')

                    st.download_button(
                        label="Download Analysis Results",
                        data=csv,
                        file_name=f'{ticker}_analysis.csv',
                        mime='text/csv',
                    )
                else:
                    st.warning("No Hammer patterns detected in the selected date range.")
            else:
                st.error("No data available for the selected date range.")

if __name__ == "__main__":
    main()