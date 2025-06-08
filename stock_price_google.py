import yfinance as yf
import streamlit as st 
import pandas as pd 


st.write("""
         # Stock Price App
         
         Shown are the stock **closing** price and **volume** of ***Google!***
         """)

tickersymbol = 'GOOGL'

# get data on this ticker
tickerData = yf.Ticker(tickersymbol)

# get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')

# open	High	Low	Close	Volume	Dividends	Stock Splits
st.subheader('Opening Price vs Time Chart')
st.line_chart(tickerDf.Open)

st.subheader('Closing Price vs Time Chart')
st.line_chart(tickerDf.Close)

st.subheader('Volume Chart')
st.line_chart(tickerDf.Volume)

st.subheader('Opening Price vs Closing Price')
chart_data = pd.DataFrame(tickerDf, columns=['Open', 'Close'])
st.bar_chart(chart_data)