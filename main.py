import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import time

display_graphs = False

@st.cache_data
def load_data(ticker, START):
    rawdata = yf.download(ticker, START, TODAY)
    rawdata.reset_index(inplace=True)
    return rawdata


def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name='stock_open', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name='stock_close', line=dict(color='#8B0000')))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)


def update_graphs():
    global display_graphs
    display_graphs = True


# sidebar elements
with st.sidebar:

    st.markdown("<h1 style='color: white; font-family: Arial, sans-serif; font-size: 60px;'>Financial Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.title("Historical Data and ML Prediction")
    st.markdown("<br>", unsafe_allow_html=True)

    start_dates = ['2000-01-01', '2010-01-01', '2015-01-01', '2020-01-01']

    TODAY = date.today().strftime("%Y-%m-%d")

    stocks = ("AAPL", "GOOG", "MSFT", "NVDA", "^GSPC")
    selected_stock = st.selectbox("Select Stock for Prediction", stocks)

    n_years = st.slider("Years of prediction", 1, 10)
    period = n_years*365

    START = st.selectbox("Select a Start Date for the Model", start_dates)

    apply = st.button("Apply Settings", type='primary')

    if apply:
        my_bar = st.progress(0, text="Loading Data...")

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text="Loading Data...")
        time.sleep(1)

        global data
        data = load_data(selected_stock, START)

        my_bar.empty()

try:
    # forecasting
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    st.title("Visualization of Data")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Raw Data')
        st.write(data.tail())
        for i in range(5):
            st.markdown("<br>", unsafe_allow_html=True)

        st.subheader('Forecast data')
        st.write(forecast.tail())

        st.subheader("Forecast Graph")
        fig1 = plot_plotly(m, forecast, uncertainty=False)
        st.plotly_chart(fig1)

    with col2:
        plot_raw_data(data)

    st.subheader("Forecast Components - Trend / Weekly / Yearly")
    fig2 = m.plot_components(forecast)
    st.write(fig2)
except:
    pass
