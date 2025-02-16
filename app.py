import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Load Data (Ensure correct datetime format)
@st.cache_data
def load_data():
    df = pd.read_csv("credit_card_transactions.csv")
    df['trans_date'] = pd.to_datetime(df['trans_date'], dayfirst=True)
    df['trans_date'] = df['trans_date'].dt.date  # Convert Timestamp to Date
    return df

data = load_data()

# Sidebar Filters
st.sidebar.header("User Input Panel")
customer_id = st.sidebar.selectbox("Select Customer ID", data['cus_id'].unique())
category = st.sidebar.selectbox("Select Spending Category", data['category'].unique())

# Date Slider
date_range = st.sidebar.slider(
    "Select Date Range", 
    min_value=min(data['trans_date']), 
    max_value=max(data['trans_date']), 
    value=(min(data['trans_date']), max(data['trans_date']))
)

# Filter Data
filtered_data = data[
    (data['cus_id'] == customer_id) & 
    (data['category'] == category) & 
    (data['trans_date'] >= date_range[0]) & 
    (data['trans_date'] <= date_range[1])
]

filtered_data

# Historical Spending Trends
st.subheader("Historical Spending Trends")
if not filtered_data.empty:
    fig = px.line(filtered_data, x='trans_date', y='tran_amt', title='Spending Trend')
    st.plotly_chart(fig)
else:
    st.write("⚠ No data available for the selected filters.")

# Prophet Forecasting
st.subheader("Future Spending Prediction")

# Check if there are enough valid rows
if len(filtered_data.dropna()) >= 2:  
    df_prophet = filtered_data[['trans_date', 'tran_amt']].rename(columns={"trans_date": "ds", "tran_amt": "y"})

    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=180)  # Predict next 6 months
    forecast = model.predict(future)

    fig_forecast = px.lin