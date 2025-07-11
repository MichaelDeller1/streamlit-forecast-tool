import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
from datetime import datetime
import matplotlib.pyplot as plt

st.title("📈 Time Series Forecasting Tool")
st.markdown("Upload your CSV file with a date column and numeric values to forecast.")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded data", df.head())

    date_column = st.selectbox("Select the date column", df.columns)

    try:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column).sort_index()
    except:
        st.error("Failed to parse the date column. Please make sure it contains recognizable dates.")

    target_columns = st.multiselect("Select numeric columns to forecast", df.select_dtypes(include=np.number).columns.tolist())

    if st.button("Run Forecast") and target_columns:
        future_dates = pd.date_range(start=df.index.max() + pd.DateOffset(months=1), periods=24, freq='MS')

        def create_features(index):
            return pd.DataFrame({
                'hour': 0,
                'dayofweek': index.dayofweek,
                'quarter': index.quarter,
                'month': index.month,
                'year': index.year,
                'dayofyear': index.dayofyear,
                'dayofmonth': index.day
            }, index=index)

        future_features = create_features(future_dates)
        result_df = df.copy()

        for col in target_columns:
            hist_df = df[[col]].copy()
            hist_df['hour'] = df.index.hour
            hist_df['dayofweek'] = df.index.dayofweek
            hist_df['quarter'] = df.index.quarter
            hist_df['month'] = df.index.month
            hist_df['year'] = df.index.year
            hist_df['dayofyear'] = df.index.dayofyear
            hist_df['dayofmonth'] = df.index.day

            X = hist_df.drop(columns=[col])
            y = hist_df[col]
            model = XGBRegressor(n_estimators=50, learning_rate=0.1, n_jobs=-1, verbosity=0)
            model.fit(X, y)

            forecast = model.predict(future_features)
            forecast_df = pd.DataFrame({col: forecast}, index=future_dates)
            result_df = pd.concat([result_df, forecast_df])

        st.write("### Forecasted Results", result_df.tail(30))
        st.line_chart(result_df[target_columns])
        st.download_button("Download Forecast as CSV", result_df.to_csv().encode('utf-8'), file_name="forecast_output.csv", mime="text/csv")

