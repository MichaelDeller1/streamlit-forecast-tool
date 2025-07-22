import pandas as pd
import numpy as np
import streamlit as st
from xgboost import XGBRegressor
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.title("📈 Time Series Forecasting Tool")
st.markdown("Upload your CSV file with a date column and numeric values to forecast.")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded data", df.head())

    date_column = st.selectbox("Select the date column", df.columns)

    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])
        df = df.set_index(date_column).sort_index()
    except Exception as e:
        st.error(f"Failed to parse the date column: {e}")

    if isinstance(df.index, pd.DatetimeIndex):
        try:
            inferred_freq = pd.infer_freq(df.index[:5])
        except Exception:
            inferred_freq = None
    else:
        inferred_freq = None

    if inferred_freq is None:
        delta = df.index.to_series().diff().mode()[0]
        if delta.days >= 28:
            inferred_freq = 'MS'
        elif delta.days >= 7:
            inferred_freq = 'W'
        else:
            inferred_freq = 'D'

    st.info(f"Detected date frequency: {inferred_freq}")

    target_columns = st.multiselect("Select numeric columns to forecast", df.select_dtypes(include=np.number).columns.tolist())
    forecast_periods = st.slider("Forecast horizon (in units of detected frequency)", 30, 730, 365)

    num_variants = st.slider("Number of alternative forecasts", 0, 3, 1)
    variant_adjustments = []
    for i in range(num_variants):
        adj = st.slider(f"Adjustment for Variant {i+1} (%)", -50, 50, i * 10)
        variant_adjustments.append(adj)

    if st.button("Run Forecast") and target_columns:

        # Determine the date offset unit
        if inferred_freq == 'D':
            future_dates = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        elif inferred_freq == 'W':
            future_dates = pd.date_range(start=df.index.max() + pd.Timedelta(weeks=1), periods=forecast_periods, freq='W')
        else:
            future_dates = pd.date_range(start=df.index.max() + pd.DateOffset(months=1), periods=forecast_periods, freq='MS')

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
        all_forecasts = {}

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
            forecasts = {f"{col} (Base)": forecast}
            for i, adj in enumerate(variant_adjustments):
                forecasts[f"{col} (Variant {i+1})"] = forecast * (1 + adj / 100.0)

            for name, values in forecasts.items():
                forecast_df = pd.DataFrame({name: values}, index=future_dates)
                all_forecasts[name] = forecast_df[name]

        forecast_output = pd.concat([df[target_columns], pd.concat(all_forecasts.values(), axis=1)], axis=0)

        st.write("### Forecasted Results", forecast_output.tail(forecast_periods))

        for col in target_columns:
            fig, ax = plt.subplots()
            ax.plot(df.index, df[col], label='Historical')
            base_key = f"{col} (Base)"
            ax.plot(all_forecasts[base_key].index, all_forecasts[base_key], linestyle='dotted', label='Forecast (Base)')
            for i in range(num_variants):
                variant_key = f"{col} (Variant {i+1})"
                ax.plot(all_forecasts[variant_key].index, all_forecasts[variant_key], linestyle='dotted', label=variant_key)

            ax.set_title(f"Forecast for {col}")
            ax.legend()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            fig.autofmt_xdate()
            st.pyplot(fig)

        st.download_button("Download Forecast as CSV", forecast_output.to_csv().encode('utf-8'), file_name="forecast_output.csv", mime="text/csv")
