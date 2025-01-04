import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import BytesIO

# Load Data
@st.cache
def load_data():
    data = pd.read_csv('data\owid-co2-data.csv')
    data['year'] = pd.to_datetime(data['year'], format='%Y')
    return data

# Helper Functions
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        dataX.append(dataset[i:(i + time_step), 0])
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# App Layout
st.set_page_config(page_title="CO₂ Emissions Prediction", layout="wide")
st.title("🌍 Advanced CO₂ Emissions Analysis & Prediction App")

data = load_data()
global_data = data[data['country'] == 'World'][['year', 'co2']].dropna()
global_data.set_index('year', inplace=True)

# Sidebar Options
st.sidebar.title("Navigation")
view = st.sidebar.radio(
    "Choose a Section:",
    ("Global Trends", "Top Emitters", "Correlation Analysis", "Future Predictions"),
)

# Global Trends Section
if view == "Global Trends":
    st.subheader("📈 Global CO₂ Emissions Trends")
    st.write("Explore how global CO₂ emissions have evolved over time.")

    # Interactive year range slider
    start_year, end_year = st.slider(
        "Select Year Range:",
        int(global_data.index.year.min()),
        int(global_data.index.year.max()),
        (2000, 2020),
    )
    filtered_data = global_data.loc[f"{start_year}":f"{end_year}"]

    # Plot Global Trends
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(filtered_data.index, filtered_data['co2'], label="CO₂ Emissions", color="blue")
    ax.set_title("Global CO₂ Emissions Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions (Million Tonnes)")
    ax.legend()
    st.pyplot(fig)

# Top Emitters Section
elif view == "Top Emitters":
    st.subheader("🌟 Top CO₂ Emitting Countries")
    st.write("Identify the countries contributing the most to global CO₂ emissions.")

    recent_year = data['year'].max().year
    recent_data = data[data['year'].dt.year == recent_year].dropna(subset=['co2'])
    top_emitters = recent_data[['country', 'co2']].sort_values(by='co2', ascending=False).head(10)

    # Bar Plot of Top Emitters
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="co2", y="country", data=top_emitters, palette="viridis", ax=ax)
    ax.set_title(f"Top 10 CO₂ Emitting Countries in {recent_year}")
    ax.set_xlabel("CO₂ Emissions (Million Tonnes)")
    ax.set_ylabel("Country")
    st.pyplot(fig)

# Correlation Analysis Section
elif view == "Correlation Analysis":
    st.subheader("🔍 Correlation Analysis")
    st.write("Analyze relationships between CO₂ emissions and other features like GDP and population.")

    # Select features for correlation
    corr_features = st.multiselect(
        "Select Features to Analyze:",
        ['co2', 'population', 'gdp', 'energy_per_capita'],
        default=['co2', 'population', 'gdp']
    )

    # Correlation Heatmap
    corr_data = data[corr_features].dropna()
    corr_matrix = corr_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

# Future Predictions Section
elif view == "Future Predictions":
    st.subheader("🚀 Predict Future CO₂ Emissions")
    st.write("Use an LSTM model to predict future CO₂ emissions.")

    # User Input for Prediction
    future_years = st.number_input("Enter number of years to predict:", min_value=1, max_value=50, value=10)
    time_step = st.slider("Select Time Step for LSTM:", min_value=5, max_value=30, value=10)

    # Prepare Data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(global_data)
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]

    # Train LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=10, verbose=0)

    # Predict Future
    last_data = scaled_data[-time_step:].reshape(1, time_step, 1)
    future_predictions = []

    for _ in range(future_years):
        future_pred = model.predict(last_data)
        future_predictions.append(future_pred[0, 0])
        future_pred_reshaped = future_pred.reshape(1, 1, 1)
        last_data = np.append(last_data[:, 1:, :], future_pred_reshaped, axis=1)

    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    future_dates = pd.date_range(start=global_data.index[-1] + pd.Timedelta(days=365), periods=future_years, freq='Y')

    # Display Results
    future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Future CO₂ Emissions'])
    st.write("### Future Predictions")
    st.dataframe(future_df)

    # Download as CSV
    csv_buffer = BytesIO()
    future_df.to_csv(csv_buffer, index=True)
    st.download_button(
        label="📥 Download Predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name="future_co2_predictions.csv",
        mime="text/csv",
    )

    # Plot Predictions
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(global_data.index, scaler.inverse_transform(scaled_data), label="Actual CO₂ Emissions")
    ax.plot(future_df.index, future_df['Future CO₂ Emissions'], label="Future CO₂ Emissions", color="orange", linestyle="--")
    ax.set_title("Predicted Future CO₂ Emissions")
    ax.set_xlabel("Year")
    ax.set_ylabel("CO₂ Emissions (Million Tonnes)")
    ax.legend()
    st.pyplot(fig)
