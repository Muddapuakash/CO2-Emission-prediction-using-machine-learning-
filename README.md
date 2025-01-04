# CO₂ Emissions Prediction and Analysis

This project is a web application built using **Streamlit** that allows users to explore and predict CO₂ emissions data on a global scale. The app provides multiple features for analyzing historical CO₂ emissions trends, identifying the top CO₂ emitting countries, exploring correlations with other factors like population and GDP, and using machine learning to predict future CO₂ emissions.

## Key Features:

1. **Global Trends**: 
   - Visualizes the global CO₂ emissions over time, allowing users to select a custom year range for analysis.
   - Interactive line chart to display CO₂ emissions data.

2. **Top Emitters**: 
   - Shows the top 10 CO₂ emitting countries in the most recent year.
   - Interactive bar chart to highlight which countries contribute the most to global CO₂ emissions.

3. **Correlation Analysis**: 
   - Enables users to analyze the relationship between CO₂ emissions and other variables such as population, GDP, and energy consumption per capita.
   - Interactive heatmap displaying the correlation between selected features.

4. **Future Predictions**: 
   - Uses a **Long Short-Term Memory (LSTM)** model to predict future CO₂ emissions.
   - Users can specify the number of years for prediction and adjust the time step for the LSTM model.
   - Visualizes the actual CO₂ emissions along with future predictions and provides an option to download the prediction data as a CSV file.

## Technologies Used:
- **Streamlit**: For building the interactive web application.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For scaling and preparing data for machine learning.
- **TensorFlow (Keras)**: For building and training the LSTM model to predict future CO₂ emissions.
- **MinMaxScaler**: For scaling CO₂ emissions data before feeding it into the model.

## How to Run:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/co2-emissions-prediction.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3.Run the Streamlit app:
 ```bash
   streamlit run app.py
