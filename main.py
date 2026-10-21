# main.py
# Project: Smart Grid AI Tools (Load Forecasting)
# Bridging Electrical Engineering and Artificial Intelligence

import numpy as np
from sklearn.linear_model import LinearRegression

def predict_power_load(historical_data):
    """
    A simple baseline AI model to forecast power consumption.
    This demonstrates how machine learning can be applied to smart grid data.
    """
    print("Initializing Smart Grid AI Model...")
    
    # Prepare data for the model
    # X represents time (hours), y represents power consumption (kW)
    X = np.array(historical_data['hours']).reshape(-1, 1)
    y = np.array(historical_data['consumption'])
    
    # Train the AI model (Linear Regression)
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict future power load for the next hour (e.g., hour 24)
    future_hour = np.array([[24]])
    prediction = model.predict(future_hour)
    
    return f"Predicted Power Load for next hour: {prediction[0]:.2f} kW"

if __name__ == "__main__":
    # Sample historical data representing hourly power consumption
    sample_data = {
        'hours': [1, 2, 3, 4, 5],
        'consumption': [120, 115, 110, 130, 150]
    }
    
    # Run the AI prediction
    result = predict_power_load(sample_data)
    print(result)
