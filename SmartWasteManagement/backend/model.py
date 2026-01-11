from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from datetime import datetime

def predict_time_to_full(data):
    """
    Predicts how many minutes until the bin is 100% full.
    data: list of dicts from database [{'timestamp': ..., 'fill_percentage': ...}]
    """
    if len(data) < 5:
        return "Insufficient Data"

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Convert timestamp to minutes from the start
    start_time = df['timestamp'].min()
    df['minutes'] = (df['timestamp'] - start_time).dt.total_seconds() / 60
    
    X = df[['minutes']]
    y = df['fill_percentage']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Current fill level
    current_fill = y.iloc[-1]
    
    if current_fill >= 100:
        return "Full"
        
    # Rate of change (slope)
    slope = model.coef_[0]
    
    if slope <= 0:
        return "Not filling"
        
    # Calculate time to reach 100%
    # y = mx + c  =>  100 = slope * time + intercept
    # time = (100 - intercept) / slope
    
    intercept = model.intercept_
    target_minutes = (100 - intercept) / slope
    
    minutes_left = target_minutes - df['minutes'].iloc[-1]
    
    if minutes_left < 0:
        return "Full"
        
    return f"{int(minutes_left)} mins"
