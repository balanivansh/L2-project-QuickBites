from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib
import warnings
import requests
import time
import os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)

# Load the Google Maps API key from environment variables
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def get_traffic_forecast_data(start_coords, end_coords, departure_timestamp):
    # Calls the Google Maps Directions API to get distance and traffic-aware travel time.
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{end_lat},{end_lon}",
        "mode": "driving",
        "departure_time": departure_timestamp,
        "key": GOOGLE_MAPS_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            leg = data["routes"][0]["legs"][0]
            road_distance_km = leg["distance"]["value"] / 1000
            if 'duration_in_traffic' in leg:
                travel_time_minutes = leg["duration_in_traffic"]["value"] / 60
            else:
                travel_time_minutes = leg["duration"]["value"] / 60
            return road_distance_km, travel_time_minutes
        else:
            return None, None
    except requests.exceptions.RequestException:
        return None, None

def load_geocoded_data():
    # Loads the geocoded restaurant and customer data from the previous step.
    try:
        df_restaurants = joblib.load('restaurants_geocoded.joblib')
        df_customers = joblib.load('customers_geocoded.joblib')
        return df_restaurants, df_customers
    except FileNotFoundError:
        print("Error: Geocoded data not found. Please run '01_acquire_and_geocode_data.py' first.")
        exit()

def generate_order_data(df_restaurants, df_customers, n_orders=500):
    # Creates a synthetic dataset of orders, fetching real travel times from the API.
    final_data = []
    print(f"\n--- Generating {n_orders} orders with realistic simulation... ---")
    for i in range(n_orders):
        res = df_restaurants.sample(1).iloc[0]
        cust = df_customers.sample(1).iloc[0]
        start_coords = (res['latitude'], res['longitude'])
        end_coords = (cust['latitude'], cust['longitude'])
        
        # Simulates a future order time to get traffic forecasts.
        random_days_in_future = np.random.randint(1, 14)
        random_hour = np.random.randint(8, 23)
        random_minute = np.random.randint(0, 60)
        future_dt = datetime.now() + timedelta(days=random_days_in_future)
        future_dt = future_dt.replace(hour=random_hour, minute=random_minute)
        departure_timestamp = int(future_dt.timestamp())
        
        dist_km, travel_time = get_traffic_forecast_data(start_coords, end_coords, departure_timestamp)
        if dist_km is None:
            continue
        
        # Calculates a realistic prep time using a normal distribution.
        avg_time = res['avg_prep_time']
        std_dev = res['prep_time_std']
        prep_time = round(np.random.normal(loc=avg_time, scale=std_dev))
        prep_time = max(5, prep_time)
        
        day_of_week = future_dt.weekday()
        delivery_time = travel_time + prep_time
        
        # Adds simulated operational friction to make the model more robust.
        if res['restaurant_name'] == 'Meghana Foods' and day_of_week >= 4:
            delivery_time += np.random.uniform(5, 10) 
        if cust['customer_location'] == 'Whitefield':
            delivery_time += np.random.uniform(3, 7)
        
        final_data.append({
            'restaurant_name': res['restaurant_name'],
            'customer_location': cust['customer_location'],
            'restaurant_type': res['restaurant_type'],
            'prep_time_minutes': prep_time,
            'day_of_week': day_of_week,
            'order_hour': random_hour,
            'distance_km': dist_km,
            'travel_time_minutes': travel_time,
            'delivery_time_minutes': max(10, delivery_time + np.random.uniform(-5, 5))
        })
        time.sleep(0.1)
        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{n_orders} orders.")
    return pd.DataFrame(final_data)

def preprocess_features(df):
    # Prepares the data for the machine learning model.
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_features = ['restaurant_name', 'customer_location', 'restaurant_type', 'day_of_week']
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    numerical_features = ['prep_time_minutes', 'distance_km', 'travel_time_minutes', 'order_hour']
    X = pd.concat([df[numerical_features].reset_index(drop=True), encoded_df], axis=1)
    y = df['delivery_time_minutes']
    print("\nData preprocessed with forecasted travel time features.")
    return X, y, encoder

def train_and_optimize_model(X, y):
    # Trains and fine-tunes a RandomForestRegressor model.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"\nBest model found with Cross-Validation R² score: {grid_search.best_score_:.2f}")
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MAE: {mae:.2f} minutes")
    print(f"Test R²: {r2:.2f}")
    return best_model

def save_components(best_model, encoder):
    # Saves the trained model and the encoder for use in the Streamlit app.
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(encoder, 'one_hot_encoder.joblib')
    print("\nModel and encoder saved successfully.")

def main():
    df_restaurants, df_customers = load_geocoded_data()
    df_orders = generate_order_data(df_restaurants, df_customers, n_orders=500)
    X, y, encoder = preprocess_features(df_orders)
    best_model = train_and_optimize_model(X, y)
    save_components(best_model, encoder)

if __name__ == "__main__":
    main()