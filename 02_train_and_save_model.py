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

warnings.filterwarnings("ignore", category=UserWarning)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def get_real_world_distance_and_time(start_coords, end_coords):
    """
    Calls the Google Maps Directions API to get travel distance and time.
    """
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{end_lat},{end_lon}",
        "mode": "driving", 
        "key": GOOGLE_MAPS_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            leg = data["routes"][0]["legs"][0]
            travel_time_minutes = leg["duration"]["value"] / 60
            road_distance_km = leg["distance"]["value"] / 1000
            polyline = data["routes"][0]["overview_polyline"]["points"]
            return road_distance_km, travel_time_minutes, polyline
        else:
            print("Directions API could not find a valid route.")
            return None, None, None
    except requests.exceptions.RequestException as e:
        print(f"Directions API request failed: {e}")
        return None, None, None

def load_geocoded_data():
    """Loads geocoded restaurant and customer data from joblib files."""
    try:
        df_restaurants = joblib.load('restaurants_geocoded.joblib')
        df_customers = joblib.load('customers_geocoded.joblib')
        print("Loaded geocoded restaurant and customer data.")
        return df_restaurants, df_customers
    except FileNotFoundError:
        print(f"Error: Geocoded data not found. Please run '01_acquire_and_geocode_data.py' first.")
        exit()

def generate_order_data(df_restaurants, df_customers, n_orders=150):
    """Generates a synthetic dataset of orders for training using a routing API."""
    final_data = []
    print("\n--- Generating unified dataset with realistic travel times... ---")
    
    # Generate data for training, making an API call for each order
    for i in range(n_orders):
        res = df_restaurants.sample(1).iloc[0]
        cust = df_customers.sample(1).iloc[0]

        start_coords = (res['latitude'], res['longitude'])
        end_coords = (cust['latitude'], cust['longitude'])

        # Use the real API call to get accurate distance and travel time
        dist_km, travel_time, _ = get_real_world_distance_and_time(start_coords, end_coords)

        # If API call fails, skip this record to avoid errors
        if dist_km is None:
            continue

        prep_time = np.random.randint(10, 40)
        order_hour = np.random.randint(8, 23)
        
        delivery_time = travel_time + prep_time
        
        if 18 <= order_hour <= 21:
            delivery_time += np.random.uniform(5, 15)
            
        final_data.append({
            'restaurant_name': res['restaurant_name'],
            'customer_location': cust['customer_location'],
            'restaurant_type': res['restaurant_type'],
            'prep_time_minutes': prep_time,
            'order_hour': order_hour,
            'distance_km': dist_km,
            'delivery_time_minutes': max(10, delivery_time + np.random.uniform(-5, 5))
        })
        
        # Add a delay to avoid hitting API rate limits
        time.sleep(0.1) 
        if (i + 1) % 50 == 0:
            print(f"Generated {i+1}/{n_orders} orders.")
            
    df = pd.DataFrame(final_data)
    print("\nGenerated unified dataset.")
    print(df.head())
    return df

def preprocess_features(df):
    """Feature engineering and preprocessing."""
    df['is_peak_hour'] = df['order_hour'].apply(lambda h: 1 if 18 <= h <= 21 else 0)
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_features = ['restaurant_name', 'customer_location', 'restaurant_type']
    encoded_features = encoder.fit_transform(df[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    numerical_features = ['prep_time_minutes', 'is_peak_hour', 'distance_km']
    X = pd.concat([df[numerical_features], encoded_df], axis=1)
    y = df['delivery_time_minutes']
    print("\nData preprocessed. Features and target created.")
    return X, y, encoder

def train_and_optimize_model(X, y):
    """Trains and tunes a RandomForestRegressor."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"\nBest model found with R² score: {grid_search.best_score_:.2f}")
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MAE: {mae:.2f} minutes")
    print(f"Test R²: {r2:.2f}")
    return best_model, X_train, X_test, y_train, y_test

def save_components(best_model, encoder, df_restaurants, df_customers):
    """Saves model, encoder, and location data."""
    joblib.dump(best_model, 'best_model.joblib')
    joblib.dump(encoder, 'one_hot_encoder.joblib')
    joblib.dump(df_restaurants, 'restaurants_geocoded.joblib')
    joblib.dump(df_customers, 'customers_geocoded.joblib')
    print("\nModel, encoder, and location data saved successfully.")

def main():
    print("--- Starting Model Training and Saving ---")
    df_restaurants, df_customers = load_geocoded_data()
    df = generate_order_data(df_restaurants, df_customers)
    X, y, encoder = preprocess_features(df)
    best_model, X_train, X_test, y_train, y_test = train_and_optimize_model(X, y)
    save_components(best_model, encoder, df_restaurants, df_customers)

if __name__ == "__main__":
    main()