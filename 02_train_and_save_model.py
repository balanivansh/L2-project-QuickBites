"""
02_train_and_save_model.py

Trains a delivery time prediction model and saves all necessary components.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from geopy.distance import geodesic
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def load_restaurant_data(path='geocoded_restaurants.csv'):
    """Loads geocoded restaurant data from CSV."""
    try:
        df = pd.read_csv(path)
        print("Loaded geocoded restaurant data.")
        return df
    except FileNotFoundError:
        print(f"Error: '{path}' not found. Please run '01_acquire_and_geocode_data.py' first.")
        exit()

def get_simulated_customers():
    """Returns a DataFrame with simulated customer locations."""
    return pd.DataFrame({
        'customer_location': ['Koramangala', 'Indiranagar', 'Jayanagar', 'Marathahalli', 'Whitefield', 'BTM'],
        'latitude': [12.9345, 12.9784, 12.9293, 12.9555, 12.9698, 12.9150],
        'longitude': [77.6254, 77.6408, 77.5845, 77.7160, 77.7500, 77.6105]
    })

def generate_order_data(df_restaurants, df_customers, n_orders=500):
    """Generates a synthetic dataset of orders for training."""
    final_data = []
    for _ in range(n_orders):
        res = df_restaurants.sample(1).iloc[0]
        cust = df_customers.sample(1).iloc[0]
        dist_km = geodesic((res['latitude'], res['longitude']), (cust['latitude'], cust['longitude'])).km
        prep_time = np.random.randint(10, 40)
        order_hour = np.random.randint(8, 23)
        delivery_time = (dist_km * 2.5) + (prep_time * 1.5)
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
    df_restaurants = load_restaurant_data()
    df_customers = get_simulated_customers()
    df = generate_order_data(df_restaurants, df_customers)
    X, y, encoder = preprocess_features(df)
    best_model, X_train, X_test, y_train, y_test = train_and_optimize_model(X, y)
    save_components(best_model, encoder, df_restaurants, df_customers)

if __name__ == "__main__":
    main()