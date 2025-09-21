from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import time
import numpy as np
import requests
import joblib
import os

# Load the Google Maps API key from environment variables
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

def get_raw_restaurant_data():
    # Defines the initial, static dataset of restaurants and their details.
    data = {
        'restaurant_name': [
            'Meghana Foods', 'Barbeque Nation', 'Dindigul Thalappakatti',
            'KFC', 'Burger King', "Domino's Pizza", 'Empire Restaurant',
            "Leon's Burgers & Wings"
        ],
        'avg_prep_time': [25, 30, 25, 12, 10, 15, 20, 15],
        'prep_time_std': [5, 6, 5, 3, 3, 4, 5, 4],
        'address': [
            '5th Block, Koramangala, Bengaluru, Karnataka',
            '13th Cross Rd, Indiranagar, Bengaluru, Karnataka',
            '6th Block, Koramangala, Bengaluru, Karnataka',
            'Outer Ring Road, Marathahalli, Bengaluru, Karnataka',
            '80 Feet Main Road, Koramangala, Bengaluru, Karnataka',
            '100 Feet Road, Indiranagar, Bengaluru, Karnataka',
            '12th Cross Rd, Indiranagar, Bengaluru, Karnataka',
            '12th Main Road, Indiranagar, Bengaluru, Karnataka'
        ],
        'restaurant_type': [
            'Indian', 'Indian', 'Indian', 'Fast Food',
            'Fast Food', 'Fast Food', 'Indian', 'Fast Food'
        ],
        'area': [
            'Koramangala', 'Indiranagar', 'Koramangala',
            'Marathahalli', 'Koramangala', 'Indiranagar', 'Indiranagar',
            'Indiranagar'
        ]
    }
    return pd.DataFrame(data)

def geocode_addresses(df, delay=1.5):
    # Converts a list of addresses into latitude and longitude coordinates using the Google Maps API.
    df['latitude'] = None
    df['longitude'] = None
    print("\n--- Geocoding restaurant addresses using Google Maps API... ---")
    for index, row in df.iterrows():
        address = row['address']
        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {"address": address, "key": GOOGLE_MAPS_API_KEY}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    location = data["results"][0]["geometry"]["location"]
                    lat, lon = location["lat"], location["lng"]
                    df.at[index, 'latitude'] = lat
                    df.at[index, 'longitude'] = lon
                    print(f"Geocoded: {row['restaurant_name']}")
                else:
                    print(f"Skipping: No results for address: {row['restaurant_name']}")
            else:
                print(f"API error for {row['restaurant_name']}: {response.status_code}")
        except Exception as e:
            print(f"Error geocoding {row['restaurant_name']}: {e}")
        time.sleep(delay)
    return df

def get_simulated_customers():
    # Defines static customer locations with pre-geocoded coordinates.
    return pd.DataFrame({
        'customer_location': ['Koramangala', 'Indiranagar', 'Jayanagar', 'Marathahalli', 'Whitefield', 'BTM'],
        'latitude': [12.9345, 12.9784, 12.9293, 12.9555, 12.9698, 12.9150],
        'longitude': [77.6254, 77.6408, 77.5845, 77.7160, 77.7500, 77.6105]
    })

def main():
    print("--- Starting Data Acquisition and Geocoding ---")
    # Step 1: Get raw restaurant data and geocode addresses.
    df_restaurants = get_raw_restaurant_data()
    df_restaurants = geocode_addresses(df_restaurants, delay=1.5)
    df_restaurants.dropna(inplace=True)
    df_restaurants['display_name'] = df_restaurants['restaurant_name'] + ', ' + df_restaurants['area']
    # Step 2: Get pre-defined customer data.
    df_customers = get_simulated_customers()
    # Step 3: Save the processed data for use in the model training script.
    joblib.dump(df_restaurants, 'restaurants_geocoded.joblib')
    joblib.dump(df_customers, 'customers_geocoded.joblib')
    print("Geocoding complete. Data saved.")
    print(df_restaurants.head())

if __name__ == "__main__":
    main()