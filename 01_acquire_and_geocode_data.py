"""
01_acquire_and_geocode_data.py

Acquires restaurant data and geocodes addresses using Nominatim.
"""

import pandas as pd
from geopy.geocoders import Nominatim
import time
import numpy as np

def get_raw_restaurant_data():
    """Returns a DataFrame with sample restaurant data."""
    data = {
        'restaurant_name': [
            'Meghana Foods', 'Barbeque Nation', 'Dindigul Thalappakatti',
            'KFC', 'Burger King', "Domino's Pizza", 'Empire Restaurant',
            "Leon's Burgers & Wings"
        ],
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

def geocode_addresses(df, geolocator, delay=1.5):
    """Geocodes addresses in the DataFrame and adds latitude/longitude columns."""
    df['latitude'] = None
    df['longitude'] = None
    print("\n--- Geocoding restaurant addresses... ---")
    for index, row in df.iterrows():
        address = row['address']
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                df.at[index, 'latitude'] = location.latitude
                df.at[index, 'longitude'] = location.longitude
                print(f"Geocoded: {row['restaurant_name']}")
            else:
                print(f"Skipping: Could not geocode address for {row['restaurant_name']}")
        except Exception as e:
            print(f"Error geocoding {row['restaurant_name']}: {e}")
        time.sleep(delay)  # Respect API rate limits
    return df

def main():
    print("--- Starting Data Acquisition and Geocoding ---")
    df_restaurants = get_raw_restaurant_data()
    geolocator = Nominatim(user_agent="food_delivery_app")
    df_restaurants = geocode_addresses(df_restaurants, geolocator)
    df_restaurants.dropna(inplace=True)
    # Add a combined name for a clearer UI display
    df_restaurants['display_name'] = df_restaurants['restaurant_name'] + ', ' + df_restaurants['area']
    df_restaurants.to_csv('geocoded_restaurants.csv', index=False)
    print("\nGeocoding complete. Data saved to 'geocoded_restaurants.csv'.")
    print(df_restaurants.head())

if __name__ == "__main__":
    main()