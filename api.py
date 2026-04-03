from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import requests
import os
import polyline
from datetime import datetime, timedelta
from dotenv import load_dotenv
import traceback

load_dotenv(override=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

try:
    print("--- Initializing ML Engine ---")
    files = ['best_model.joblib', 'one_hot_encoder.joblib', 'restaurants_geocoded.joblib', 'customers_geocoded.joblib']
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"File found: {f} ({size} bytes)")
            if size < 500:
                print(f"WARNING: {f} is suspiciously small. It might be a Git LFS pointer.")
        else:
            print(f"CRITICAL ERROR: {f} not found in {os.getcwd()}")

    model = joblib.load('best_model.joblib')
    encoder = joblib.load('one_hot_encoder.joblib')
    df_restaurants = joblib.load('restaurants_geocoded.joblib')
    df_customers = joblib.load('customers_geocoded.joblib')
    print("Success: All models and data loaded.")
except Exception as e:
    print(f"Error loading models/data: {e}")
    print(traceback.format_exc())
    model, encoder, df_restaurants, df_customers = None, None, None, None

def get_traffic_forecast_data(start_coords, end_coords, order_hour):
    now = datetime.now()
    departure_dt = now.replace(hour=order_hour, minute=0, second=0, microsecond=0)
    if departure_dt < now:
        departure_dt += timedelta(days=1)
    departure_timestamp = int(departure_dt.timestamp())

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
        encoded_polyline = data["routes"][0]["overview_polyline"]["points"]
        return road_distance_km, travel_time_minutes, encoded_polyline
    else:
        error_msg = data.get("error_message", "No error message provided.")
        raise Exception(f"Directions API Error: {error_msg}")

def calculate_delivery_fee(distance_km, order_hour):
    base_fee, per_km_charge, peak_hour_surge = 30, 8, 25
    fee = base_fee + (distance_km * per_km_charge)
    if 18 <= order_hour <= 21:
        fee += peak_hour_surge
    return fee

@app.get("/api/options")
def get_options():
    if df_restaurants is None or df_customers is None:
        raise HTTPException(
            status_code=503, 
            detail="AI Engine is initializing or failed to load data. Please check server logs."
        )
    try:
        # Avoid numpy scalar serialization issues by converting to list of primitives
        restaurant_options = df_restaurants['display_name'].unique().tolist()
        customer_options = df_customers['customer_location'].unique().tolist()
        return {
            "restaurants": restaurant_options,
            "customers": customer_options,
            "days": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PredictRequest(BaseModel):
    restaurant_name: str
    customer_location: str
    prep_time: int
    order_hour: int
    day_name: str

@app.post("/api/predict")
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly.")
    
    try:
        # Exact matching handling in case of display_name
        res_display_name = request.restaurant_name
        res_name = res_display_name.split(', ')[0]
        
        res_loc = df_restaurants[df_restaurants['restaurant_name'] == res_name].iloc[0]
        cust_loc = df_customers[df_customers['customer_location'] == request.customer_location].iloc[0]
        
        start_coords = (float(res_loc['latitude']), float(res_loc['longitude']))
        end_coords = (float(cust_loc['latitude']), float(cust_loc['longitude']))

        dist_km, travel_time, encoded_polyline = get_traffic_forecast_data(start_coords, end_coords, request.order_hour)

        day_options = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        day_of_week = day_options.get(request.day_name, 0)

        input_df = pd.DataFrame([{'restaurant_name': res_name, 'customer_location': request.customer_location,
                                  'restaurant_type': res_loc['restaurant_type'], 'day_of_week': day_of_week}])

        encoded_input = encoder.transform(input_df)
        encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(input_df.columns))

        input_data = pd.DataFrame({
            'prep_time_minutes': [request.prep_time],
            'distance_km': [dist_km],
            'travel_time_minutes': [travel_time],
            'order_hour': [request.order_hour]
        })

        final_input = pd.concat([input_data, encoded_df], axis=1)

        train_columns = model.feature_names_in_
        final_input = final_input.reindex(columns=train_columns, fill_value=0)

        predicted_time = float(model.predict(final_input)[0])

        estimated_fee = calculate_delivery_fee(dist_km, request.order_hour)
        
        route_points = polyline.decode(encoded_polyline)

        return {
            "predicted_time": predicted_time,
            "prep_time": request.prep_time,
            "travel_time": travel_time,
            "dist_km": dist_km,
            "estimated_fee": estimated_fee,
            "start_coords": start_coords,
            "end_coords": end_coords,
            "route_points": route_points
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
