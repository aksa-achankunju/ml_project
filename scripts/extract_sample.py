"""
Data Sampling & Preprocessing Script for RoadSafe AI System.
Extracts a clean, balanced sample from the 7.7M US Accidents dataset,
performs data hygiene, handles missing values, and creates standard features.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

RAW_DATA_PATH = r"C:\Users\pathi\OneDrive\Desktop\ML\ClassProject\US_Accidents_March23.csv"
OUTPUT_DIR = r"C:\Users\pathi\OneDrive\Desktop\ML\ClassProject\RoadSafe_AI\data"
SAMPLE_OUTPUT = os.path.join(OUTPUT_DIR, "roadsafe_clean_sample.csv")
BASELINE_OUTPUT = os.path.join(OUTPUT_DIR, "reference_baseline.csv")
FEATURE_STORE_OUTPUT = os.path.join(OUTPUT_DIR, "feature_store.json")

COLS_TO_LOAD = [
    'Severity', 'Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'Distance(mi)', 
    'City', 'County', 'State', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition', 
    'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 
    'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Sunrise_Sunset'
]

def clean_weather_condition(val):
    if not isinstance(val, str):
        return 'Clear'
    v = val.lower()
    if 'snow' in v or 'ice' in v or 'wintry' in v or 'sleet' in v:
        return 'Snow'
    elif 'thunder' in v or 'storm' in v or 'hail' in v or 'squall' in v:
        return 'Severe'
    elif 'rain' in v or 'drizzle' in v or 'shower' in v:
        return 'Rain'
    elif 'fog' in v or 'mist' in v or 'haze' in v or 'smoke' in v:
        return 'Fog'
    elif 'cloud' in v or 'overcast' in v:
        return 'Cloudy'
    elif 'clear' in v or 'fair' in v:
        return 'Clear'
    else:
        return 'Other'

def extract_and_clean():
    print(f"Reading chunks from {RAW_DATA_PATH}...")
    target_per_class = {1: 15000, 2: 25000, 3: 25000, 4: 15000}
    collected = {1: [], 2: [], 3: [], 4: []}
    counts = {1: 0, 2: 0, 3: 0, 4: 0}

    chunk_size = 250000
    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=chunk_size, usecols=COLS_TO_LOAD):
        # Drop rows where coordinates or time is missing
        chunk = chunk.dropna(subset=['Start_Lat', 'Start_Lng', 'Start_Time', 'End_Time', 'Severity'])
        for s in [1, 2, 3, 4]:
            if counts[s] < target_per_class[s]:
                sub = chunk[chunk['Severity'] == s]
                needed = target_per_class[s] - counts[s]
                take = sub.iloc[:needed]
                if len(take) > 0:
                    collected[s].append(take)
                    counts[s] += len(take)
        if all(counts[s] >= target_per_class[s] for s in [1, 2, 3, 4]):
            break

    df = pd.concat([pd.concat(collected[s]) for s in [1, 2, 3, 4]], ignore_index=True)
    print(f"Initial raw sample shape: {df.shape}")

    # Feature Engineering
    print("Performing feature engineering...")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # Calculate duration in minutes
    df['Duration_Minutes'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0
    # Clean unreasonable durations (clip between 5 and 720 minutes)
    df['Duration_Minutes'] = df['Duration_Minutes'].fillna(60.0).clip(5.0, 720.0)

    df['Hour'] = df['Start_Time'].dt.hour.fillna(12).astype(int)
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek.fillna(2).astype(int)
    df['Month'] = df['Start_Time'].dt.month.fillna(6).astype(int)
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['Rush_Hour'] = df['Hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)

    # Impute numerical features
    df['Temperature(F)'] = df['Temperature(F)'].fillna(df['Temperature(F)'].median())
    df['Humidity(%)'] = df['Humidity(%)'].fillna(df['Humidity(%)'].median())
    df['Pressure(in)'] = df['Pressure(in)'].fillna(df['Pressure(in)'].median())
    df['Visibility(mi)'] = df['Visibility(mi)'].fillna(df['Visibility(mi)'].median())
    df['Wind_Speed(mph)'] = df['Wind_Speed(mph)'].fillna(df['Wind_Speed(mph)'].median())
    df['Precipitation(in)'] = df['Precipitation(in)'].fillna(0.0)
    df['Distance(mi)'] = df['Distance(mi)'].fillna(0.0).clip(0.0, 50.0)

    # Clean weather condition
    df['Weather_Condition'] = df['Weather_Condition'].apply(clean_weather_condition)

    # Clean day/night
    df['Sunrise_Sunset'] = df['Sunrise_Sunset'].fillna('Day').apply(lambda x: 'Day' if str(x).lower().startswith('d') else 'Night')

    # Road features as boolean -> int
    road_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 
                 'Railway', 'Roundabout', 'Station', 'Stop', 'Traffic_Calming', 'Traffic_Signal']
    for c in road_cols:
        df[c] = df[c].fillna(False).astype(int)

    # State cleaning
    top_states = df['State'].value_counts().nlargest(15).index.tolist()
    df['State_Clean'] = df['State'].apply(lambda x: x if x in top_states else 'Other')

    # Shuffle the dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"Final cleaned dataset shape: {df.shape}")
    print("Class distribution:\n", df['Severity'].value_counts())

    # Save cleaned sample
    df.to_csv(SAMPLE_OUTPUT, index=False)
    print(f"Saved cleaned sample to {SAMPLE_OUTPUT}")

    # Save reference baseline (2000 records) for Drift Monitoring (CO6)
    df.iloc[:2000].to_csv(BASELINE_OUTPUT, index=False)
    print(f"Saved drift reference baseline to {BASELINE_OUTPUT}")

    # Build Feature Store dictionary (CO6)
    feature_store_data = {
        "metadata": {
            "entity": "accident_record",
            "version": "v1.0",
            "source_dataset": "US_Accidents_March23.csv",
            "total_records_processed": 80000,
            "classes": [1, 2, 3, 4]
        },
        "feature_statistics": {
            "Temperature(F)": {"mean": float(df['Temperature(F)'].mean()), "std": float(df['Temperature(F)'].std()), "median": float(df['Temperature(F)'].median())},
            "Humidity(%)": {"mean": float(df['Humidity(%)'].mean()), "std": float(df['Humidity(%)'].std()), "median": float(df['Humidity(%)'].median())},
            "Pressure(in)": {"mean": float(df['Pressure(in)'].mean()), "std": float(df['Pressure(in)'].std()), "median": float(df['Pressure(in)'].median())},
            "Visibility(mi)": {"mean": float(df['Visibility(mi)'].mean()), "std": float(df['Visibility(mi)'].std()), "median": float(df['Visibility(mi)'].median())},
            "Wind_Speed(mph)": {"mean": float(df['Wind_Speed(mph)'].mean()), "std": float(df['Wind_Speed(mph)'].std()), "median": float(df['Wind_Speed(mph)'].median())},
            "Precipitation(in)": {"mean": float(df['Precipitation(in)'].mean()), "std": float(df['Precipitation(in)'].std()), "median": float(df['Precipitation(in)'].median())},
            "Distance(mi)": {"mean": float(df['Distance(mi)'].mean()), "std": float(df['Distance(mi)'].std()), "median": float(df['Distance(mi)'].median())},
            "Duration_Minutes": {"mean": float(df['Duration_Minutes'].mean()), "std": float(df['Duration_Minutes'].std()), "median": float(df['Duration_Minutes'].median())}
        },
        "categorical_vocabularies": {
            "Weather_Condition": list(df['Weather_Condition'].unique()),
            "Sunrise_Sunset": list(df['Sunrise_Sunset'].unique()),
            "State_Clean": top_states + ['Other']
        },
        "road_features": road_cols
    }
    with open(FEATURE_STORE_OUTPUT, 'w') as f:
        json.dump(feature_store_data, f, indent=2)
    print(f"Feature store schema saved to {FEATURE_STORE_OUTPUT}")

if __name__ == '__main__':
    extract_and_clean()
