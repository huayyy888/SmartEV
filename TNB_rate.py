import pandas as pd
import numpy as np

# --- 1. TNB Residential TOU Tariff Definition (Unit: sen/kWh) ---
# Note: For EV scheduling, a Time-of-Use model is adopted as the price signal.
PRICE_OFF_PEAK = 34.43 # Off-peak price
PRICE_PEAK = 38.52     # Peak price

# Define peak hours for working days (Mon-Fri)
PEAK_START_HOUR = 14  # 14:00 (2 PM)
PEAK_END_HOUR = 22    # 22:00 (10 PM)

def generate_tnb_tou_data(start_date, num_days, time_step_minutes=15):
    """
    Generates TNB TOU price time series data based on typical tariff rules.
    
    Args:
        start_date (str): Start date and time of the dataset (e.g., '2025-01-06 00:00:00')
        num_days (int): Number of days to generate data for
        time_step_minutes (int): Time resolution in minutes (15 min is common for EV2Gym)
        
    Returns:
        pd.DataFrame: DataFrame containing the price time series
    """
    # 1. Create the time series index
    end_date = pd.to_datetime(start_date) + pd.Timedelta(days=num_days) - pd.Timedelta(minutes=time_step_minutes)
    time_index = pd.date_range(start=start_date, end=end_date, freq=f'{time_step_minutes}min')

    prices = []

    for timestamp in time_index:
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek  # 0=Monday, 6=Sunday

        # Check for weekend (Saturday=5, Sunday=6)
        is_weekend = day_of_week >= 5

        if is_weekend:
            # Weekends are off-peak price all day
            price = PRICE_OFF_PEAK
        elif PEAK_START_HOUR <= hour < PEAK_END_HOUR:
            # Working days: 14:00 to 22:00 is peak price
            price = PRICE_PEAK
        else:
            # Working days: 22:00 to 14:00 is off-peak price
            price = PRICE_OFF_PEAK
        
        prices.append(price)

    # 2. Create the DataFrame
    df = pd.DataFrame({
        'Timestamp': time_index,
        'Price_sen_per_kWh': prices,
        'DayOfWeek': [t.dayofweek for t in time_index], 
        'Hour': [t.hour for t in time_index]
    })
    
    # Set Timestamp as the index (Required by EV2Gym for time series data)
    df = df.set_index('Timestamp')
    return df

# --- Configuration and Execution ---
# Generate 30 days of data for RL training
NUM_DAYS_FOR_TRAINING = 30 

price_data_30_days = generate_tnb_tou_data(
    start_date='2025-01-06 00:00:00', # Start date (e.g., a Monday)
    num_days=NUM_DAYS_FOR_TRAINING 
)

# Save to CSV file
output_filename = 'tnb_tou_price.csv'
price_data_30_days.to_csv(output_filename)

print(f"TNB TOU price dataset successfully generated and saved to: {output_filename}")
print(f"Total data rows (15-minute intervals): {len(price_data_30_days)} rows")
print("\nFirst 20 rows of the dataset example:")
print(price_data_30_days.head(20))