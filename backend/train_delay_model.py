import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import os

class TrainDelayModel:
    """
    A model that predicts train delays on the London-Norwich route
    based on historical data from 2022-2024.
    """
    
    def __init__(self):
        """Initialize the model with station mappings and route sequences."""
        # Station mapping between codes and human-readable names
        self.station_code_to_name = {
            'LST': 'London Liverpool Street', 'SRA': 'Stratford',
            'CHM': 'Chelmsford', 'COL': 'Colchester',
            'MNG': 'Manningtree', 'IPS': 'Ipswich',
            'SMK': 'Stowmarket', 'DIS': 'Diss',
            'NRW': 'Norwich', 'WTM': 'Witham',
            'SNF': 'Shenfield'
        }
        
        # Create reverse mapping from names to codes
        self.station_name_to_code = {v: k for k, v in self.station_code_to_name.items()}
        
        # Define station sequences for both directions
        self.london_to_norwich = ['LST', 'SRA', 'SNF', 'WTM', 'CHM', 'COL', 'MNG', 'IPS', 'SMK', 'DIS', 'NRW']
        self.norwich_to_london = self.london_to_norwich[::-1]
        
        # Define adjacent station pairs for both directions
        self.adjacent_pairs = {
            'LTN': [f"{self.london_to_norwich[i]}-{self.london_to_norwich[i+1]}" 
                    for i in range(len(self.london_to_norwich)-1)],
            'NTL': [f"{self.norwich_to_london[i]}-{self.norwich_to_london[i+1]}" 
                    for i in range(len(self.norwich_to_london)-1)]
        }
        
        # Storage for trained models and data
        self.data = None
        self.segment_models = {}
        self.journey_times = {}
    
    def load_data(self, file_paths):
        """Load and preprocess train data from multiple CSV files."""
        print(f"Loading data from {len(file_paths)} files...")
        
        dfs = []
        for file_path in file_paths:
            try:
                if not os.path.exists(file_path):
                    continue
                    
                df = pd.read_csv(file_path)
                
                # Determine direction based on filename
                if 'London_to_Norwich' in file_path or '_LTN' in file_path:
                    df['direction'] = 'LTN'
                    df['sequence'] = 'london_to_norwich'
                else:
                    df['direction'] = 'NTL'
                    df['sequence'] = 'norwich_to_london'
                
                # Standardize column names for NTL files
                if df['direction'].iloc[0] == 'NTL':
                    column_mapping = {
                        'planned_arrival_time': 'gbtt_pta',
                        'planned_departure_time': 'gbtt_ptd',
                        'actual_arrival_time': 'actual_ta',
                        'actual_departure_time': 'actual_td'
                    }
                    for old_col, new_col in column_mapping.items():
                        if old_col in df.columns:
                            df = df.rename(columns={old_col: new_col})
                
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if not dfs:
            raise ValueError("No valid data files loaded. Please check your file paths.")
        
        self.raw_data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.raw_data)} total records")
        
        self._preprocess_data()
        return self
    
    def _preprocess_data(self):
        """Clean and preprocess the raw train data."""
        print("Preprocessing data...")
        df = self.raw_data.copy()
        
        # Parse dates with multiple formats
        def parse_date(date_str):
            if pd.isna(date_str): return pd.NaT
            formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%m-%Y', '%m-%d-%Y']
            for fmt in formats:
                try: return pd.to_datetime(date_str, format=fmt)
                except: continue
            try: return pd.to_datetime(date_str, dayfirst=True)
            except: return pd.NaT
        
        df['date_of_service'] = df['date_of_service'].apply(parse_date)
        df = df[~df['date_of_service'].isna()]
        
        # Extract time features
        df['day_of_week'] = df['date_of_service'].dt.dayofweek
        df['month'] = df['date_of_service'].dt.month
        df['hour_of_day'] = df['gbtt_pta'].apply(
            lambda x: int(x.split(':')[0]) if pd.notna(x) and ':' in str(x) else np.nan
        )
        
        # Calculate delays
        def time_diff_minutes(t1, t2):
            if pd.isna(t1) or pd.isna(t2): return np.nan
            try:
                h1, m1 = map(int, t1.split(':'))
                h2, m2 = map(int, t2.split(':'))
                min1, min2 = h1 * 60 + m1, h2 * 60 + m2
                diff = min2 - min1
                return diff if -180 <= diff <= 180 else np.nan
            except: return np.nan
        
        df['arrival_delay'] = df.apply(
            lambda row: time_diff_minutes(row['gbtt_pta'], row['actual_ta'])
            if not pd.isna(row['gbtt_pta']) and not pd.isna(row['actual_ta'])
            else np.nan, axis=1
        )
        
        df['departure_delay'] = df.apply(
            lambda row: time_diff_minutes(row['gbtt_ptd'], row['actual_td'])
            if not pd.isna(row['gbtt_ptd']) and not pd.isna(row['actual_td'])
            else np.nan, axis=1
        )
        
        # Map stations to positions in sequence
        def get_position(row):
            station = row['location']
            if pd.isna(station): return -1
            direction = row['direction']
            
            if direction == 'LTN':
                return self.london_to_norwich.index(station) if station in self.london_to_norwich else -1
            elif direction == 'NTL':
                return self.norwich_to_london.index(station) if station in self.norwich_to_london else -1
            return -1
        
        df['station_position'] = df.apply(get_position, axis=1)
        
        # Filter valid data
        valid_data = df[
            (df['station_position'] >= 0) & 
            (~df['arrival_delay'].isna() | ~df['departure_delay'].isna())
        ].copy()
        
        # Create a single delay column
        valid_data['delay'] = valid_data['arrival_delay'].fillna(valid_data['departure_delay'])
        
        self.data = valid_data
        print(f"Preprocessed data: {len(self.data)} valid records")
        return self
    
    def calculate_journey_times(self):
        """Calculate typical journey times between station pairs."""
        print("Calculating journey times...")
        
        journey_groups = self.data.groupby(['rid', 'direction'])
        journey_times = {'LTN': {}, 'NTL': {}}
        
        for (rid, direction), journey in journey_groups:
            journey = journey.sort_values('station_position')
            
            for i, row1 in journey.iterrows():
                from_station = row1['location']
                from_time = row1['actual_ta'] if not pd.isna(row1['actual_ta']) else row1['actual_td']
                
                if pd.isna(from_time): continue
                
                for j, row2 in journey[journey['station_position'] > row1['station_position']].iterrows():
                    to_station = row2['location']
                    to_time = row2['actual_ta'] if not pd.isna(row2['actual_ta']) else row2['actual_td']
                    
                    if pd.isna(to_time): continue
                    
                    # Only calculate times for adjacent station pairs
                    key = f"{from_station}-{to_station}"
                    if key not in self.adjacent_pairs[direction]:
                        continue
                    
                    try:
                        h1, m1 = map(int, from_time.split(':'))
                        h2, m2 = map(int, to_time.split(':'))
                        
                        min1 = h1 * 60 + m1
                        min2 = h2 * 60 + m2
                        
                        if min2 < min1: min2 += 24 * 60
                        
                        journey_time = min2 - min1
                        
                        if 0 < journey_time < 180:
                            if key not in journey_times[direction]:
                                journey_times[direction][key] = []
                            
                            journey_times[direction][key].append(journey_time)
                    except: continue
        
        self.journey_times = {'LTN': {}, 'NTL': {}}
        
        for direction in ['LTN', 'NTL']:
            for key, times in journey_times[direction].items():
                if len(times) >= 5:
                    self.journey_times[direction][key] = {
                        'mean': np.mean(times),
                        'median': np.median(times),
                        'min': np.min(times),
                        'max': np.max(times),
                        'std': np.std(times),
                        'count': len(times)
                    }
        
        print(f"Journey times calculated for {sum(len(d) for d in self.journey_times.values())} station pairs")
        return self
    
    def create_delay_models(self):
        """Create RandomForest models to predict delay propagation between stations."""
        print("Creating delay propagation models...")
        
        journey_groups = self.data.groupby(['rid', 'direction'])
        segment_data = {'LTN': {}, 'NTL': {}}
        
        # Create a list of valid adjacent station pairs
        for (rid, direction), journey in journey_groups:
            journey = journey.sort_values('station_position')
            
            for i in range(len(journey) - 1):
                if i + 1 >= len(journey): continue
                    
                from_row = journey.iloc[i]
                to_row = journey.iloc[i+1]
                
                from_station = from_row['location']
                to_station = to_row['location']
                
                # Skip non-adjacent stations
                key = f"{from_station}-{to_station}"
                if key not in self.adjacent_pairs[direction]:
                    continue
                
                # Skip if delays are missing
                if pd.isna(from_row['delay']) or pd.isna(to_row['delay']):
                    continue
                
                # Create features and target
                features = [
                    from_row['delay'],
                    from_row['day_of_week'],
                    from_row['hour_of_day'] if not pd.isna(from_row['hour_of_day']) else 12,
                    from_row['month']
                ]
                
                target = to_row['delay']
                
                if key not in segment_data[direction]:
                    segment_data[direction][key] = {'X': [], 'y': []}
                
                segment_data[direction][key]['X'].append(features)
                segment_data[direction][key]['y'].append(target)
        
        # Train models with higher minimum sample requirement (30)
        self.segment_models = {'LTN': {}, 'NTL': {}}
        
        for direction in ['LTN', 'NTL']:
            # Train general fallback model for each direction
            all_X, all_y = [], []
            for key, data in segment_data[direction].items():
                if len(data['X']) >= 10:  # Include all reasonable data for fallback
                    all_X.extend(data['X'])
                    all_y.extend(data['y'])
            
            if all_X:
                X = np.array(all_X)
                y = np.array(all_y)
                
                fallback_model = RandomForestRegressor(n_estimators=100, random_state=42)
                fallback_model.fit(X, y)
                
                self.segment_models[direction]['fallback'] = {
                    'model': fallback_model,
                    'model_type': 'RandomForest',
                    'feature_names': ['initial_delay', 'day_of_week', 'hour_of_day', 'month'],
                    'sample_count': len(all_X)
                }
            
            # Train specific models only for pairs with enough data
            for key, data in segment_data[direction].items():
                if len(data['X']) >= 50:  # Higher threshold for specific models
                    X = np.array(data['X'])
                    y = np.array(data['y'])
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=72
                    )
                    
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    pred = model.predict(X_test)
                    mae = mean_absolute_error(y_test, pred)
                    
                    self.segment_models[direction][key] = {
                        'model': model,
                        'mae': mae,
                        'model_type': 'RandomForest',
                        'feature_names': ['initial_delay', 'day_of_week', 'hour_of_day', 'month'],
                        'sample_count': len(data['X'])
                    }
        
        total_models = sum(len(models) for models in self.segment_models.values())
        print(f"Created {total_models} delay propagation models")
        return self
    
    def add_minutes_to_time(self, time_str, minutes_to_add):
        """Add minutes to a time string and return a new time string."""
        h, m = map(int, time_str.split(':'))
        total_minutes = h * 60 + m + round(minutes_to_add)
        
        if total_minutes < 0:
            total_minutes += 24 * 60
        
        total_minutes = total_minutes % (24 * 60)
        new_h = total_minutes // 60
        new_m = total_minutes % 60
        
        return f"{new_h:02d}:{new_m:02d}"
    
    def get_route_and_sequence(self, from_station, to_station):
        """Determine the direction and station sequence for a journey."""
        # Convert station names to codes if needed
        if from_station in self.station_name_to_code:
            from_station = self.station_name_to_code[from_station]
        if to_station in self.station_name_to_code:
            to_station = self.station_name_to_code[to_station]
            
        # Check if going London to Norwich
        ltn_from_idx = self.london_to_norwich.index(from_station) if from_station in self.london_to_norwich else -1
        ltn_to_idx = self.london_to_norwich.index(to_station) if to_station in self.london_to_norwich else -1
        
        if ltn_from_idx != -1 and ltn_to_idx != -1 and ltn_from_idx < ltn_to_idx:
            return {
                'direction': 'LTN',
                'sequence': self.london_to_norwich[ltn_from_idx:ltn_to_idx + 1]
            }
        
        # Check if going Norwich to London
        ntl_from_idx = self.norwich_to_london.index(from_station) if from_station in self.norwich_to_london else -1
        ntl_to_idx = self.norwich_to_london.index(to_station) if to_station in self.norwich_to_london else -1
        
        if ntl_from_idx != -1 and ntl_to_idx != -1 and ntl_from_idx < ntl_to_idx:
            return {
                'direction': 'NTL',
                'sequence': self.norwich_to_london[ntl_from_idx:ntl_to_idx + 1]
            }
        
        return None
    
    def predict_journey(self, current_station, destination, scheduled_arrival_time, current_delay, 
                        day_of_week=None, hour_of_day=None, month=None):
        """Predict arrival time, additional delay, and journey time for a train journey."""
        # Use current date/time for missing values
        now = datetime.now()
        if day_of_week is None: day_of_week = now.weekday()
        if hour_of_day is None: hour_of_day = now.hour
        if month is None: month = now.month
        
        # Get route information
        route_info = self.get_route_and_sequence(current_station, destination)
        
        if not route_info:
            return {'error': f"Cannot determine a valid route from {current_station} to {destination}"}
        
        direction = route_info['direction']
        sequence = route_info['sequence']
        
        # Start with current delay
        delay = current_delay
        predictions = [{'station': sequence[0], 'delay': delay}]
        
        # Predict delay at each subsequent station
        for i in range(1, len(sequence)):
            from_station = sequence[i-1]
            to_station = sequence[i]
            key = f"{from_station}-{to_station}"
            
            # Get model for this segment if available
            if key in self.segment_models[direction]:
                model_info = self.segment_models[direction][key]
                model = model_info['model']
                features = np.array([[delay, day_of_week, hour_of_day, month]])
                delay = model.predict(features)[0]
            elif 'fallback' in self.segment_models[direction]:
                # Use fallback model
                model_info = self.segment_models[direction]['fallback']
                model = model_info['model']
                features = np.array([[delay, day_of_week, hour_of_day, month]])
                delay = model.predict(features)[0]
            else:
                # No model - use simple heuristic
                delay = delay * 0.95
            
            predictions.append({'station': to_station, 'delay': delay})
        
        # Calculate journey time
        journey_time = 0
        segments_found = 0
        
        for i in range(len(sequence) - 1):
            seg_key = f"{sequence[i]}-{sequence[i+1]}"
            if seg_key in self.journey_times[direction]:
                journey_time += self.journey_times[direction][seg_key]['mean']
                segments_found += 1
        
        if segments_found == 0:
            # Fallback: use average of 15 minutes per station
            journey_time = 15 * (len(sequence) - 1)
        
        # Calculate expected arrival time
        final_delay = predictions[-1]['delay']
        additional_delay = final_delay - current_delay
        expected_arrival = self.add_minutes_to_time(scheduled_arrival_time, final_delay)
        
        # Format the result
        result = {
            'journey': {
                'from': self.station_code_to_name.get(sequence[0], sequence[0]),
                'to': self.station_code_to_name.get(sequence[-1], sequence[-1]),
                'direction': 'London to Norwich' if direction == 'LTN' else 'Norwich to London'
            },
            'current_status': {
                'station': self.station_code_to_name.get(sequence[0], sequence[0]),
                'current_delay': round(current_delay, 1),
                'scheduled_arrival': scheduled_arrival_time
            },
            'prediction': {
                'expected_arrival_time': expected_arrival,
                'estimated_journey_time_minutes': round(journey_time, 1),
                'additional_delay_minutes': round(additional_delay, 1),
                'total_delay_at_destination': round(final_delay, 1)
            },
            'intermediate_stations': [
                {
                    'station': self.station_code_to_name.get(p['station'], p['station']),
                    'predicted_delay': round(p['delay'], 1)
                } for p in predictions
            ]
        }
        
        return result
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        model_dir = os.path.dirname(filepath)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        model_data = {
            'station_code_to_name': self.station_code_to_name,
            'london_to_norwich': self.london_to_norwich,
            'norwich_to_london': self.norwich_to_london,
            'segment_models': {
                direction: {
                    key: {k: v for k, v in model_info.items() if k != 'model'} 
                    for key, model_info in models.items()
                } for direction, models in self.segment_models.items()
            },
            'journey_times': self.journey_times
        }
        
        # Save the models separately
        for direction, models in self.segment_models.items():
            for key, model_info in models.items():
                safe_key = key.replace('-', '_')
                model_path = f"{filepath}_{direction}_{safe_key}.pkl"
                model_data['segment_models'][direction][key]['model_path'] = model_path
                joblib.dump(model_info['model'], model_path)
        
        # Save the model metadata
        metadata_path = f"{filepath}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            joblib.dump(model_data, f)
        
        print(f"Model saved to {metadata_path} with {sum(len(m) for m in self.segment_models.values())} segment models")
        return self
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from a file."""
        model = cls()
        
        metadata_path = f"{filepath}_metadata.pkl"
        print(f"Loading model from {metadata_path}")
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Model file {metadata_path} not found")
            
        with open(metadata_path, 'rb') as f:
            model_data = joblib.load(f)
        
        model.station_code_to_name = model_data['station_code_to_name']
        model.station_name_to_code = {v: k for k, v in model.station_code_to_name.items()}
        model.london_to_norwich = model_data['london_to_norwich']
        model.norwich_to_london = model_data['norwich_to_london']
        model.journey_times = model_data['journey_times']
        
        model.segment_models = {'LTN': {}, 'NTL': {}}
        
        for direction, models in model_data['segment_models'].items():
            for key, model_info in models.items():
                model.segment_models[direction][key] = model_info.copy()
                model_path = model_info['model_path']
                
                try:
                    if os.path.exists(model_path):
                        model.segment_models[direction][key]['model'] = joblib.load(model_path)
                    else:
                        print(f"Warning: Model file {model_path} not found, using fallback")
                        fallback = RandomForestRegressor(n_estimators=10, random_state=42)
                        fallback.fit([[0], [10], [20]], [0, 9, 18])
                        model.segment_models[direction][key]['model'] = fallback
                except Exception as e:
                    print(f"Error loading model: {e}, using fallback")
                    fallback = RandomForestRegressor(n_estimators=10, random_state=42)
                    fallback.fit([[0], [10], [20]], [0, 9, 18])
                    model.segment_models[direction][key]['model'] = fallback
        
        print(f"Model loaded with {sum(len(m) for m in model.segment_models.values())} segment models")
        return model


def train_delay_model():
    """Train the delay prediction model using available data files."""
    data_files = [
        "data/2022_service_details_London_to_Norwich.csv",
        "data/2022_service_details_Norwich_to_London.csv",
        "data/2023_service_details_London_to_Norwich.csv", 
        "data/2023_service_details_Norwich_to_London.csv",
        "data/2024_service_details_London_to_Norwich.csv",
        "data/2024_service_details_Norwich_to_London.csv",
    ]
    
    # Filter out duplicate files
    data_files = list(set(data_files))
    
    print(f"Looking for data files in: {os.getcwd()}")
    print(f"Found {len(data_files)} potential data files")
    
    # Create and train the model
    model = TrainDelayModel()
    model.load_data(data_files)
    model.calculate_journey_times()
    model.create_delay_models()
    model.save_model("train_delay_model")
    
    return model

def main():
    """Main function to parse command line arguments and run the appropriate function."""
    parser = argparse.ArgumentParser(description='Train Delay Prediction Model')
    
    # Mode selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train a new model')
    group.add_argument('--predict', action='store_true', help='Make a prediction')
    
    # Prediction arguments
    parser.add_argument('--from', dest='from_station', type=str, help='Current station')
    parser.add_argument('--to', dest='to_station', type=str, help='Destination station')
    parser.add_argument('--scheduled', type=str, help='Scheduled arrival time (HH:MM)')
    parser.add_argument('--delay', type=float, help='Current delay in minutes')
    
    args = parser.parse_args()
    
    try:
        if args.train:
            print("Training new model...")
            model = train_delay_model()
            print("Model training complete!")
            
        elif args.predict:
            # Check if we have all required arguments
            if not all([args.from_station, args.to_station, args.scheduled, args.delay is not None]):
                print("Error: Missing required arguments for prediction")
                print("Required: --from, --to, --scheduled, --delay")
                return
                
            # Load the model
            try:
                model = TrainDelayModel.load_model("train_delay_model")
            except:
                print("Error: No trained model found. Please train a model first with --train")
                return
                
            # Make prediction
            prediction = model.predict_journey(
                current_station=args.from_station,
                destination=args.to_station,
                scheduled_arrival_time=args.scheduled,
                current_delay=args.delay
            )
            
            # Display result
            if 'error' in prediction:
                print(f"Error: {prediction['error']}")
            else:
                print(f"\nJourney: {prediction['journey']['from']} to {prediction['journey']['to']}")
                print(f"Direction: {prediction['journey']['direction']}")
                print(f"Current delay: {prediction['current_status']['current_delay']} minutes")
                print(f"Scheduled arrival: {prediction['current_status']['scheduled_arrival']}")
                print(f"\nPrediction:")
                print(f"  Expected arrival time: {prediction['prediction']['expected_arrival_time']}")
                print(f"  Additional delay: {prediction['prediction']['additional_delay_minutes']} minutes")
                print(f"  Journey time: {prediction['prediction']['estimated_journey_time_minutes']} minutes")
                print(f"  Total delay at destination: {prediction['prediction']['total_delay_at_destination']} minutes")
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()