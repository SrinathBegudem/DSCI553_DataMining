# Method Description:
# This project implements an advanced recommendation system for Yelp ratings prediction.
# Starting with my homework 3 code (RMSE 0.99), I iteratively optimized the system through
# comprehensive feature engineering and model tuning. I initially improved to RMSE 0.98,
# then dedicated substantial effort to win the competition.
#
# Key improvements include:
# 1. Enhanced feature extraction: Added user metrics (elite status, compliment averages),
#    business attributes (price range, number of categories), and interaction data (checkins).
# 2. Feature selection: Used SHAP and Boruta analysis locally to identify critical features,
#    applying RFE to evaluate accuracy variations with different feature sets.
# 3. Model optimization: Fine-tuned XGBoost hyperparameters through extensive Optuna trials
#    with particular focus on regularization parameters.
# 4. Feature scaling: Applied MinMaxScaler to normalize features before model training.
# 5. Efficient processing: Optimized memory usage and execution flow to meet time constraints.
#
# While I experimented with ensemble models and multi-level approaches, these exceeded the
# 25-minute execution limit. The final submission uses a single optimized XGBoost model with
# carefully selected features, strictly using training data for training and validation data
# for testing, with no data leakage.
#
# Error Distribution:
# >=0 and <1: 102701
# >=1 and <2: 32454
# >=2 and <3: 6076
# >=3 and <4: 809
# >=4: 4
#
# RMSE: 0.974887
#
# Execution Time: 600 seconds
# ===================================================================

import csv
import json
import os
import sys
import time
from datetime import datetime
import math # For sqrt

import numpy as np
import pandas as pd
from pyspark import SparkConf, SparkContext
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error # To calculate RMSE
from xgboost import XGBRegressor
import collections
import collections.abc

# Handle collections warning for compatibility with different Python versions
if not hasattr(collections, 'Mapping') or not issubclass(collections.Mapping, collections.abc.Mapping):
    collections.Mapping = collections.abc.Mapping

class Path:
    """
    Defines paths for temporary files that store intermediate processed data.
    Using distinct filenames prevents using stale data from previous runs.
    """
    yelp_train_processed: str = "yelp_train_processed_final_21f.csv"
    yelp_test_processed: str = "yelp_test_processed_final_21f.csv"

class DataReader:
    """
    Handles reading data using Spark RDDs with comprehensive error handling.
    Ensures robustness when dealing with potentially malformed input data.
    """
    def __init__(self, sc: SparkContext):
        self.sc = sc # Store SparkContext instance

    def read_csv_spark(self, path: str):
        """
        Reads CSV files, filters header row, and splits rows into columns.
        Includes robust error handling for empty files and malformed data.
        
        Args:
            path: Path to the CSV file
            
        Returns:
            Tuple of (data_rdd, header_columns)
        """
        try:
            rdd = self.sc.textFile(path)
            if rdd.isEmpty(): 
                print(f"Warning: CSV file is empty: {path}")
                return self.sc.parallelize([]), []
            
            header = rdd.first()
            data_rdd = rdd.filter(lambda row: row != header)
            
            if data_rdd.isEmpty(): 
                print(f"Warning: No data rows in CSV: {path}")
                return self.sc.parallelize([]), []
                
            return data_rdd.map(lambda r: r.split(",")).filter(lambda x: len(x)>1), header.split(',')
        except Exception as e: 
            print(f"Error reading CSV {path}: {e}")
            return self.sc.parallelize([]), []

    def read_json_spark(self, path: str):
        """
        Reads JSON lines file, parsing each line safely to handle malformed JSON.
        
        Args:
            path: Path to the JSON file
            
        Returns:
            RDD containing parsed JSON objects
        """
        try:
            raw_rdd = self.sc.textFile(path)
            if raw_rdd.isEmpty(): 
                print(f"Warning: JSON file is empty: {path}")
                return self.sc.parallelize([])
                
            def safe_json(l): 
                try: return json.loads(l)
                except: return None
                
            rdd = raw_rdd.map(safe_json).filter(lambda x: x is not None)
            
            if rdd.isEmpty(): 
                print(f"Warning: No valid JSON objects in: {path}")
                return self.sc.parallelize([])
                
            return rdd
        except Exception as e: 
            print(f"Error reading JSON {path}: {e}")
            return self.sc.parallelize([])

class BusinessData:
    """
    Processes business data RDD, extracting business features including price range.
    Handles attribute extraction and type conversion safely.
    """
    @staticmethod
    def parse_row(row: dict):
        """
        Parses a single business JSON object, adding price_range and other derived fields.
        Handles missing values and type conversions safely.
        """
        # Extract basic attributes with safe default values
        attrs = row.get("attributes")
        cats_str = row.get("categories", "")
        
        # Calculate derived features
        row["num_attrs"] = len(attrs) if isinstance(attrs, dict) else 0
        row["num_categories"] = len(cats_str.split(",")) if isinstance(cats_str, str) else 0
        row["stars"] = float(row.get("stars", 3.5))
        row["review_count"] = int(row.get("review_count", 0))
        row["is_open"] = int(row.get("is_open", 0))
        
        # Extract price range (important feature identified by SHAP analysis)
        price_range = 0
        if isinstance(attrs, dict):
            try: 
                price_range = int(attrs.get("RestaurantsPriceRange2", "0"))
            except: 
                price_range = 0  # Default if missing/invalid
        row["price_range"] = price_range
        
        return row

    @staticmethod
    def process(rdd):
        """
        Transforms business RDD to extract features including price_range.
        Returns a dictionary mapping business_id to feature tuple.
        """
        if rdd.isEmpty(): return {}
        try:
            # The tuple now includes price_range as the 6th element
            feature_rdd = rdd.map(BusinessData.parse_row).map(
                lambda row: (row.get("business_id"),
                (row["stars"], row["review_count"], row["is_open"], row["num_attrs"],
                 row["num_categories"], row["price_range"])) 
            ).filter(lambda x: x[0] is not None)
            
            return feature_rdd.cache().collectAsMap()
        except Exception as e: 
            print(f"Error processing BusinessData: {e}")
            return {}

class UserData:
    """
    Processes user data RDD to extract comprehensive user features including
    compliments, elite status, membership duration, and social connections.
    """
    # All compliment types to aggregate
    compliment_keys = ["compliment_hot","compliment_more","compliment_profile","compliment_cute",
                       "compliment_list","compliment_note","compliment_plain","compliment_cool",
                       "compliment_funny","compliment_writer","compliment_photos"]
    lc = len(compliment_keys) if compliment_keys else 1
    
    @staticmethod
    def parse_row(row: dict):
        """
        Parses a single user JSON object, extracting and computing derived features.
        Calculates membership duration, elite status, and average compliments.
        """
        # Extract elite status and friends count
        elite = row.get("elite", "None")
        friends = row.get("friends", "None")
        
        # Calculate elite and friends metrics
        row["num_elite"] = len(elite.split(",")) if isinstance(elite, str) and elite != "None" else 0
        row["num_friends"] = len(friends.split(",")) if isinstance(friends, str) and friends != "None" else 0
        
        # Calculate average compliments (feature identified as important)
        row["avg_compliment"] = float(sum(int(row.get(k,0)) for k in UserData.compliment_keys)) / UserData.lc
        
        # Calculate membership duration in years
        membership_years = 0.0
        if isinstance(row.get("yelping_since"), str):
            try:
                yelp_since = pd.to_datetime(row.get("yelping_since"), errors='coerce')
                if not pd.isna(yelp_since): 
                    membership_years = max(0.0, (datetime.now() - yelp_since).days / 365.25)
            except Exception: 
                pass
        row["membership_years"] = membership_years
        
        # Extract basic metrics with defaults
        row["average_stars"] = float(row.get("average_stars", 3.5))
        row["review_count"] = int(row.get("review_count", 0))
        row["useful"] = int(row.get("useful",0))
        row["funny"] = int(row.get("funny",0))
        row["cool"] = int(row.get("cool",0))
        row["fans"] = int(row.get("fans",0))
        
        return row
        
    @staticmethod
    def process(rdd):
        """
        Transforms user RDD to extract and compute user features.
        Returns a dictionary mapping user_id to feature tuple.
        """
        if rdd.isEmpty(): return {}
        try:
            return rdd.map(UserData.parse_row).map(
                lambda row: (row.get("user_id"),
                (row["review_count"],row["useful"],row["funny"],row["cool"],row["fans"],
                 row["average_stars"],row["num_elite"],row["num_friends"],
                 row["avg_compliment"],row["membership_years"]))
            ).filter(lambda x: x[0] is not None).cache().collectAsMap()
        except Exception as e: 
            print(f"Error processing UserData: {e}")
            return {}

class ReviewData:
    """
    Processes review data RDD to extract user-business specific statistics
    from interactions in the training data.
    """
    @staticmethod
    def process(rdd):
        """
        Extracts averages for stars and votes by user-business pairs.
        Returns dictionary keyed by (user_id, business_id) pairs.
        """
        if rdd.isEmpty(): return {}
        try:
            def safe_map(rd: dict):
                """Maps review to (user_id, business_id) key with relevant metrics"""
                try:
                    uid, bid = rd.get("user_id"), rd.get("business_id")
                    if uid is None or bid is None: return None
                    # Extract votes received by THIS SPECIFIC review (user-business interaction)
                    return ((uid, bid), (float(rd.get("stars",0)), 
                                         float(rd.get("useful",0)), 
                                         float(rd.get("funny",0)), 
                                         float(rd.get("cool",0)), 1))
                except: 
                    return None
                    
            # Aggregate review data by user-business pair
            return rdd.map(safe_map).filter(lambda x: x is not None).reduceByKey(
                lambda a,b: (a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3],a[4]+b[4])
            ).mapValues(
                # Calculate averages for the aggregated values
                lambda v: (v[0]/v[4],v[1]/v[4],v[2]/v[4],v[3]/v[4]) if v[4]>0 else (0,0,0,0)
            ).cache().collectAsMap()
        except Exception as e: 
            print(f"Error processing ReviewData: {e}")
            return {}

class TipData:
    """
    Processes tip data RDD for user-business specific statistics.
    Extracts like counts and tip frequencies.
    """
    @staticmethod
    def process(rdd):
        """
        Aggregates tip metrics by user-business pairs.
        Returns dictionary with (total_likes, num_tips) tuples.
        """
        if rdd.isEmpty(): return {}
        try:
            def safe_map(row: dict):
                """Maps tip to (user_id, business_id) key with metrics"""
                try:
                    uid, bid = row.get("user_id"), row.get("business_id")
                    if uid is None or bid is None: return None
                    return ((uid, bid), (int(row.get("likes",0)), 1))  # (total_likes, num_tips)
                except: 
                    return None
                    
            # Aggregate tips by user-business pair
            return rdd.map(safe_map).filter(lambda x: x is not None).reduceByKey(
                lambda a,b: (a[0]+b[0], a[1]+b[1])
            ).cache().collectAsMap()
        except Exception as e: 
            print(f"Error processing TipData: {e}")
            return {}

class PhotoData:
    """
    Processes photo data RDD for business-specific statistics.
    Extracts photo counts and distinct label counts.
    """
    @staticmethod
    def process(rdd):
        """
        Aggregates photo metrics by business.
        Returns dictionary with (distinct_labels, total_photos) tuples.
        """
        if rdd.isEmpty(): return {}
        try:
            def safe_map(row: dict):
                """Maps photo to business_id with label and count"""
                try:
                    lbl, bid = row.get("label"), row.get("business_id")
                    if isinstance(lbl, str) and bid is not None: 
                        return (bid, ([lbl], 1))  # (list_of_labels, photo_count)
                    return None
                except: 
                    return None
                    
            # Aggregate photos by business
            return rdd.map(safe_map).filter(lambda x: x is not None).reduceByKey(
                lambda a,b: (a[0]+b[0], a[1]+b[1])
            ).mapValues(
                lambda v: (len(set(v[0])), v[1])  # (distinct_labels, total_photos)
            ).cache().collectAsMap()
        except Exception as e: 
            print(f"Error processing PhotoData: {e}")
            return {}

class CheckinData:
    """
    Processes checkin data RDD for business-specific total checkins.
    This feature was identified as important through SHAP analysis.
    """
    @staticmethod
    def process(rdd):
        """
        Calculates total checkins for each business.
        Returns dictionary mapping business_id to checkin count.
        """
        if rdd.isEmpty(): return {}
        try:
            def safe_map(row_dict):
                """Maps checkin data to business_id with total count"""
                try:
                    bid = row_dict.get("business_id")
                    time_dict = row_dict.get("time")
                    if bid is None or not isinstance(time_dict, dict): return None
                    total_checkins = sum(time_dict.values())  # Sum all check-in counts
                    return (bid, total_checkins)
                except: 
                    return None
                    
            # Sum checkins for each business (handles potential duplicate entries)
            reduced_rdd = rdd.map(safe_map).filter(lambda x: x is not None).reduceByKey(lambda x, y: x + y)
            return reduced_rdd.cache().collectAsMap()
        except Exception as e:
            print(f"Error processing CheckinData: {e}")
            return {}

class ModelBasedConfig:
    """
    Stores configuration for the model-based approach.
    Includes feature drop lists and optimized model parameters.
    """
    # Define the 4 user-business specific review features to be dropped
    # These were found to potentially cause overfitting based on local validation
    features_to_drop_from_generated: list = [
        "ub_review_avg_stars", "ub_useful_votes", "ub_funny_votes", "ub_cool_votes"
    ]
    
    # Columns to drop before model training (IDs, target, and the 4 specific review stats)
    drop_cols: list = ["user_id", "business_id", "rating"] + features_to_drop_from_generated
    
    # Best parameters found via extensive Optuna tuning (500+ trials)
    # These were selected based on their performance and execution time constraints
    params: dict = {
        'learning_rate': 0.01843630531111744, 
        'n_estimators': 1200, 
        'max_depth': 11,
        'min_child_weight': 112, 
        'gamma': 0.8252496439150409, 
        'subsample': 0.9537964027679018,
        'colsample_bytree': 0.352705973553061, 
        'lambda': 68.75749748337805,
        'alpha': 0.19199863219597071, 
        'random_state': 2020, 
        'n_jobs': -1,
        'objective': 'reg:linear'
    }
    pred_cols: list = ["user_id", "business_id", "prediction"]


def create_dataset_vector(row, usr_dict, bus_dict, review_dict_ub, tip_dict_ub, photo_dict_bus, checkin_dict):
    """
    Creates a comprehensive feature vector for each user-business pair.
    Includes features from all data sources and handles missing values.
    
    Args:
        row: CSV row containing user_id, business_id, and optional rating
        usr_dict: Dictionary of user features
        bus_dict: Dictionary of business features
        review_dict_ub: Dictionary of user-business review stats
        tip_dict_ub: Dictionary of user-business tip stats
        photo_dict_bus: Dictionary of business photo stats
        checkin_dict: Dictionary of business checkin counts
        
    Returns:
        Tuple of (user_id, business_id, feature_vector, rating)
    """
    # Extract user_id, business_id, and rating from row
    if len(row) == 3: 
        usr_id, bus_id, rating_str = row
    elif len(row) == 2: 
        usr_id, bus_id, rating_str = row[0], row[1], None
    else: 
        return None
        
    # Convert rating to float if present
    try: 
        rating = float(rating_str) if rating_str is not None else None
    except: 
        rating = None

    # Define default values for missing entries
    default_user = (0,0,0,0,0,3.5,0,0,0,0)
    default_business = (3.5,0,0,0,0,0)  # Added default for price
    default_review_ub = (0,0,0,0)
    default_tip_ub = (0,0)
    default_photo_bus = (0,0)

    # Extract user features
    (usr_review_count, usr_total_useful_votes, usr_total_funny_votes, usr_total_cool_votes, 
     usr_fans, usr_avg_stars, num_elite, num_friends, usr_avg_compliment, 
     membership_years) = usr_dict.get(usr_id, default_user)
     
    # Extract business features (including price_range)
    (bus_avg_stars, bus_review_count, bus_is_open, num_attrs, num_categories,
     price_range) = bus_dict.get(bus_id, default_business)
     
    # Extract user-business review features
    (ub_review_avg_stars, ub_useful_votes, ub_funny_votes, 
     ub_cool_votes) = review_dict_ub.get((usr_id, bus_id), default_review_ub)
     
    # Extract tip features
    (tip_total_likes, tip_count) = tip_dict_ub.get((usr_id, bus_id), default_tip_ub)
    
    # Extract photo features
    (photo_distinct_labels, photo_total_count) = photo_dict_bus.get(bus_id, default_photo_bus)
    
    # Extract checkin features
    total_checkins = checkin_dict.get(bus_id, 0)

    # Assemble all features in specific order 
    # (order must match feature_names in process_data)
    feature_vector_values = [
        # User features (10)
        usr_review_count, usr_total_useful_votes, usr_total_funny_votes, usr_total_cool_votes, 
        usr_fans, usr_avg_stars, num_elite, num_friends, usr_avg_compliment, membership_years,
        
        # Business features (5+1)
        bus_avg_stars, bus_review_count, bus_is_open, num_attrs, num_categories,
        
        # User-Business review features (4) - These will be dropped before model training
        ub_review_avg_stars, ub_useful_votes, ub_funny_votes, ub_cool_votes,
        
        # Tip, photo, and additional features (5)
        tip_total_likes, tip_count,
        photo_distinct_labels, photo_total_count,
        total_checkins, price_range
    ]
    
    # Convert all features to float and handle NaN/infinite values
    feature_vector_float = [float(f) if f is not None and np.isfinite(f) else 0.0 
                           for f in feature_vector_values]
                           
    return (usr_id, bus_id, feature_vector_float, rating)


def save_predictions(data: list, output_file_name: str):
    """
    Saves the final predictions to a CSV file in the required format.
    
    Args:
        data: List of (user_id, business_id, prediction) tuples
        output_file_name: Path to save the predictions
    """
    header = ModelBasedConfig.pred_cols
    try:
        with open(output_file_name, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
        print(f"Predictions successfully saved to {output_file_name}")
    except Exception as e: 
        print(f"Error writing output file {output_file_name}: {e}")


def process_data(folder_path: str, test_file_name: str, sc: SparkContext):
    """
    Main data processing pipeline that loads data, extracts features, and saves processed files.
    Uses Spark RDDs for initial processing and aggregation.
    
    Args:
        folder_path: Path to the folder containing all input data files
        test_file_name: Path to the test/validation file
        sc: SparkContext for distributed processing
    """
    start_time = time.time()
    print("Starting Data Processing (with checkins, price_range)...")
    data_reader = DataReader(sc)
    processed_train_path = Path.yelp_train_processed
    processed_test_path = Path.yelp_test_processed
    
    try:
        # Step 1: Load all feature dictionaries from JSON files
        print("  Loading feature dictionaries...")
        
        # User features dictionary: user_id -> feature tuple
        usr_dict = UserData.process(data_reader.read_json_spark(os.path.join(folder_path, "user.json")))
        
        # Business features dictionary: business_id -> feature tuple
        bus_dict = BusinessData.process(data_reader.read_json_spark(os.path.join(folder_path, "business.json")))
        
        # User-Business review features: (user_id, business_id) -> feature tuple
        review_dict_ub = ReviewData.process(data_reader.read_json_spark(os.path.join(folder_path, "review_train.json")))
        
        # User-Business tip features: (user_id, business_id) -> feature tuple
        tip_dict_ub = TipData.process(data_reader.read_json_spark(os.path.join(folder_path, "tip.json")))
        
        # Business photo features: business_id -> feature tuple
        photo_dict_bus = PhotoData.process(data_reader.read_json_spark(os.path.join(folder_path, "photo.json")))
        
        # Business checkin features: business_id -> count
        checkin_dict = CheckinData.process(data_reader.read_json_spark(os.path.join(folder_path, "checkin.json")))
        
        print("  Finished loading feature dictionaries.")

        # Step 2: Load train and test CSV files
        print("  Loading train/test CSVs...")
        train_rdd, _ = data_reader.read_csv_spark(os.path.join(folder_path, "yelp_train.csv"))
        test_rdd, _ = data_reader.read_csv_spark(test_file_name)

        # Step 3: Create feature vectors for each user-business pair
        print("  Creating feature vectors (including new features)...")
        train_vectors_rdd = train_rdd.map(
            lambda r: create_dataset_vector(r, usr_dict, bus_dict, review_dict_ub, 
                                           tip_dict_ub, photo_dict_bus, checkin_dict)
        ).filter(lambda x: x is not None)
        
        test_vectors_rdd = test_rdd.map(
            lambda r: create_dataset_vector(r, usr_dict, bus_dict, review_dict_ub, 
                                           tip_dict_ub, photo_dict_bus, checkin_dict)
        ).filter(lambda x: x is not None)

        # Step 4: Collect vectors to the driver node
        print("  Collecting feature vectors...")
        train_collected = train_vectors_rdd.collect()
        test_collected = test_vectors_rdd.collect()
        print(f"  Collected {len(train_collected)} train vectors and {len(test_collected)} test vectors.")

        # Define feature names in exact order generated by create_dataset_vector
        feature_names = [
            # User features (10)
            "usr_review_count", "usr_useful", "usr_funny", "usr_cool", "usr_fans", "usr_avg_stars",
            "num_elite", "num_friends", "usr_avg_comp", "membership_years",
            
            # Business features (5)
            "bus_avg_stars", "bus_review_count", "bus_is_open", "num_attrs", "num_categories",
            
            # User-business specific review stats (4) - will be dropped before model training
            "ub_review_avg_stars", "ub_useful_votes", "ub_funny_votes", "ub_cool_votes",
            
            # Tip, photo features (4)
            "tip_total_likes", "tip_count",
            "photo_distinct_labels", "photo_total_count",
            
            # Additional features identified as important (2)
            "total_checkins", "price_range"
        ]
        column_names = ["user_id", "business_id"] + feature_names + ["rating"]
        
        # Step 5: Create Pandas DataFrames with all features
        print("  Creating Pandas DataFrames...")
        train_df = pd.DataFrame(
            [(i[0], i[1], *i[2], i[3]) for i in train_collected], 
            columns=column_names
        )
        test_df = pd.DataFrame(
            [(i[0], i[1], *i[2], i[3]) for i in test_collected],
            columns=column_names
        )
        
        # Step 6: Save processed data to CSV files for model training
        print(f"  Saving processed DataFrames with {len(feature_names)} features...")
        train_df.to_csv(processed_train_path, index=False)
        test_df.to_csv(processed_test_path, index=False)
        
    except Exception as e: 
        print(f"Exception during data processing:\n{e}")
        import traceback
        traceback.print_exc()
        raise
        
    print(f"Data Processing Duration: {(time.time()-start_time):.2f} s\n")


def train_predict_model(train_data_path: str, test_data_path: str):
    """
    Trains the XGBoost model on processed data and generates predictions.
    Uses MinMaxScaler for feature normalization before training.
    
    Args:
        train_data_path: Path to processed training data CSV
        test_data_path: Path to processed test data CSV
        
    Returns:
        List of (user_id, business_id, prediction) tuples or None on error
    """
    start_time = time.time()
    print("Starting Model Training and Prediction (21 features)...")
    try:
        # Step 1: Read processed data (contains all 25 generated features)
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        print(f"  Loaded processed train data: {train_df.shape}")
        print(f"  Loaded processed test data: {test_df.shape}")

        # Step 2: Handle missing values and infinities
        train_df.fillna(0, inplace=True)
        test_df.fillna(0, inplace=True)
        train_df.replace([np.inf, -np.inf], 0, inplace=True)
        test_df.replace([np.inf, -np.inf], 0, inplace=True)

        # Step 3: Define features (X) and target (y) for training
        # Drop user/business IDs, target variable, and user-business specific review stats
        X_train = train_df.drop(columns=ModelBasedConfig.drop_cols)
        y_train = train_df["rating"]
        X_test = test_df.drop(columns=ModelBasedConfig.drop_cols)
        test_ids = test_df[["user_id", "business_id"]]  # Keep original test IDs

        # Print feature count for verification
        model_feature_names = X_train.columns.tolist()
        print(f"  Training model with {len(model_feature_names)} features.")
        # print(f"  Features: {model_feature_names}")  # Uncomment to verify features

        # Step 4: Ensure column order consistency between train and test
        X_test = X_test[model_feature_names]

        # Step 5: Scale features using MinMaxScaler (important for model performance)
        print("  Scaling features...")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


# Step 6: Train XGBoost Regression Model with optimized parameters
        print("  Training final XGBoost model...")
        final_params = ModelBasedConfig.params.copy()
        model = XGBRegressor(**final_params)
        model.fit(X_train_scaled, y_train)

        # Step 7: Predict ratings for test data
        print("  Predicting on test data...")
        y_test_pred = model.predict(X_test_scaled)
        y_test_pred_clipped = np.clip(y_test_pred, 1.0, 5.0)  # Clip to valid rating range [1-5]

        # Step 8: Format predictions for output
        output_df = test_ids.copy()
        output_df["prediction"] = y_test_pred_clipped
        pred_data = output_df.values.tolist()

    except FileNotFoundError: 
        print(f"Error: Processed file not found.")
        return None
    except Exception as e: 
        print(f"Error during model training/prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

    execution_time = time.time() - start_time
    print(f"Model Training & Prediction Time: {execution_time:.2f} s\n")
    return pred_data


def calculate_rmse_from_files(test_processed_path: str, output_file_name: str):
    """
    Calculates RMSE and error distribution by comparing processed test data and predictions.
    Only used for validation purposes to evaluate model performance.
    
    Args:
        test_processed_path: Path to the processed test data (with ground truth ratings)
        output_file_name: Path to the prediction output file
    """
    try:
        # Load test data with ground truth ratings and predictions
        test_df = pd.read_csv(test_processed_path)  # Contains all generated features + rating
        pred_df = pd.read_csv(output_file_name)     # Contains user_id, business_id, prediction
        
        if 'rating' not in test_df.columns: 
            print("Warning: 'rating' column missing in test data.")
            return
            
        # Merge test data with predictions on user_id and business_id
        merged_df = pd.merge(test_df[['user_id', 'business_id', 'rating']],
                             pred_df[['user_id', 'business_id', 'prediction']],
                             on=['user_id', 'business_id'], how='inner')
        merged_df.dropna(subset=['rating'], inplace=True)
        
        if merged_df.empty: 
            print("Warning: No matching rows/valid ratings for evaluation.")
            return

        # Calculate RMSE
        actual = merged_df['rating']
        predicted = merged_df['prediction']
        rmse = math.sqrt(mean_squared_error(actual, predicted))
        print(f"\nRMSE (calculated from files): {rmse:.6f}")

        # Calculate Error Distribution
        merged_df["error"] = abs(actual - predicted)
        bins = [-np.inf, 1, 2, 3, 4, np.inf]
        labels = [">=0 and <1:", ">=1 and <2:", ">=2 and <3:", ">=3 and <4:", ">=4:"]
        merged_df["Error Distribution:"] = pd.cut(merged_df["error"], bins=bins, labels=labels, right=False)
        error_distribution = merged_df["Error Distribution:"].value_counts().sort_index()
        print("\nError Distribution:")
        print(error_distribution)

    except FileNotFoundError: 
        print(f"Warning: Could not find files to calculate RMSE.")
    except Exception as e: 
        print(f"Error calculating RMSE/Error Distribution: {e}")


def main(folder_path: str, test_file_name: str, output_file_name: str):
    """
    Main execution workflow that coordinates data processing, model training, and prediction.
    Handles SparkContext initialization and cleanup of temporary files.

    Args:
        folder_path: Path to the folder containing all input data
        test_file_name: Path to the test/validation file
        output_file_name: Path to save the prediction results
    """
    overall_start_time = time.time()
    sc_local = None

    try:
        # Step 1: Initialize Spark context with memory configuration
        conf_local = SparkConf().setAppName("Competition_Main_21_Features")
        try:
            conf_local.set("spark.driver.memory", "4g").set("spark.executor.memory", "4g")
            sc_local = SparkContext(conf=conf_local)
            sc_local.setLogLevel("ERROR")
            print("SparkContext initialized with 4g memory.")
        except Exception as memory_error:
            print(f"Failed to set 4g memory: {memory_error}. Using default settings.")
            conf_local = SparkConf().setAppName("Competition_Main_21_Features")  # Recreate conf_local with default settings
            sc_local = SparkContext(conf=conf_local)
            sc_local.setLogLevel("ERROR")

        # Step 2: Process data using Spark RDDs (generates 25 features)
        process_data(folder_path, test_file_name, sc_local)

        # Step 3: Train model and generate predictions (uses 21 features after dropping 4)
        pred_data = train_predict_model(Path.yelp_train_processed, Path.yelp_test_processed)

        # Step 4: Save predictions and calculate metrics if successful
        if pred_data is not None:
            save_predictions(pred_data, output_file_name)
            # Calculate final metrics using the saved files (for validation)
            calculate_rmse_from_files(Path.yelp_test_processed, output_file_name)
        else:
            print("Prediction generation failed.")

    except Exception as e:
        print(f"An error occurred in the main workflow: {e}")
        traceback.print_exc()  # Print the full traceback to help debug
    finally:
        # Step 5: Clean up resources
        # Stop Spark context
        if sc_local:
            try:
                if not sc_local._jsc.sc().isStopped():
                    print("Stopping Spark context...")
                    sc_local.stop()
                else:
                    print("SparkContext was already stopped.")
            except Exception as stop_error:
                print(f"Error stopping Spark context: {stop_error}")
                traceback.print_exc()  # Print traceback for stop error

        # Remove temporary intermediate files
        try:
            if os.path.exists(Path.yelp_train_processed):
                os.remove(Path.yelp_train_processed)
            if os.path.exists(Path.yelp_test_processed):
                os.remove(Path.yelp_test_processed)
            print("Cleaned up intermediate processed files.")
        except OSError as os_error:
            print(f"Error removing temporary files: {os_error}")
            traceback.print_exc()  # Print traceback for OS error

        # Print total execution time
        execution_time = time.time() - overall_start_time
        print(f"\nTotal Execution Time: {execution_time:.2f} s\n")


if __name__ == "__main__":
    # Validate command-line arguments
    if len(sys.argv) != 4:
        print("Usage: spark-submit competition.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)
        
    # Extract command-line arguments
    main(sys.argv[1], sys.argv[2], sys.argv[3])