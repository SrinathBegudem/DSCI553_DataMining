import sys
import time
import json
import math
import numpy as np
from xgboost import XGBRegressor
from pyspark import SparkContext, SparkConf

# Handle collections warning in newer Python versions
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping

# ===================================================================
# DATA LOADING FUNCTIONS
# ===================================================================

def read_csv_data(file_path, include_ratings=True):
    """
    Reads CSV data with appropriate error handling.
    
    Args:
        file_path (str): Path to the CSV file
        include_ratings (bool): Whether to include ratings column
        
    Returns:
        RDD of parsed rows
    """
    # Load the raw CSV file as an RDD of text lines
    raw_data = sc.textFile(file_path)
    
    # Extract the header line
    header = raw_data.first()
    
    # Split rows and filter out malformed ones
    parsed_data = (raw_data.filter(lambda row: row != header)
                   .map(lambda row: row.split(",")))
    
    # Map to appropriate format based on whether we need ratings
    if include_ratings:
        return parsed_data.filter(lambda r: len(r) >= 3)
    else:
        return parsed_data.filter(lambda r: len(r) >= 2)

def load_json_data(file_path, mapper_function):
    """
    Loads and processes JSON data.
    
    Args:
        file_path (str): Path to the JSON file
        mapper_function (function): Function to map each JSON object
        
    Returns:
        RDD of mapped data
    """
    # Load file and parse JSON
    data = sc.textFile(file_path)
    return data.map(lambda row: json.loads(row)).map(mapper_function)

# ===================================================================
# FEATURE EXTRACTION FUNCTIONS
# ===================================================================

def extract_review_features(folder_path):
    """
    Extracts features from review data.
    
    Args:
        folder_path (str): Path to the folder containing review data
        
    Returns:
        dict: Mapping of business_id to review features (useful, funny, cool)
    """
    # ---------------------------
    # 1. Load review data
    # ---------------------------
    review_mapper = lambda row: (row['business_id'], 
                               (float(row['useful']), float(row['funny']), float(row['cool'])))
    review_data = load_json_data(folder_path + '/review_train.json', review_mapper)
    
    # Group by business_id
    review_grouped = review_data.groupByKey().mapValues(list)
    
    # ---------------------------
    # 2. Collect and process review data
    # ---------------------------
    # First collect the raw review data
    review_dict = {}
    for business_id, features_list in review_grouped.collect():
        review_dict[business_id] = features_list
    
    # Calculate average metrics per business
    business_review_features = {}
    for business_id, reviews in review_dict.items():
        if not reviews:
            continue
            
        useful_sum = funny_sum = cool_sum = 0
        count = len(reviews)
        
        for useful, funny, cool in reviews:
            useful_sum += useful
            funny_sum += funny
            cool_sum += cool
            
        business_review_features[business_id] = (
            useful_sum / count,  # Average useful rating
            funny_sum / count,   # Average funny rating
            cool_sum / count     # Average cool rating
        )
        
    return business_review_features

def extract_user_features(folder_path):
    """
    Extracts features from user data.
    
    Args:
        folder_path (str): Path to the folder containing user data
        
    Returns:
        dict: Mapping of user_id to user features (avg_stars, review_count, fans)
    """
    # ---------------------------
    # 1. Load user data
    # ---------------------------
    user_mapper = lambda row: (row['user_id'], 
                             (float(row['average_stars']), 
                              float(row['review_count']), 
                              float(row['fans'])))
    user_data = load_json_data(folder_path + '/user.json', user_mapper)
    
    # ---------------------------
    # 2. Collect user data
    # ---------------------------
    user_features = {}
    for user_id, features in user_data.collect():
        user_features[user_id] = features
        
    return user_features

def extract_business_features(folder_path):
    """
    Extracts features from business data.
    
    Args:
        folder_path (str): Path to the folder containing business data
        
    Returns:
        dict: Mapping of business_id to business features (stars, review_count)
    """
    # ---------------------------
    # 1. Load business data
    # ---------------------------
    business_mapper = lambda row: (row['business_id'], 
                                 (float(row['stars']), 
                                  float(row['review_count'])))
    business_data = load_json_data(folder_path + '/business.json', business_mapper)
    
    # ---------------------------
    # 2. Collect business data
    # ---------------------------
    business_features = {}
    for business_id, features in business_data.collect():
        business_features[business_id] = features
        
    return business_features

# ===================================================================
# FEATURE VECTOR BUILDING FUNCTIONS
# ===================================================================

def build_feature_vector(user_id, business_id, business_review_features, user_features, business_features):
    """
    Builds a feature vector for a user-business pair.
    
    Args:
        user_id (str): ID of the user
        business_id (str): ID of the business
        business_review_features (dict): Review features per business
        user_features (dict): Features per user
        business_features (dict): Features per business
        
    Returns:
        list: Feature vector
    """
    # ---------------------------
    # 1. Extract review features
    # ---------------------------
    if business_id in business_review_features:
        useful, funny, cool = business_review_features[business_id]
    else:
        # Default values if missing
        useful = funny = cool = 0.0
    
    # ---------------------------
    # 2. Extract user features
    # ---------------------------
    if user_id in user_features:
        user_avg_stars, user_review_count, user_fans = user_features[user_id]
    else:
        # Default values if missing
        user_avg_stars = 3.5  # Global average
        user_review_count = 0.0
        user_fans = 0.0
    
    # ---------------------------
    # 3. Extract business features
    # ---------------------------
    if business_id in business_features:
        business_avg_stars, business_review_count = business_features[business_id]
    else:
        # Default values if missing
        business_avg_stars = 3.5  # Global average
        business_review_count = 0.0
    
    # ---------------------------
    # 4. Build complete feature vector
    # ---------------------------
    # Combine all features into a single vector
    return [
        useful, funny, cool,
        user_avg_stars, user_review_count, user_fans,
        business_avg_stars, business_review_count
    ]

def prepare_training_data(train_data, business_review_features, user_features, business_features):
    """
    Prepares feature vectors and target values for model training.
    
    Args:
        train_data (RDD): Training data with user_id, business_id, rating
        business_review_features (dict): Review features per business
        user_features (dict): Features per user
        business_features (dict): Features per business
        
    Returns:
        tuple: (X_train, y_train) as numpy arrays
    """
    X_train = []  # Feature vectors
    y_train = []  # Target ratings
    
    # For each training sample, create a feature vector
    for row in train_data.collect():
        user_id, business_id, rating = row[0], row[1], float(row[2])
        
        # Store target rating
        y_train.append(rating)
        
        # Build feature vector
        feature_vector = build_feature_vector(
            user_id, business_id, 
            business_review_features, user_features, business_features
        )
        
        X_train.append(feature_vector)
    
    # Convert to numpy arrays for XGBoost
    return np.array(X_train, dtype='float32'), np.array(y_train, dtype='float32')

def prepare_test_data(test_data, business_review_features, user_features, business_features):
    """
    Prepares feature vectors for prediction.
    
    Args:
        test_data (RDD): Test data with user_id, business_id
        business_review_features (dict): Review features per business
        user_features (dict): Features per user
        business_features (dict): Features per business
        
    Returns:
        tuple: (user_business_pairs, X_test)
    """
    user_business_pairs = []
    X_test = []
    
    # For each test sample, create a feature vector
    for row in test_data.collect():
        user_id, business_id = row[0], row[1]
        
        # Store user-business pair for output
        user_business_pairs.append((user_id, business_id))
        
        # Build feature vector
        feature_vector = build_feature_vector(
            user_id, business_id, 
            business_review_features, user_features, business_features
        )
        
        X_test.append(feature_vector)
    
    # Convert to numpy array for XGBoost
    return user_business_pairs, np.array(X_test, dtype='float32')

# ===================================================================
# MODEL TRAINING AND PREDICTION
# ===================================================================

def train_xgboost_model(X_train, y_train):
    """
    Trains an XGBoost regression model.
    
    Args:
        X_train (numpy.array): Training feature vectors
        y_train (numpy.array): Target rating values
        
    Returns:
        XGBRegressor: Trained model
    """
    # ---------------------------
    # 1. Set model parameters
    # ---------------------------
    # These parameters are tuned for good performance
    params = {
        'learning_rate': 0.02,
        'max_depth': 17,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'min_child_weight': 101,
        'n_estimators': 300,
        'random_state': 42
    }
    
    # ---------------------------
    # 2. Initialize and train model
    # ---------------------------
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model

def predict_ratings(model, X_test):
    """
    Generates predictions from a trained model.
    
    Args:
        model (XGBRegressor): Trained model
        X_test (numpy.array): Test feature vectors
        
    Returns:
        numpy.array: Predicted ratings
    """
    # Generate raw predictions
    predictions = model.predict(X_test)
    
    # Ensure predictions are within valid range [1.0, 5.0]
    return np.clip(predictions, 1.0, 5.0)

# ===================================================================
# EVALUATION FUNCTIONS
# ===================================================================

def calculate_rmse(predictions, actual_ratings):
    """
    Calculates Root Mean Square Error between predictions and actual ratings.
    
    Args:
        predictions (list): List of predicted ratings
        actual_ratings (list): List of actual ratings
        
    Returns:
        float: RMSE value
    """
    if len(predictions) != len(actual_ratings):
        raise ValueError("Prediction and actual rating lists must be the same length")
        
    squared_errors = [(pred - actual) ** 2 for pred, actual in zip(predictions, actual_ratings)]
    mean_squared_error = sum(squared_errors) / len(squared_errors)
    
    return math.sqrt(mean_squared_error)

def calculate_error_distribution(predictions, actual_ratings):
    """
    Calculates the distribution of prediction errors across different ranges.
    
    Args:
        predictions (list): List of predicted ratings
        actual_ratings (list): List of actual ratings
        
    Returns:
        dict: Dictionary with error ranges as keys and counts as values
    """
    error_ranges = {
        ">=0 and <1": 0,
        ">=1 and <2": 0,
        ">=2 and <3": 0,
        ">=3 and <4": 0,
        ">=4": 0
    }
    
    for pred, actual in zip(predictions, actual_ratings):
        error = abs(pred - actual)
        
        if error < 1:
            error_ranges[">=0 and <1"] += 1
        elif error < 2:
            error_ranges[">=1 and <2"] += 1
        elif error < 3:
            error_ranges[">=2 and <3"] += 1
        elif error < 4:
            error_ranges[">=3 and <4"] += 1
        else:
            error_ranges[">=4"] += 1
            
    return error_ranges

# ===================================================================
# MAIN FUNCTION
# ===================================================================

def main():
    """
    Main function for Task 2.2: Model-based Recommendation System
    """
    # ---------------------------
    # 1. Setup and Initialization
    # ---------------------------
    # Start timing execution
    start_time = time.time()
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_2.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Initialize Spark context
    conf = SparkConf().setAppName("Task2_2_ModelBasedRS")
    global sc
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")  # Reduce log verbosity
    
    # ---------------------------
    # 2. Extract Features
    # ---------------------------
    print("Extracting features...")
    
    # Extract review features
    business_review_features = extract_review_features(folder_path)
    
    # Extract user features
    user_features = extract_user_features(folder_path)
    
    # Extract business features
    business_features = extract_business_features(folder_path)
    
    # ---------------------------
    # 3. Prepare Training Data
    # ---------------------------
    print("Preparing training data...")
    
    # Load training data
    train_data = read_csv_data(folder_path + '/yelp_train.csv')
    
    # Create feature vectors and target values
    X_train, y_train = prepare_training_data(
        train_data, business_review_features, user_features, business_features
    )
    
    # ---------------------------
    # 4. Train Model
    # ---------------------------
    print("Training XGBoost model...")
    
    # Train XGBoost model
    model = train_xgboost_model(X_train, y_train)
    
    # ---------------------------
    # 5. Prepare Test Data
    # ---------------------------
    print("Preparing test data...")
    
    # Load test data
    test_data = read_csv_data(test_file, include_ratings=False)
    
    # Create feature vectors for prediction
    user_business_pairs, X_test = prepare_test_data(
        test_data, business_review_features, user_features, business_features
    )
    
    # ---------------------------
    # 6. Generate Predictions
    # ---------------------------
    print("Generating predictions...")
    
    # Make predictions using trained model
    predictions = predict_ratings(model, X_test)
    
    # ---------------------------
    # 7. Write Output File
    # ---------------------------
    print("Writing output file...")
    
    # Write predictions to CSV
    with open(output_file, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for i, (user_id, business_id) in enumerate(user_business_pairs):
            f.write(f"{user_id},{business_id},{predictions[i]}\n")
    
    # ---------------------------
    # 8. Calculate RMSE (if test file has ratings)
    # ---------------------------
    try:
        # Try to read test data with ratings
        test_data_with_ratings = read_csv_data(test_file, include_ratings=True)
        test_ratings = test_data_with_ratings.map(lambda r: (r[0], r[1], float(r[2]))).collect()
        
        # Create dictionaries for lookup
        actual_ratings_dict = {(user_id, business_id): rating for user_id, business_id, rating in test_ratings}
        actual_ratings = []
        
        # Extract actual ratings in the same order as predictions
        for i, (user_id, business_id) in enumerate(user_business_pairs):
            if (user_id, business_id) in actual_ratings_dict:
                actual_ratings.append(actual_ratings_dict[(user_id, business_id)])
            else:
                # If no actual rating, remove the corresponding prediction
                predictions[i] = None
        
        # Remove None values
        filtered_predictions = [p for p in predictions if p is not None]
        
        # Calculate and print RMSE
        if filtered_predictions:
            rmse = calculate_rmse(filtered_predictions, actual_ratings)
            print(f"RMSE: {rmse}")
            
            # Calculate error distribution
            error_dist = calculate_error_distribution(filtered_predictions, actual_ratings)
            print("Error Distribution:")
            for range_label, count in error_dist.items():
                print(f"  {range_label}: {count}")
    except:
        # If RMSE calculation fails, just skip it
        pass
    
    # ---------------------------
    # 9. Finalization
    # ---------------------------
    # Print execution time
    execution_time = time.time() - start_time
    print(f"Task 2.2 completed successfully in {execution_time:.2f} seconds.")
    
    # Stop Spark context
    sc.stop()

if __name__ == "__main__":
    main()