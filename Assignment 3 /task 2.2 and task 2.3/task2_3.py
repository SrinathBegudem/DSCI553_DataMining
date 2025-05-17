import sys
import time
import json
import math
import numpy as np
from xgboost import XGBRegressor
from pyspark import SparkContext, SparkConf
from collections import defaultdict

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
# ITEM-BASED CF FUNCTIONS
# ===================================================================

def build_cf_datastructures(train_data):
    """
    Builds data structures needed for collaborative filtering.
    
    Args:
        train_data (RDD): Training data with user_id, business_id, rating
        
    Returns:
        tuple: Various dictionaries for collaborative filtering
    """
    # ---------------------------
    # 1. Build business-to-users mapping
    # ---------------------------
    business_user_rdd = train_data.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set)
    business_users_dict = business_user_rdd.collectAsMap()
    
    # ---------------------------
    # 2. Build user-to-businesses mapping
    # ---------------------------
    user_business_rdd = train_data.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set)
    user_businesses_dict = user_business_rdd.collectAsMap()
    
    # ---------------------------
    # 3. Calculate business average ratings
    # ---------------------------
    business_avg_rdd = (train_data.map(lambda row: (row[1], float(row[2])))
                                 .groupByKey()
                                 .mapValues(lambda ratings: sum(ratings)/len(ratings)))
    business_avg_dict = business_avg_rdd.collectAsMap()
    
    # ---------------------------
    # 4. Calculate user average ratings
    # ---------------------------
    user_avg_rdd = (train_data.map(lambda row: (row[0], float(row[2])))
                             .groupByKey()
                             .mapValues(lambda ratings: sum(ratings)/len(ratings)))
    user_avg_dict = user_avg_rdd.collectAsMap()
    
    # ---------------------------
    # 5. Build business-to-user-ratings mapping
    # ---------------------------
    business_user_ratings_rdd = (train_data.map(lambda row: (row[1], (row[0], float(row[2]))))
                                          .groupByKey())
    
    business_user_ratings_dict = {}
    for business, user_ratings in business_user_ratings_rdd.collect():
        temp = {}
        for user, rating in user_ratings:
            temp[user] = rating
        business_user_ratings_dict[business] = temp
    
    # ---------------------------
    # 6. Calculate global average
    # ---------------------------
    global_avg = train_data.map(lambda row: float(row[2])).mean()
    
    return (business_users_dict, user_businesses_dict, business_avg_dict, 
            user_avg_dict, business_user_ratings_dict, global_avg)

def compute_pearson_similarity(business_1, business_2, business_users_dict, 
                              business_avg_dict, business_user_ratings_dict, similarity_cache, global_avg):
    """
    Computes Pearson similarity between two businesses with caching.
    
    Args:
        business_1 (str): First business ID
        business_2 (str): Second business ID
        business_users_dict (dict): Mapping of business to users who rated it
        business_avg_dict (dict): Mapping of business to average rating
        business_user_ratings_dict (dict): Mapping of business to {user: rating}
        similarity_cache (dict): Cache of already computed similarities
        global_avg (float): Global average rating
        
    Returns:
        float: Pearson similarity between the two businesses
    """
    # Create a canonical key for caching (always sort to ensure consistency)
    cache_key = tuple(sorted([business_1, business_2]))
    
    # Return cached value if already computed
    if cache_key in similarity_cache:
        return similarity_cache[cache_key]
    
    # Check if either business doesn't exist
    if business_1 not in business_users_dict or business_2 not in business_users_dict:
        similarity_cache[cache_key] = 0.0
        return 0.0
    
    # Find common users who rated both businesses
    users1 = business_users_dict.get(business_1, set())
    users2 = business_users_dict.get(business_2, set())
    common_users = users1.intersection(users2)
    
    # Handle special cases with few common users
    if len(common_users) <= 1:
        # Use rating difference-based similarity as fallback
        b1_avg = business_avg_dict.get(business_1, global_avg)
        b2_avg = business_avg_dict.get(business_2, global_avg)
        similarity = (5.0 - abs(b1_avg - b2_avg)) / 5.0
        
        # Penalize for having very few common users
        similarity *= 0.5
        
        similarity_cache[cache_key] = similarity
        return similarity
    
    # Optimization for exactly 2 common users
    if len(common_users) == 2:
        users_list = list(common_users)
        b1_ratings = business_user_ratings_dict.get(business_1, {})
        b2_ratings = business_user_ratings_dict.get(business_2, {})
        
        rating1_1 = b1_ratings.get(users_list[0], global_avg)
        rating1_2 = b2_ratings.get(users_list[0], global_avg)
        rating2_1 = b1_ratings.get(users_list[1], global_avg)
        rating2_2 = b2_ratings.get(users_list[1], global_avg)
        
        sim1 = (5.0 - abs(rating1_1 - rating1_2)) / 5.0
        sim2 = (5.0 - abs(rating2_1 - rating2_2)) / 5.0
        similarity = (sim1 + sim2) / 2
        
        # Slight penalty for having only 2 common users
        similarity *= 0.7
        
        similarity_cache[cache_key] = similarity
        return similarity
    
    # For cases with 3+ common users, compute full Pearson correlation
    ratings1 = []
    ratings2 = []
    
    b1_ratings = business_user_ratings_dict.get(business_1, {})
    b2_ratings = business_user_ratings_dict.get(business_2, {})
    
    for user in common_users:
        if user in b1_ratings and user in b2_ratings:
            ratings1.append(b1_ratings[user])
            ratings2.append(b2_ratings[user])
    
    # If we lost some common users due to dict lookup, recheck
    if len(ratings1) <= 1:
        similarity = (5.0 - abs(business_avg_dict.get(business_1, global_avg) - 
                              business_avg_dict.get(business_2, global_avg))) / 5.0
        similarity *= 0.5
        similarity_cache[cache_key] = similarity
        return similarity
    
    # Calculate means
    mean1 = sum(ratings1) / len(ratings1)
    mean2 = sum(ratings2) / len(ratings2)
    
    # Center the ratings
    centered1 = [r - mean1 for r in ratings1]
    centered2 = [r - mean2 for r in ratings2]
    
    # Calculate numerator and denominator for Pearson correlation
    numerator = sum(a * b for a, b in zip(centered1, centered2))
    denominator = (sum(a ** 2 for a in centered1) ** 0.5) * (sum(b ** 2 for b in centered2) ** 0.5)
    
    # Handle division by zero
    if denominator == 0:
        similarity = 0.0
    else:
        similarity = numerator / denominator
        
        # Apply significance weighting - higher weight for more common users
        significance_factor = min(1.0, len(ratings1) / 50)
        similarity *= significance_factor
    
    # Cache the result for future use
    similarity_cache[cache_key] = similarity
    return similarity

def predict_rating_item_based(user_id, business_id, user_businesses_dict, business_users_dict,
                             business_avg_dict, user_avg_dict, business_user_ratings_dict, 
                             similarity_cache, global_avg, neighbor_count=15):
    """
    Predicts rating using item-based collaborative filtering.
    
    Args:
        user_id (str): ID of the user
        business_id (str): ID of the business
        user_businesses_dict (dict): Mapping of user to businesses they rated
        business_users_dict (dict): Mapping of business to users who rated it
        business_avg_dict (dict): Mapping of business to average rating
        user_avg_dict (dict): Mapping of user to average rating
        business_user_ratings_dict (dict): Mapping of business to {user: rating}
        similarity_cache (dict): Cache of already computed similarities
        global_avg (float): Global average rating
        neighbor_count (int): Number of neighbors to consider
        
    Returns:
        tuple: (predicted_rating, confidence) - confidence is 0.0-1.0 indicating prediction quality
    """
    # Case 1: If user hasn't rated any businesses or business hasn't been rated by anyone
    if user_id not in user_businesses_dict or business_id not in business_users_dict:
        # If user has previous ratings, use their average
        if user_id in user_avg_dict:
            return user_avg_dict[user_id], 0.1
        # If business has ratings, use its average
        elif business_id in business_avg_dict:
            return business_avg_dict[business_id], 0.1
        # Fallback to global average
        else:
            return global_avg, 0.05
    
    # Case 2: Use item-based CF with Pearson similarity
    similar_items = []
    
    # For each business rated by this user, compute similarity with target business
    for rated_business in user_businesses_dict[user_id]:
        # Skip if same business
        if rated_business == business_id:
            continue
        
        # Skip if business not in training data
        if rated_business not in business_users_dict:
            continue
        
        # Compute similarity between the two businesses
        similarity = compute_pearson_similarity(
            business_id, rated_business, business_users_dict,
            business_avg_dict, business_user_ratings_dict, similarity_cache, global_avg
        )
        
        # Only use positive similarity scores
        if similarity > 0:
            rating = business_user_ratings_dict.get(rated_business, {}).get(user_id, global_avg)
            similar_items.append((similarity, rating))
    
    # If no similar items, use user's average rating
    if not similar_items:
        return user_avg_dict.get(user_id, global_avg), 0.1
    
    # Sort by similarity (descending) and take top neighbors
    similar_items.sort(key=lambda x: -x[0])
    top_neighbors = similar_items[:neighbor_count]
    
    # Calculate weighted average rating
    numerator = sum(sim * rating for sim, rating in top_neighbors)
    denominator = sum(abs(sim) for sim, rating in top_neighbors)
    
    if denominator == 0:
        return user_avg_dict.get(user_id, global_avg), 0.1
    
    # Calculate confidence based on similarity values and number of neighbors
    confidence = min(0.6, (len(top_neighbors) / neighbor_count) * 
                   (sum(sim for sim, _ in top_neighbors) / len(top_neighbors)))
    
    # Compute final prediction
    predicted_rating = numerator / denominator
    
    # Ensure rating is within valid range
    predicted_rating = min(max(predicted_rating, 1.0), 5.0)
    
    return predicted_rating, confidence

# ===================================================================
# MODEL-BASED FUNCTIONS
# ===================================================================

def extract_review_features(folder_path):
    """
    Extracts features from review data.
    
    Args:
        folder_path (str): Path to the folder containing review data
        
    Returns:
        dict: Mapping of business_id to review features (useful, funny, cool)
    """
    # Load review data
    review_mapper = lambda row: (row['business_id'], 
                               (float(row['useful']), float(row['funny']), float(row['cool'])))
    review_data = load_json_data(folder_path + '/review_train.json', review_mapper)
    
    # Group by business_id
    review_grouped = review_data.groupByKey().mapValues(list)
    
    # Collect and process review data
    business_review_features = {}
    
    for business_id, features_list in review_grouped.collect():
        if not features_list:
            continue
            
        useful_sum = funny_sum = cool_sum = 0
        count = len(features_list)
        
        for useful, funny, cool in features_list:
            useful_sum += useful
            funny_sum += funny
            cool_sum += cool
            
        business_review_features[business_id] = (
            useful_sum / count,  # Average useful rating
            funny_sum / count,   # Average funny rating
            cool_sum / count,    # Average cool rating
            count                # Number of reviews
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
    # Load user data with more features
    user_mapper = lambda row: (row['user_id'], 
                             (float(row['average_stars']), 
                              float(row['review_count']), 
                              float(row['fans']),
                              float(row.get('useful', 0)),
                              float(row.get('funny', 0)),
                              float(row.get('cool', 0)),
                              float(len(row.get('friends', [])))
                             ))
    user_data = load_json_data(folder_path + '/user.json', user_mapper)
    
    # Collect user data
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
    # Load business data with more features
    business_mapper = lambda row: (row['business_id'], 
                                 (float(row['stars']), 
                                  float(row['review_count']),
                                  float(1 if row.get('is_open', 0) == 1 else 0),
                                  float(len(row.get('categories', '').split(',')) if row.get('categories') else 0)
                                 ))
    business_data = load_json_data(folder_path + '/business.json', business_mapper)
    
    # Collect business data
    business_features = {}
    for business_id, features in business_data.collect():
        business_features[business_id] = features
        
    return business_features

def build_feature_vector(user_id, business_id, business_review_features, 
                        user_features, business_features, user_avg_dict,
                        business_avg_dict, global_avg):
    """
    Builds a feature vector for a user-business pair.
    
    Args:
        user_id (str): ID of the user
        business_id (str): ID of the business
        business_review_features (dict): Review features per business
        user_features (dict): Features per user
        business_features (dict): Features per business
        user_avg_dict (dict): Mapping of user to average rating
        business_avg_dict (dict): Mapping of business to average rating
        global_avg (float): Global average rating
        
    Returns:
        list: Feature vector
    """
    # Extract review features
    if business_id in business_review_features:
        useful, funny, cool, num_reviews = business_review_features[business_id]
    else:
        # Default values if missing
        useful = funny = cool = 0.0
        num_reviews = 0.0
    
    # Extract user features
    if user_id in user_features:
        user_avg_stars, user_review_count, user_fans, user_useful, user_funny, user_cool, user_friends = user_features[user_id]
    else:
        # Default values if missing
        user_avg_stars = user_avg_dict.get(user_id, global_avg)
        user_review_count = user_useful = user_funny = user_cool = user_friends = user_fans = 0.0
    
    # Extract business features
    if business_id in business_features:
        business_avg_stars, business_review_count, is_open, num_categories = business_features[business_id]
    else:
        # Default values if missing
        business_avg_stars = business_avg_dict.get(business_id, global_avg)
        business_review_count = is_open = num_categories = 0.0
    
    # Build extended feature vector with more interactions/combinations
    return [
        # Base features
        useful, funny, cool, num_reviews,
        user_avg_stars, user_review_count, user_fans, user_useful, user_funny, user_cool, user_friends,
        business_avg_stars, business_review_count, is_open, num_categories,
        
        # Interaction features
        user_avg_stars * business_avg_stars,  # Rating interaction
        user_review_count * business_review_count,  # Activity level interaction
        
        # Normalized features
        useful / (num_reviews + 1),  # Per-review useful
        (user_useful + user_funny + user_cool) / (user_review_count + 1),  # User engagement
        
        # Business popularity and user activity match
        min(user_review_count, business_review_count) / (max(user_review_count, business_review_count) + 1),
        
        # Bias terms
        user_avg_stars - global_avg,  # User rating bias
        business_avg_stars - global_avg,  # Business rating bias
    ]

def prepare_model_data(train_data, test_data, business_review_features, 
                     user_features, business_features, user_avg_dict,
                     business_avg_dict, global_avg):
    """
    Prepares data for model training and prediction.
    
    Args:
        train_data (RDD): Training data
        test_data (RDD): Test data
        business_review_features (dict): Review features per business
        user_features (dict): Features per user
        business_features (dict): Features per business
        user_avg_dict (dict): Mapping of user to average rating
        business_avg_dict (dict): Mapping of business to average rating
        global_avg (float): Global average rating
        
    Returns:
        tuple: (X_train, y_train, X_test, user_business_pairs)
    """
    # Prepare training data
    X_train = []
    y_train = []
    
    for row in train_data.collect():
        user_id, business_id, rating = row[0], row[1], float(row[2])
        
        y_train.append(rating)
        
        feature_vector = build_feature_vector(
            user_id, business_id, 
            business_review_features, user_features, business_features,
            user_avg_dict, business_avg_dict, global_avg
        )
        X_train.append(feature_vector)
    
    # Prepare test data
    X_test = []
    user_business_pairs = []
    
    for row in test_data.collect():
        user_id, business_id = row[0], row[1]
        
        user_business_pairs.append((user_id, business_id))
        
        feature_vector = build_feature_vector(
            user_id, business_id, 
            business_review_features, user_features, business_features,
            user_avg_dict, business_avg_dict, global_avg
        )
        X_test.append(feature_vector)
    
    # Convert to numpy arrays for XGBoost
    X_train = np.array(X_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    X_test = np.array(X_test, dtype='float32')
    
    return X_train, y_train, X_test, user_business_pairs

def train_xgboost_model(X_train, y_train):
    """
    Trains an XGBoost regression model.
    
    Args:
        X_train (numpy.array): Training feature vectors
        y_train (numpy.array): Target rating values
        
    Returns:
        XGBRegressor: Trained model
    """
    # Enhanced parameters based on results
    params = {
        'learning_rate': 0.01,
        'max_depth': 17,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'min_child_weight': 101,
        'n_estimators': 500,
        'random_state': 42,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'gamma': 0.1
    }
    
    # Initialize and train model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model

def predict_model_based(model, X_test):
    """
    Generates predictions from the model.
    
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
# HYBRID RECOMMENDATION FUNCTIONS
# ===================================================================

def calculate_dynamic_weight(user_id, business_id, user_businesses_dict, business_users_dict, 
                           user_avg_dict, business_avg_dict, cf_confidence):
    """
    Calculates a dynamic weight for combining CF and model-based predictions.
    
    Args:
        user_id (str): ID of the user
        business_id (str): ID of the business
        user_businesses_dict (dict): Mapping of user to businesses they rated
        business_users_dict (dict): Mapping of business to users who rated it
        user_avg_dict (dict): Mapping of user to average rating
        business_avg_dict (dict): Mapping of business to average rating
        cf_confidence (float): Confidence in the CF prediction
        
    Returns:
        float: Weight for CF prediction (between 0 and 1)
    """
    # Favor model-based prediction by starting with low weight for CF
    weight = 0.2 * cf_confidence
    
    # Check if user exists in training data
    if user_id in user_businesses_dict:
        user_activity = len(user_businesses_dict[user_id])
        # If user has many ratings, increase CF weight
        if user_activity > 30:
            weight += 0.05
        # If user has very few ratings, decrease CF weight further
        elif user_activity < 5:
            weight -= 0.05
    else:
        # New user - rely almost entirely on model
        weight = 0.05
    
    # Check if business exists in training data
    if business_id in business_users_dict:
        business_popularity = len(business_users_dict[business_id])
        # If business has many ratings, increase CF weight slightly
        if business_popularity > 50:
            weight += 0.05
        # If business has very few ratings, decrease CF weight further
        elif business_popularity < 10:
            weight -= 0.05
    else:
        # New business - rely almost entirely on model
        weight = 0.05
    
    # If user and business both have extreme average ratings, CF might be more reliable
    if user_id in user_avg_dict and business_id in business_avg_dict:
        user_avg = user_avg_dict[user_id]
        business_avg = business_avg_dict[business_id]
        
        if (user_avg > 4.5 or user_avg < 1.5) and (business_avg > 4.5 or business_avg < 1.5):
            weight += 0.1
    
    # Normalize weight to be between 0 and 1
    weight = min(max(weight, 0.05), 0.4)  # Cap at 40% for CF
    
    return weight

def combine_predictions(cf_predictions, model_predictions, user_business_pairs, 
                       user_businesses_dict, business_users_dict, user_avg_dict, business_avg_dict):
    """
    Combines item-based CF and model-based predictions.
    
    Args:
        cf_predictions (list): List of (prediction, confidence) tuples from CF
        model_predictions (numpy.array): Array of predictions from model
        user_business_pairs (list): List of (user_id, business_id) pairs
        user_businesses_dict (dict): Mapping of user to businesses they rated
        business_users_dict (dict): Mapping of business to users who rated it
        user_avg_dict (dict): Mapping of user to average rating
        business_avg_dict (dict): Mapping of business to average rating
        
    Returns:
        list: Combined predictions as (user_id, business_id, prediction) tuples
    """
    combined_predictions = []
    
    for i, ((user_id, business_id), (cf_pred, cf_confidence)) in enumerate(zip(user_business_pairs, cf_predictions)):
        model_pred = model_predictions[i]
        
        # Calculate dynamic weight based on various factors
        cf_weight = calculate_dynamic_weight(
            user_id, business_id, 
            user_businesses_dict, business_users_dict,
            user_avg_dict, business_avg_dict, cf_confidence
        )
        
        # Combine predictions using weighted average
        final_pred = cf_weight * cf_pred + (1 - cf_weight) * model_pred
        
        # Ensure prediction is within valid range
        final_pred = min(max(final_pred, 1.0), 5.0)
        
        combined_predictions.append((user_id, business_id, final_pred))
    
    return combined_predictions

# ===================================================================
# EVALUATION FUNCTIONS
# ===================================================================

def calculate_rmse(predictions, validation_data):
    """
    Calculates Root Mean Square Error.
    
    Args:
        predictions (list): List of (user_id, business_id, prediction) tuples
        validation_data (RDD): Validation data with (user_id, business_id, rating)
        
    Returns:
        float: RMSE value
    """
    # Convert predictions to a dictionary for fast lookup
    pred_dict = {(user_id, business_id): pred for user_id, business_id, pred in predictions}
    
    # Create (prediction, actual) pairs
    squared_errors = []
    
    for row in validation_data.collect():
        user_id, business_id, actual = row[0], row[1], float(row[2])
        key = (user_id, business_id)
        
        if key in pred_dict:
            pred = pred_dict[key]
            squared_errors.append((pred - actual) ** 2)
    
    # Calculate RMSE
    if not squared_errors:
        return 0.0
    
    return math.sqrt(sum(squared_errors) / len(squared_errors))

def calculate_error_distribution(predictions, validation_data):
    """
    Calculates the distribution of prediction errors.
    
    Args:
        predictions (list): List of (user_id, business_id, prediction) tuples
        validation_data (RDD): Validation data with (user_id, business_id, rating)
        
    Returns:
        dict: Dictionary with error ranges as keys and counts as values
    """
    # Convert predictions to a dictionary for fast lookup
    pred_dict = {(user_id, business_id): pred for user_id, business_id, pred in predictions}
    
    # Initialize error ranges
    error_ranges = {
        ">=0 and <1": 0,
        ">=1 and <2": 0,
        ">=2 and <3": 0,
        ">=3 and <4": 0,
        ">=4": 0
    }
    
    # Count errors in each range
    for row in validation_data.collect():
        user_id, business_id, actual = row[0], row[1], float(row[2])
        key = (user_id, business_id)
        
        if key in pred_dict:
            pred = pred_dict[key]
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
    Main function for Task 2.3: Hybrid Recommendation System
    """
    # ---------------------------
    # 1. Setup and Initialization
    # ---------------------------
    # Start timing execution
    start_time = time.time()
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_3.py <folder_path> <test_file_name> <output_file_name>")
        sys.exit(1)
        
    folder_path = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Initialize Spark context
    conf = SparkConf().setAppName("Task2_3_HybridRS")
    global sc
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")  # Reduce log verbosity
    
    print("Hybrid recommendation system starting...")
    
    # ---------------------------
    # 2. Load and Process Data
    # ---------------------------
    print("Loading and processing data...")
    
    # Read training data
    train_data = read_csv_data(folder_path + '/yelp_train.csv')
    train_data.cache()
    
    # Read test data
    test_data = read_csv_data(test_file, include_ratings=False)
    
    # ---------------------------
    # 3. Build Item-based CF Components
    # ---------------------------
    print("Building item-based CF components...")
    
    # Build data structures for CF
    (business_users_dict, user_businesses_dict, business_avg_dict, 
     user_avg_dict, business_user_ratings_dict, global_avg) = build_cf_datastructures(train_data)
    
    # Initialize similarity cache
    similarity_cache = {}
    
    # ---------------------------
    # 4. Extract Features for Model-based RS
    # ---------------------------
    print("Extracting features for model-based RS...")
    
    # Extract features from additional data sources
    business_review_features = extract_review_features(folder_path)
    user_features = extract_user_features(folder_path)
    business_features = extract_business_features(folder_path)
    
    # ---------------------------
    # 5. Prepare Data for Model Training
    # ---------------------------
    print("Preparing data for model training...")
    
    # Prepare training and test data for the model
    X_train, y_train, X_test, user_business_pairs = prepare_model_data(
        train_data, test_data, 
        business_review_features, user_features, business_features,
        user_avg_dict, business_avg_dict, global_avg
    )
    
    # ---------------------------
    # 6. Train and Use Model-based RS
    # ---------------------------
    print("Training model-based RS...")
    
    # Train XGBoost model
    model = train_xgboost_model(X_train, y_train)
    
    # Generate model-based predictions
    model_predictions = predict_model_based(model, X_test)
    
    # ---------------------------
    # 7. Generate Item-based CF Predictions
    # ---------------------------
    print("Generating item-based CF predictions...")
    
    # Generate item-based CF predictions for test data
    cf_predictions = []
    
    for user_id, business_id in user_business_pairs:
        prediction, confidence = predict_rating_item_based(
            user_id, business_id, 
            user_businesses_dict, business_users_dict,
            business_avg_dict, user_avg_dict, business_user_ratings_dict,
            similarity_cache, global_avg
        )
        cf_predictions.append((prediction, confidence))
    
    # ---------------------------
    # 8. Combine Predictions
    # ---------------------------
    print("Combining predictions...")
    
    # Combine CF and model-based predictions
    combined_predictions = combine_predictions(
        cf_predictions, model_predictions, user_business_pairs,
        user_businesses_dict, business_users_dict, user_avg_dict, business_avg_dict
    )
    
    # ---------------------------
    # 9. Write Output File
    # ---------------------------
    print("Writing output file...")
    
    # Write predictions to CSV
    with open(output_file, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for user_id, business_id, prediction in combined_predictions:
            f.write(f"{user_id},{business_id},{prediction}\n")
    
    # ---------------------------
    # 10. Evaluate Performance (if possible)
    # ---------------------------
   # try:
     #   # Try to read validation data with ratings
       # validation_data = read_csv_data(test_file, include_ratings=True)
        
       # # Calculate and print RMSE
       # rmse = calculate_rmse(combined_predictions, validation_data)
       # print(f"RMSE: {rmse}")
        
       # # Calculate and print error distribution
       # error_dist = calculate_error_distribution(combined_predictions, validation_data)
       # print("Error Distribution:")
       # for range_label, count in error_dist.items():
         #   print(f"  {range_label}: {count}")
   # except Exception as e:
        # If evaluation fails, just skip it
        # print(f"Evaluation skipped: {str(e)}")
      #  pass
    
    # ---------------------------
    # 11. Finalization
    # ---------------------------
    # Print execution time
    execution_time = time.time() - start_time
    print(f"Task 2.3 completed successfully in {execution_time:.2f} seconds.")
    
    # Stop Spark context
    sc.stop()

if __name__ == "__main__":
    main()