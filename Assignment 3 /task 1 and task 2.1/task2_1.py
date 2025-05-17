import sys
import time
import csv
import math
from itertools import combinations
from collections import defaultdict
from pyspark import SparkContext, SparkConf

# ===================================================================
# HELPER FUNCTIONS FOR DATA PROCESSING
# ===================================================================

def read_train_data(train_file):
    """
    Reads the training CSV file and processes it into (user, business, rating) records.
    Handles potential CSV format issues to prevent index errors.
    
    Returns:
        RDD of tuples (user_id, business_id, rating)
    """
    # Load the raw CSV file as an RDD of text lines
    raw_data = sc.textFile(train_file)
    
    # Extract the header line (first line of the CSV)
    header = raw_data.first()
    
    # Filter out header and safely parse rows with these steps:
    # 1. Remove the header row
    # 2. Split each row by comma delimiter
    # 3. Filter out any malformed rows that don't have enough columns
    # 4. Convert to the required format with rating as float
    return (raw_data.filter(lambda row: row != header)
            .map(lambda row: row.split(","))
            .filter(lambda r: len(r) >= 3)  # Filter out malformed rows
            .map(lambda r: (r[0], r[1], float(r[2]))))

def read_test_data(test_file, include_ratings=False):
    """
    Reads the test CSV file containing (user, business) pairs.
    Handles potential CSV format issues to prevent index errors.
    
    Args:
        test_file (str): Path to test file
        include_ratings (bool): Whether to include ratings in the result or just user-business pairs
    
    Returns:
        RDD of tuples (user_id, business_id) or (user_id, business_id, rating)
    """
    # Load the raw CSV file as an RDD of text lines
    raw_data = sc.textFile(test_file)
    
    # Extract the header line (first line of the CSV)
    header = raw_data.first()
    
    # Split rows and filter out malformed ones:
    # 1. Remove the header row
    # 2. Split each row by comma delimiter
    # 3. Apply different filter based on whether we need ratings or not
    parsed_data = (raw_data.filter(lambda row: row != header)
                   .map(lambda row: row.split(","))
                   .filter(lambda r: len(r) >= 3 if include_ratings else len(r) >= 2))
    
    # Return data in the appropriate format:
    # - If ratings are needed, include the third column as a float
    # - Otherwise, just return user_id and business_id
    if include_ratings:
        return parsed_data.map(lambda r: (r[0], r[1], float(r[2])))
    else:
        return parsed_data.map(lambda r: (r[0], r[1]))

def compute_averages(train_rdd):
    """
    Computes average ratings at three levels:
    - Per business: a mapping of business -> average rating.
    - Per user: a mapping of user -> average rating.
    - Global average: overall average rating.
    
    Returns:
        Tuple: (business_avg, user_avg, global_avg)
    """
    # ---------------------------
    # 1. Calculate business averages
    # ---------------------------
    # For each record, create (business_id, (rating, 1)) pairs
    # Reduce by key to sum ratings and count per business
    # Compute average by dividing sum by count
    business_avg = (train_rdd.map(lambda x: (x[1], (x[2], 1)))
                              .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                              .mapValues(lambda t: t[0] / t[1])
                              .collectAsMap())
    
    # ---------------------------
    # 2. Calculate user averages
    # ---------------------------
    # Similar approach but grouping by user_id instead
    user_avg = (train_rdd.map(lambda x: (x[0], (x[2], 1)))
                          .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))
                          .mapValues(lambda t: t[0] / t[1])
                          .collectAsMap())
    
    # ---------------------------
    # 3. Calculate global average
    # ---------------------------
    # Extract all ratings with a count of 1, then reduce to get total sum and count
    # Divide to get the overall average rating across all users and businesses
    total, count = train_rdd.map(lambda x: (x[2], 1)).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]))
    global_avg = total / count
    
    return business_avg, user_avg, global_avg

def build_rating_dicts(train_rdd):
    """
    Builds two key dictionaries from the training data:
    1. user_businesses: mapping user -> {business: rating}
    2. business_users: mapping business -> {user: rating}
    
    These dictionaries provide fast lookup for ratings during prediction and
    similarity computation.
    
    Returns:
        Tuple: (user_businesses, business_users)
    """
    # ---------------------------
    # 1. Build user -> businesses dictionary
    # ---------------------------
    # For each record, create (user_id, (business_id, rating)) pairs
    # Group by user_id to get all businesses rated by each user
    # Convert grouped values to dictionaries for fast lookup
    user_businesses = (train_rdd.map(lambda x: (x[0], (x[1], x[2])))
                           .groupByKey()
                           .mapValues(lambda recs: dict(recs))
                           .collectAsMap())
    
    # ---------------------------
    # 2. Build business -> users dictionary
    # ---------------------------
    # Similar approach but in reverse: group by business_id
    business_users = (train_rdd.map(lambda x: (x[1], (x[0], x[2])))
                               .groupByKey()
                               .mapValues(lambda recs: dict(recs))
                               .collectAsMap())
    
    return user_businesses, business_users

# ===================================================================
# SIMILARITY CALCULATION FUNCTIONS
# ===================================================================

def compute_pearson_similarity(business_1, business_2, business_users, business_avg, similarity_cache):
    """
    Computes the Pearson similarity between two businesses.
    Uses a cache to avoid redundant computations.
    
    Returns:
        float: The Pearson similarity between the two businesses
    """
    # ---------------------------
    # 1. Check cache for existing calculation
    # ---------------------------
    # Create a consistent key for the cache by sorting business IDs
    # This ensures (A,B) and (B,A) use the same cache entry
    cache_key = tuple(sorted([business_1, business_2]))
    
    # Return cached value if already computed to save time
    if cache_key in similarity_cache:
        return similarity_cache[cache_key]
    
    # ---------------------------
    # 2. Find common users who rated both businesses
    # ---------------------------
    # Get the set of users who rated each business
    users1 = set(business_users[business_1].keys())
    users2 = set(business_users[business_2].keys())
    
    # Find the intersection (users who rated both businesses)
    common_users = users1.intersection(users2)
    
    # ---------------------------
    # 3. Handle special cases with few common users
    # ---------------------------
    # If <= 1 common user, use fallback similarity based on average rating difference
    if len(common_users) <= 1:
        similarity = (5.0 - abs(business_avg[business_1] - business_avg[business_2])) / 5.0
        similarity_cache[cache_key] = similarity
        return similarity
    
    # Optimization for exactly 2 common users: use average of rating differences
    if len(common_users) == 2:
        users_list = list(common_users)
        sim1 = (5.0 - abs(business_users[business_1][users_list[0]] - 
                         business_users[business_2][users_list[0]])) / 5.0
        sim2 = (5.0 - abs(business_users[business_1][users_list[1]] - 
                         business_users[business_2][users_list[1]])) / 5.0
        similarity = (sim1 + sim2) / 2
        similarity_cache[cache_key] = similarity
        return similarity
    
    # ---------------------------
    # 4. Calculate Pearson correlation for 3+ common users
    # ---------------------------
    # Extract ratings for common users
    ratings1 = [business_users[business_1][user] for user in common_users]
    ratings2 = [business_users[business_2][user] for user in common_users]
    
    # Calculate means for both sets of ratings
    mean1 = sum(ratings1) / len(ratings1)
    mean2 = sum(ratings2) / len(ratings2)
    
    # Center the ratings by subtracting means (creates zero-centered data)
    centered1 = [r - mean1 for r in ratings1]
    centered2 = [r - mean2 for r in ratings2]
    
    # Calculate numerator: sum of (x-mean_x)*(y-mean_y)
    numerator = sum(a * b for a, b in zip(centered1, centered2))
    
    # Calculate denominator: sqrt(sum((x-mean_x)²) * sum((y-mean_y)²))
    denominator = (sum(a ** 2 for a in centered1) ** 0.5) * (sum(b ** 2 for b in centered2) ** 0.5)
    
    # Handle division by zero case
    if denominator == 0:
        similarity = 0.0
    else:
        similarity = numerator / denominator
    
    # Cache the result for future use and return
    similarity_cache[cache_key] = similarity
    return similarity

# ===================================================================
# RATING PREDICTION FUNCTION
# ===================================================================

def predict_rating(user_id, business_id, user_businesses, business_users, 
                  business_avg, user_avg, global_avg, similarity_cache, neighbor_count=15):
    """
    Predicts the rating for a given (user, business) pair using item-based CF.
    
    Returns:
        float: Predicted rating (1.0-5.0)
    """
    # ---------------------------
    # 1. Check if user has already rated this business
    # ---------------------------
    # If so, return the actual rating (though this is unlikely in test data)
    if user_id in user_businesses and business_id in user_businesses[user_id]:
        return user_businesses[user_id][business_id]
    
    # ---------------------------
    # 2. Handle cold start problems (new user or new business)
    # ---------------------------
    # Case 2a: New user - return business average or global average
    if user_id not in user_businesses:
        return business_avg.get(business_id, global_avg)
    
    # Case 2b: New business - return user average or global average
    if business_id not in business_users:
        return user_avg.get(user_id, global_avg)
    
    # ---------------------------
    # 3. Use item-based CF with Pearson similarity
    # ---------------------------
    # Find similar businesses that the user has rated
    similar_items = []
    
    # For each business the user has rated, compute similarity with target business
    for other_business, rating in user_businesses[user_id].items():
        # Skip if same business or if other business not in training data
        if other_business == business_id or other_business not in business_users:
            continue
        
        # Compute similarity between the target and other business
        similarity = compute_pearson_similarity(
            business_id, other_business, business_users, business_avg, similarity_cache
        )
        
        # Only use positive similarities (more informative for prediction)
        if similarity > 0:
            similar_items.append((similarity, rating))
    
    # ---------------------------
    # 4. Fallback if no similar items found
    # ---------------------------
    if not similar_items:
        return user_avg.get(user_id, global_avg)
    
    # ---------------------------
    # 5. Calculate weighted average of ratings from similar items
    # ---------------------------
    # Sort by similarity (descending) and select top neighbors
    similar_items.sort(key=lambda x: -x[0])
    top_neighbors = similar_items[:neighbor_count]
    
    # Compute weighted average rating (sum of similarity * rating / sum of similarities)
    numerator = sum(sim * r for sim, r in top_neighbors)
    denominator = sum(abs(sim) for sim, r in top_neighbors)
    
    # Handle edge case of zero denominator
    if denominator == 0:
        return user_avg.get(user_id, global_avg)
    
    # Calculate final rating and ensure it's within valid range [1.0, 5.0]
    predicted_rating = numerator / denominator
    return min(max(predicted_rating, 1.0), 5.0)

# ===================================================================
# EVALUATION FUNCTIONS
# ===================================================================

def calculate_rmse(predictions, validation_rdd):
    """
    Calculates the Root Mean Square Error by comparing predictions against validation data.
    
    Args:
        predictions (list): List of (user_id, business_id, predicted_rating) tuples
        validation_rdd (RDD): RDD of validation data with (user_id, business_id, actual_rating)
        
    Returns:
        float: RMSE value
    """
    # ---------------------------
    # 1. Create fast lookup dictionary for predictions
    # ---------------------------
    # Convert predictions list to dictionary with (user_id, business_id) as key
    pred_dict = {(user_id, business_id): rating for user_id, business_id, rating in predictions}
    
    # ---------------------------
    # 2. Map validation data to (predicted, actual) pairs
    # ---------------------------
    def get_prediction(item):
        user_id, business_id, actual = item
        key = (user_id, business_id)
        if key in pred_dict:
            return (pred_dict[key], actual)
        return None
    
    # Apply the mapping function and filter out any missing predictions
    paired_ratings = validation_rdd.map(get_prediction).filter(lambda x: x is not None)
    
    # ---------------------------
    # 3. Calculate squared errors
    # ---------------------------
    # For each (predicted, actual) pair, compute the squared difference
    squared_errors = paired_ratings.map(lambda x: (x[0] - x[1]) ** 2)
    
    # ---------------------------
    # 4. Compute final RMSE value
    # ---------------------------
    # Count the number of pairs (needed for denominator)
    count = squared_errors.count()
    
    # Handle edge case of no valid predictions
    if count == 0:
        return 0.0
    
    # Sum all squared errors and calculate RMSE
    total_error = squared_errors.sum()
    return math.sqrt(total_error / count)

# ===================================================================
# MAIN FUNCTION
# ===================================================================

def main():
    """
    Main function for Task 2_1: Item-based Collaborative Filtering
    """
    # ---------------------------
    # 1. Setup and Initialization
    # ---------------------------
    # Start timing for performance measurement
    start_time = time.time()
    
    # Initialize Spark context with appropriate configuration
    conf = SparkConf().setAppName("Task2_1_ItemBasedCF")
    global sc
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")  # Reduce log verbosity
    
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2_1.py <train_file_name> <test_file_name> <output_file_name>")
        sys.exit(1)
    
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # ---------------------------
    # 2. Data Loading and Preprocessing
    # ---------------------------
    # Read training data and cache for multiple uses
    train_rdd = read_train_data(train_file)
    train_rdd.cache()  # Important optimization since we'll use this RDD multiple times
    
    # Compute rating averages at different levels (business, user, global)
    # These will be used for cold-start handling and similarity calculation
    business_avg, user_avg, global_avg = compute_averages(train_rdd)
    
    # Build dictionaries for fast rating lookups:
    # - user_businesses: map users to businesses they've rated
    # - business_users: map businesses to users who rated them
    user_businesses, business_users = build_rating_dicts(train_rdd)
    
    # ---------------------------
    # 3. Read Test Data (Prediction Targets)
    # ---------------------------
    # Read test data without ratings - just (user_id, business_id) pairs
    # We'll predict ratings for these pairs
    test_rdd = read_test_data(test_file, include_ratings=False)
    test_pairs = test_rdd.collect()  # Collect to driver for prediction
    
    # ---------------------------
    # 4. Similarity Cache Initialization
    # ---------------------------
    # Create an empty cache for storing computed similarities
    # This significantly speeds up predictions by avoiding redundant calculations
    similarity_cache = {}
    
    # ---------------------------
    # 5. Prediction Generation
    # ---------------------------
    # Make predictions for each user-business pair in the test set
    predictions = []
    for user_id, business_id in test_pairs:
        # Use item-based collaborative filtering to predict rating
        predicted_rating = predict_rating(
            user_id, business_id, user_businesses, business_users,
            business_avg, user_avg, global_avg, similarity_cache
        )
        predictions.append((user_id, business_id, predicted_rating))
    
    # ---------------------------
    # 6. Result Sorting and Output
    # ---------------------------
    # Sort results lexicographically as required by assignment
    predictions.sort(key=lambda x: (x[0], x[1]))
    
    # Write predictions to output file in CSV format
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "business_id", "prediction"])
        for row in predictions:
            writer.writerow(row)
    
    # ---------------------------
    # 7. Optional RMSE Calculation
    # ---------------------------
    # Try to calculate RMSE if validation data has ratings (for self-evaluation)
    try:
        # Read validation data with ratings (same file but including the rating column)
        validation_rdd = read_test_data(test_file, include_ratings=True)
        
        # Only proceed if we have validation data
        if validation_rdd.count() > 0:
            # Calculate and print RMSE
            rmse = calculate_rmse(predictions, validation_rdd)
            print(f"RMSE: {rmse}")
            
            # ---------------------------
            # 8. Error Distribution Analysis
            # ---------------------------
            # Create a dictionary mapping (user_id, business_id) to actual rating
            validation_dict = validation_rdd.map(lambda x: ((x[0], x[1]), x[2])).collectAsMap()
            
            # Initialize counters for different error ranges
            error_ranges = {">=0 and <1": 0, ">=1 and <2": 0, ">=2 and <3": 0, ">=3 and <4": 0, ">=4": 0}
            
            # Count number of predictions in each error range
            for user_id, business_id, pred_rating in predictions:
                key = (user_id, business_id)
                if key in validation_dict:
                    actual = validation_dict[key]
                    error = abs(pred_rating - actual)
                    
                    # Categorize error into appropriate range
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
            
            # Print error distribution
            print("Error Distribution:")
            for range_label, count in error_ranges.items():
                print(f"  {range_label}: {count}")
    except:
        # If RMSE calculation fails, just skip it
        # This keeps the output format intact for grading
        pass
    
    # ---------------------------
    # 9. Finalization
    # ---------------------------
    # Calculate and print execution time
    execution_time = time.time() - start_time
    print(f"Task 2.1 completed successfully in {execution_time:.2f} seconds.")
    
    # Clean up by stopping the Spark context
    sc.stop()

if __name__ == "__main__":
    main()