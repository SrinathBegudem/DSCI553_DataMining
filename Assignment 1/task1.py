import sys
import json
from pyspark import SparkContext

def process_reviews(input_path, output_path):
    """
    Process Yelp reviews dataset to compute statistical insights.
    """

    # Step 1: Initialize Spark Context
    sc = SparkContext("local[*]", "DSCI 553 Task 1")
    sc.setLogLevel("ERROR")  # Suppress unnecessary log messages

    try:
        # Step 2: Load dataset and parse JSON records
        review_rdd = sc.textFile(input_path).map(json.loads).cache()

        # Step 3: Compute required statistics

        # (A) Total number of reviews
        total_reviews = review_rdd.count()

        # (B) Number of reviews written in the year 2018
        reviews_2018 = review_rdd.filter(lambda x: x["date"].startswith("2018")).count()

        # (C) Number of distinct users who wrote reviews
        distinct_users = review_rdd.map(lambda x: x["user_id"]).distinct().count()

        # (D) Top 10 users with the highest number of reviews
        top_users = (review_rdd.map(lambda x: (x["user_id"], 1))
                     .reduceByKey(lambda a, b: a + b)
                     .takeOrdered(10, key=lambda x: (-x[1], x[0])))

        # (E) Number of distinct businesses that received reviews
        distinct_businesses = review_rdd.map(lambda x: x["business_id"]).distinct().count()

        # (F) Top 10 businesses with the highest number of reviews
        top_businesses = (review_rdd.map(lambda x: (x["business_id"], 1))
                          .reduceByKey(lambda a, b: a + b)
                          .takeOrdered(10, key=lambda x: (-x[1], x[0])))

        # Step 4: Prepare the output dictionary
        output_data = {
            "n_review": total_reviews,
            "n_review_2018": reviews_2018,
            "n_user": distinct_users,
            "top10_user": [[user_id, count] for user_id, count in top_users],
            "n_business": distinct_businesses,
            "top10_business": [[business_id, count] for business_id, count in top_businesses]
        }

        # Step 5: Write the output to a JSON file
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

    finally:
        sc.stop()  # Step 6: Stop Spark Context

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: spark-submit task1.py <input_filepath> <output_filepath>")
        sys.exit(1)

    process_reviews(sys.argv[1], sys.argv[2])






















# import sys
# import json
# from datetime import datetime
# from pyspark import SparkContext

# def process_reviews(input_path, output_path):
#     """
#     Processes the Yelp review dataset using Spark RDDs to compute various statistics.

#     Parameters:
#     input_path (str): Path to the input JSON file containing review data.
#     output_path (str): Path to save the output JSON file with computed results.

#     Output:
#     A JSON file containing:
#     - Total number of reviews
#     - Number of reviews in 2018
#     - Number of distinct users
#     - Top 10 users with the most reviews
#     - Number of distinct businesses
#     - Top 10 businesses with the most reviews
#     """

#     # Initialize Spark Context
#     sc = SparkContext("local[*]", "DSCI 553 Task 1")
#     sc.setLogLevel("ERROR")  # Suppress unnecessary log messages for cleaner output

#     try:
#         # Step 1: Load the input file into an RDD
#         review_rdd = sc.textFile(input_path)

#         # Edge Case: If the input file is empty, exit the program
#         if review_rdd.isEmpty():
#             print("❌ ERROR: Input file is empty!")
#             sys.exit(1)

#         # Parse JSON records from text format to dictionary format
#         review_rdd = review_rdd.map(json.loads).cache()

#         # Step 2: Compute required statistics

#         # (A) Total number of reviews
#         total_reviews = review_rdd.count()

#         # (B) Number of reviews written in the year 2018
#         reviews_2018 = review_rdd.filter(
#             lambda x: x.get("date", "").startswith("2018")
#         ).count()

#         # (C) Number of unique users who wrote reviews
#         distinct_users = review_rdd.map(lambda x: x.get("user_id", "")).distinct().count()

#         # (D) Top 10 users with the highest number of reviews (sorted by count, then user_id)
#         top_users = (review_rdd.map(lambda x: (x["user_id"], 1))
#                      .reduceByKey(lambda a, b: a + b)
#                      .takeOrdered(10, key=lambda x: (-x[1], x[0])))

#         # (E) Number of distinct businesses that received reviews
#         distinct_businesses = review_rdd.map(lambda x: x.get("business_id", "")).distinct().count()

#         # (F) Top 10 businesses with the highest number of reviews (sorted by count, then business_id)
#         top_businesses = (review_rdd.map(lambda x: (x["business_id"], 1))
#                           .reduceByKey(lambda a, b: a + b)
#                           .takeOrdered(10, key=lambda x: (-x[1], x[0])))

#         # Step 3: Prepare the output dictionary in the required format
#         output_data = {
#             "n_review": total_reviews,
#             "n_review_2018": reviews_2018,
#             "n_user": distinct_users,
#             "top10_user": [[user_id, count] for user_id, count in top_users],  # Convert tuples to lists
#             "n_business": distinct_businesses,
#             "top10_business": [[business_id, count] for business_id, count in top_businesses]  # Convert tuples to lists
#         }

#         # Step 4: Write the output to a JSON file
#         with open(output_path, "w") as f:
#             json.dump(output_data, f, indent=4)

#         print(f"✅ Output successfully written to {output_path}")

#     except Exception as e:
#         # Handle unexpected errors gracefully
#         print(f"❌ ERROR: {str(e)}")
#         sys.exit(1)

#     finally:
#         # Step 5: Stop the SparkContext to free up resources
#         sc.stop()

# # Entry point: Ensure correct command-line arguments are provided
# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: spark-submit task1.py <input_filepath> <output_filepath>")
#         sys.exit(1)

#     review_filepath = sys.argv[1]
#     output_filepath = sys.argv[2]
#     process_reviews(review_filepath, output_filepath)
