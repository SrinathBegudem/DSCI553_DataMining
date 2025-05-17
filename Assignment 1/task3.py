import sys
import json
import time
from pyspark import SparkContext

def process_task3(review_path, business_path, output_path_a, output_path_b):
    """
    Computes average stars per city and compares sorting methods.
    """

    # Step 1: Initialize Spark Context
    sc = SparkContext("local[*]", "DSCI 553 Task 3")
    sc.setLogLevel("ERROR")

    try:
        # Step 2: Load datasets and parse JSON records
        review_rdd = sc.textFile(review_path).map(json.loads)
        business_rdd = sc.textFile(business_path).map(json.loads)

        # Step 3: Extract relevant fields
        review_stars_rdd = review_rdd.map(lambda x: (x["business_id"], x["stars"]))
        business_city_rdd = business_rdd.map(lambda x: (x["business_id"], x["city"]))

        # Step 4: Join datasets
        joined_rdd = review_stars_rdd.join(business_city_rdd)

        # Step 5: Compute average stars per city
        avg_stars_per_city = (
            joined_rdd.map(lambda x: (x[1][1], (x[1][0], 1)))  # (city, (stars, count))
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))  # Aggregate sums
            .mapValues(lambda x: round(x[0] / x[1], 2))  # Compute average
        )

        # Step 6: Sort by stars descending, then city name lexicographically
        sorted_avg_stars = avg_stars_per_city.sortBy(lambda x: (-x[1], x[0]))

        # Step 7: Save output for Task 3(A)
        with open(output_path_a, "w") as f:
            f.write("city,stars\n")
            for city, stars in sorted_avg_stars.collect():
                f.write(f"{city},{stars}\n")

        # Step 8: Compare sorting methods for top 10 cities

        # Method 1: Sort using Python
        start_time = time.time()
        sorted_avg_stars_py = sorted_avg_stars.take(100)  # Avoid collecting everything
        sorted_avg_stars_py = sorted(sorted_avg_stars_py, key=lambda x: (-x[1], x[0]))[:10]
        m1_time = round(time.time() - start_time, 5)

        # Method 2: Sort using Spark’s takeOrdered
        start_time = time.time()
        sorted_avg_stars_spark = sorted_avg_stars.takeOrdered(10, key=lambda x: (-x[1], x[0]))
        m2_time = round(time.time() - start_time, 5)

        # Step 9: Save output for Task 3(B)
        output_b = {
            "m1": m1_time,
            "m2": m2_time,
            "reason": "Spark sorting is optimized for distributed computation, but since the dataset is small, Python sorting performs similarly."
        }

        with open(output_path_b, "w") as f:
            json.dump(output_b, f, indent=4)

    finally:
        sc.stop()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: spark-submit task3.py <review_file> <business_file> <output_file_a> <output_file_b>")
        sys.exit(1)

    process_task3(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])






























# import sys
# import json
# import time
# from pyspark import SparkContext


# def process_task3(review_path, business_path, output_path_a, output_path_b):
#     """
#     Process Yelp reviews and business data to compute the average stars per city and compare sorting methods.
#     """
#     # Initialize SparkContext
#     sc = SparkContext("local[*]", "DSCI 553 Task 3")
#     sc.setLogLevel("ERROR")

#     try:
#         # Load datasets and convert to RDDs
#         review_rdd = sc.textFile(review_path).map(json.loads)
#         business_rdd = sc.textFile(business_path).map(json.loads)

#         # Extract relevant fields
#         review_stars_rdd = review_rdd.map(lambda x: (x["business_id"], x["stars"]))
#         business_city_rdd = business_rdd.map(lambda x: (x["business_id"], x["city"]))

#         # Join datasets on business_id
#         joined_rdd = review_stars_rdd.join(business_city_rdd)

#         # Compute average stars per city
#         avg_stars_per_city = (
#             joined_rdd.map(lambda x: (x[1][1], (x[1][0], 1)))  # (city, (stars, count))
#             .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))  # Aggregate sums
#             .mapValues(lambda x: round(x[0] / x[1], 2))  # Compute average with rounding
#         )

#         # Sort by stars descending, then city name lexicographically
#         sorted_avg_stars = avg_stars_per_city.sortBy(lambda x: (-x[1], x[0]))

#         # Save output for Task 3(A)
#         with open(output_path_a, "w") as f:
#             f.write("city,stars\n")
#             for city, stars in sorted_avg_stars.collect():
#                 f.write(f"{city},{stars}\n")

#         # Compare sorting methods for top 10 cities

#         # Method 1: Sort using Python
#         start_time = time.time()
#         sorted_avg_stars_py = sorted_avg_stars.collect()
#         sorted_avg_stars_py = sorted(sorted_avg_stars_py, key=lambda x: (-x[1], x[0]))[:10]
#         m1_time = round(time.time() - start_time, 5)

#         # Method 2: Sort using Spark's takeOrdered
#         start_time = time.time()
#         sorted_avg_stars_spark = sorted_avg_stars.takeOrdered(10, key=lambda x: (-x[1], x[0]))
#         m2_time = round(time.time() - start_time, 5)

#         # Save output for Task 3(B)
#         output_b = {
#             "m1": m1_time,
#             "m2": m2_time,
#             "reason": "Spark sorting is optimized for distributed computation, but in this case, the dataset is small, so Python sorting performs similarly."
#         }

#         with open(output_path_b, "w") as f:
#             json.dump(output_b, f, indent=4)

#         print(f"✅ Task 3 completed. Output saved to {output_path_a} and {output_path_b}")

#     except Exception as e:
#         print(f"❌ ERROR: {str(e)}")
#         sys.exit(1)

#     finally:
#         sc.stop()


# if __name__ == "__main__":
#     if len(sys.argv) != 5:
#         print("Usage: spark-submit task3.py <review_file> <business_file> <output_file_a> <output_file_b>")
#         sys.exit(1)

#     process_task3(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
