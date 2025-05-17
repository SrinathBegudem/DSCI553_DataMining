import sys
import json
import time
from pyspark import SparkContext

def process_partition(iterator):
    """
    Processes a partition and counts occurrences of business_id.
    """
    count_map = {}
    for business_id, _ in iterator:
        count_map[business_id] = count_map.get(business_id, 0) + 1
    return iter(count_map.items())

def analyze_partitioning(input_path, output_path, num_partitions):
    """
    Analyze the effects of default and custom partitioning on business reviews data.
    """

    # Step 1: Initialize Spark Context
    sc = SparkContext("local[*]", "DSCI 553 Task 2")
    sc.setLogLevel("ERROR")

    try:
        # Step 2: Load dataset and parse JSON records
        review_rdd = sc.textFile(input_path).map(json.loads)

        # Step 3: Default partitioning analysis
        start_time = time.time()
        default_top10 = (review_rdd.map(lambda x: (x["business_id"], 1))
                         .reduceByKey(lambda a, b: a + b)
                         .takeOrdered(10, key=lambda x: (-x[1], x[0])))
        default_time = time.time() - start_time

        default_partitions = review_rdd.getNumPartitions()
        default_partition_sizes = review_rdd.glom().map(len).collect()

        # Step 4: Custom partitioning analysis
        start_time = time.time()
        partitioned_rdd = (review_rdd.map(lambda x: (x["business_id"], 1))
                           .partitionBy(num_partitions, lambda key: hash(key) % num_partitions)  # Fix applied
                           .mapPartitions(process_partition)
                           .cache())

        custom_top10 = (partitioned_rdd.reduceByKey(lambda a, b: a + b)
                        .takeOrdered(10, key=lambda x: (-x[1], x[0])))
        custom_time = time.time() - start_time

        custom_partitions = partitioned_rdd.getNumPartitions()
        custom_partition_sizes = partitioned_rdd.glom().map(len).collect()

        # Step 5: Save output to JSON file
        output_data = {
            "default": {"n_partition": default_partitions, "n_items": default_partition_sizes, "exe_time": default_time},
            "customized": {"n_partition": custom_partitions, "n_items": custom_partition_sizes, "exe_time": custom_time}
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

    finally:
        sc.stop()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit task2.py <input_filepath> <output_filepath> <n_partition>")
        sys.exit(1)

    analyze_partitioning(sys.argv[1], sys.argv[2], int(sys.argv[3]))

























# import sys
# import json
# import time
# from pyspark import SparkContext


# def partition_function(key):
#     """
#     Custom partition function using hash-based partitioning.
#     This helps distribute data more evenly across partitions.
#     """
#     return hash(key) % num_partitions


# def process_partition(iterator):
#     """
#     Helper function to count occurrences of business_id within a partition.
#     """
#     count_map = {}
#     for business_id, _ in iterator:
#         count_map[business_id] = count_map.get(business_id, 0) + 1
#     return iter(count_map.items())


# def analyze_partitioning(input_path, output_path, num_partitions):
#     """
#     Function to analyze default and custom partitioning on business reviews data.
#     Measures execution time and data distribution across partitions.
#     """
#     sc = SparkContext("local[*]", "DSCI 553 Task 2")
#     sc.setLogLevel("ERROR")

#     try:
#         # Load input data
#         review_rdd = sc.textFile(input_path).map(json.loads)

#         # Default partitioning analysis
#         start_time = time.time()
#         top10_default = (review_rdd.map(lambda x: (x["business_id"], 1))
#                          .reduceByKey(lambda a, b: a + b)
#                          .takeOrdered(10, key=lambda x: (-x[1], x[0])))
#         default_time = time.time() - start_time

#         default_partitions = review_rdd.getNumPartitions()
#         default_partition_sizes = review_rdd.glom().map(len).collect()

#         # Custom partitioning analysis
#         start_time = time.time()
#         partitioned_rdd = (review_rdd.map(lambda x: (x["business_id"], 1))
#                            .partitionBy(num_partitions, partition_function)
#                            .mapPartitions(process_partition))

#         top10_custom = (partitioned_rdd.reduceByKey(lambda a, b: a + b)
#                         .takeOrdered(10, key=lambda x: (-x[1], x[0])))
#         custom_time = time.time() - start_time

#         custom_partitions = partitioned_rdd.getNumPartitions()
#         custom_partition_sizes = partitioned_rdd.glom().map(len).collect()

#         # Prepare output in the required format
#         output_data = {
#             "default": {
#                 "n_partition": default_partitions,
#                 "n_items": default_partition_sizes,
#                 "exe_time": default_time
#             },
#             "customized": {
#                 "n_partition": custom_partitions,
#                 "n_items": custom_partition_sizes,
#                 "exe_time": custom_time
#             }
#         }

#         # Write output to JSON file
#         with open(output_path, "w") as f:
#             json.dump(output_data, f, indent=4)

#         print(f"✅ Output successfully written to {output_path}")

#     except Exception as e:
#         print(f"❌ ERROR: {str(e)}")
#         sys.exit(1)
    
#     finally:
#         sc.stop()


# if __name__ == "__main__":
#     if len(sys.argv) != 4:
#         print("Usage: spark-submit task2.py <input_filepath> <output_filepath> <n_partition>")
#         sys.exit(1)

#     review_filepath = sys.argv[1]
#     output_filepath = sys.argv[2]
#     num_partitions = int(sys.argv[3])

#     analyze_partitioning(review_filepath, output_filepath, num_partitions)
