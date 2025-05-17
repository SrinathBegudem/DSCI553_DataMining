"""
=================================================================================
    DSCI-553: Assignment 4 (Spring 2025)
    Task 1: Community Detection using GraphFrames & Label Propagation Algorithm
---------------------------------------------------------------------------------
    OVERVIEW:
    1) We read the CSV input containing user_id,business_id (with a header).
    2) For each user, gather the set of businesses they reviewed.
    3) Construct an undirected graph where:
        - Each distinct user is a node.
        - An edge between two users exists only if they have >= filter_threshold
          common businesses reviewed.
    4) Use GraphFrames + labelPropagation(maxIter=5) to detect communities.
    5) Sort the discovered communities by:
        (a) The ascending size of each community (smaller sets first).
        (b) The lexicographically smallest user in that community.
        Then, within a community, sort the user IDs in ascending order.
    6) Output each community on a new line. Each line's format:
        'userA', 'userB', 'userC', ...
---------------------------------------------------------------------------------
    EXECUTION EXAMPLE:
        spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 \
                     task1.py 7 /path/to/ub_sample_data.csv /path/to/output.txt
---------------------------------------------------------------------------------
    NOTE:
     - We rely on GraphFrames 0.8.2 for Spark 3.1.2.
     - This code is purely for Task 1. Do NOT use it for Task 2, as Task 2
       requires strictly Spark RDD (no DataFrames, no GraphFrames).
=================================================================================
"""

import sys
import time
import os
from itertools import combinations

# -------------- SPARK IMPORTS -------------- #
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from graphframes import GraphFrame
# ------------------------------------------- #
# ------------------------------------------------------------------------------
# We specify the GraphFrames package, so Spark knows where to fetch it. 
# (You can also provide this via 'spark-submit --packages ...')
# ------------------------------------------------------------------------------
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"

def main():
    """
    Main entry point for Task 1.
    1) Parse command-line args
    2) Read data, build user-business mapping
    3) Construct graph using threshold
    4) Run LPA to detect communities
    5) Output final results
    6) Print duration
    """
    # -------------------------
    # 1. Parse command-line arguments
    # -------------------------
    # Expecting 3 arguments in total:
    #   1) filter_threshold  (integer)
    #   2) input_file_path   (str)
    #   3) output_file_path  (str)
    # Example: spark-submit ... task1.py 7 input.csv output.txt
    if len(sys.argv) != 4:
        print("Usage: task1.py <filter_threshold> <input_file_path> <output_file_path>")
        sys.exit(-1)

    filter_threshold = int(sys.argv[1])
    input_file_path  = sys.argv[2]
    output_file_path = sys.argv[3]

    # Start measuring runtime
    start_time = time.time()

    # -------------------------------------
    # 2. Initialize Spark Context & Session
    # -------------------------------------
    # We create a SparkContext for RDD operations,
    # and a SparkSession (or SQLContext) for DataFrame operations.
    sc = SparkContext(appName="task1")
    sc.setLogLevel("ERROR")  # reduce Spark logging in console
    spark = SparkSession.builder \
        .config("spark.some.config.option", "some-value") \
        .getOrCreate()
    sqlContext = SQLContext(sc)

    # -------------------------------------
    # 3. Load and preprocess the input CSV
    # -------------------------------------
    # We'll read lines from the CSV, skip the header, then parse each row into (user_id, business_id).
    # Example of row: "user_id,business_id"
    # We do an initial textFile read, remove the header, then split by comma.
    input_rdd = sc.textFile(input_file_path)
    header = input_rdd.first()
    data_rdd = input_rdd.filter(lambda line: line != header)\
                        .map(lambda line: line.strip().split(","))  # => (user_id, business_id)

    # -------------------------------------
    # 4. Build dictionary: user -> set of businesses
    # -------------------------------------
    # We want something like: user_business_map[user] = {business1, business2, ...}
    # Steps:
    #   - map each row -> (user_id, business_id)
    #   - groupByKey => user_id => list of business_ids
    #   - convert the list to a set for easy intersection
    #
    # We collectAsMap() to bring it to the driver. This can be big, but we’ll assume
    # dataset is small enough for assignment. In a real system, we might do a different approach.
    user_business_map = data_rdd \
        .map(lambda x: (x[0], x[1])) \
        .groupByKey() \
        .mapValues(lambda biz_list: set(biz_list)) \
        .collectAsMap()

    # We'll also get a distinct list of all users.
    distinct_users = data_rdd.map(lambda x: x[0]).distinct().collect()

    # -------------------------------------
    # 5. Identify all edges based on threshold
    # -------------------------------------
    # We want an edge between userA and userB if they share at least
    # 'filter_threshold' number of businesses. 
    # We'll do a combination of all distinct users two at a time
    # because the PDF says each edge is undirected, unweighted.
    #
    #    for pair in combinations(distinct_users, 2):
    #       if overlap >= filter_threshold => create an edge in both directions
    #
    # We'll store edges in a list of tuples (src, dst).
    # Then we will create a DataFrame from that list for the GraphFrame.
    from itertools import combinations
    edges_list = []
    # We'll store a set of nodes that appear, so we only keep relevant users in final graph
    nodes_set = set()

    for userA, userB in combinations(distinct_users, 2):
        # Compute intersection size
        businessesA = user_business_map[userA]
        businessesB = user_business_map[userB]
        common_count = len(businessesA.intersection(businessesB))

        if common_count >= filter_threshold:
            # Add these two users to our nodes set
            nodes_set.add(userA)
            nodes_set.add(userB)
            # Because GraphFrames require directed edges for construction,
            # we add both (userA -> userB) and (userB -> userA).
            edges_list.append((userA, userB))
            edges_list.append((userB, userA))

    # Now convert nodes_set into a list so we can build a DataFrame.
    nodes_list = list(nodes_set)

    # -------------------------------------
    # 6. Create DataFrames for GraphFrames
    # -------------------------------------
    # GraphFrame requires:
    #   - A vertices DataFrame with a column "id" 
    #   - An edges DataFrame with columns "src" and "dst"
    #
    # We'll create them from the Python lists:
    #    nodes_list -> Spark DataFrame ["id"]
    #    edges_list -> Spark DataFrame ["src", "dst"]
    nodes_df = sqlContext.createDataFrame([(n,) for n in nodes_list], ["id"])
    edges_df = sqlContext.createDataFrame(edges_list, ["src", "dst"])

    # -------------------------------------
    # 7. Construct the GraphFrame and run LPA
    # -------------------------------------
    # Label Propagation Algorithm:
    #   - We use maxIter=5 (as required by the PDF).
    #   - The result is a DataFrame of columns: ["id", "label"]
    #     The "label" indicates the community ID.
    GF = GraphFrame(nodes_df, edges_df)
    lpa_result_df = GF.labelPropagation(maxIter=5)

    # -------------------------------------
    # 8. Group the LPA results into communities
    # -------------------------------------
    # The LPA result has "id" (user) and "label" (long).
    # We want to group all users that share the same label, 
    # then produce a list of user IDs in each community.
    #
    # Steps:
    #   - convert the DF to an RDD
    #   - map each row => (label, id)
    #   - groupByKey => label => all users with that label
    #   - sort each community's users lexicographically
    #   - collect all communities as a list of user-lists
    #   - sort them by the rules in the PDF:
    #        1) ascending length
    #        2) if two communities have the same length, compare by
    #           the lexicographically smallest user in that community.
    #   - write them out
    # 
    # CAREFUL with multiple sorts. We'll do a single sort with a tuple:
    #   (len(community), community[0])
    # where community[0] is the first user in lexicographic order.
    lpa_rdd = lpa_result_df.rdd.map(lambda row: (row["label"], row["id"]))
    # groupByKey => label => [user1, user2, ...]
    label_groups = lpa_rdd.groupByKey().mapValues(list)  # => (label, [users...])
    
    # For each (label, [users...]), sort the user IDs, then transform to just that sorted list
    # We'll ignore the label after we produce the sorted user list.
    communities_rdd = label_groups.map(lambda x: sorted(x[1]))

    # Next, we want to sort these communities by:
    #   1) ascending length
    #   2) ascending first user 
    # Because the list is already sorted inside, the first user is at index [0].
    # So the final sort key is (len(community), community[0]).
    sorted_communities = communities_rdd.sortBy(lambda community: (len(community), community[0]))

    # -------------------------------------
    # 9. Write the output file
    # -------------------------------------
    # The PDF says each community line should be of the form:
    #   'uid1', 'uid2', 'uid3', ...
    # So we will format them carefully with quotes and commas,
    # ensuring that within a community, the user IDs are also lex sorted 
    # (which we did above).
    final_communities = sorted_communities.collect()
    
    with open(output_file_path, 'w') as out_file:
        for community in final_communities:
            # community is like ['userA', 'userB', 'userC'] 
            # We want the line to read: 'userA', 'userB', 'userC'
            # We'll do a join with "', '" as the separator, 
            # then add single quotes around the entire thing properly.
            line = ", ".join(f"'{u}'" for u in community)
            out_file.write(line + "\n")

    # -------------------------------------
    # 10. Print duration
    # -------------------------------------
    end_time = time.time()
    duration = end_time - start_time
    print("Duration:", duration)

    # Stop the SparkContext just to be clean.
    sc.stop()

# ------------------------------------------------------------------------------
# Standard Python convention: if we're running this file directly,
# call main(). If we import it as a module, it won't run immediately.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
