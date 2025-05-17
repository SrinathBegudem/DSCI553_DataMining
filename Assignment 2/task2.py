#!/usr/bin/env python3
import sys
import time
import random
import copy
from itertools import combinations
from collections import defaultdict
from pyspark import SparkContext

##############################################################################
#                             HELPER FUNCTIONS                               #
##############################################################################

def custom_hash(x, y, n_buckets):
    """
    Compute a hash value for a pair of items.
    This function first tries to convert x and y to integers. If that fails,
    it falls back to Python's built-in hash. Then, it computes (int(x) ^ int(y)) modulo n_buckets.
    """
    try:
        xx = int(x)
    except ValueError:
        xx = hash(x)
    try:
        yy = int(y)
    except ValueError:
        yy = hash(y)
    return (xx ^ yy) % n_buckets

def partition_function(x):
    """
    A simple partitioning function that uses the length of the key (x[0])
    plus a random integer to assign a partition.
    """
    from random import randint
    return (len(x[0]) + randint(0,1000))

def local_candidate_generation(partition_baskets, total_count, global_support, num_buckets):
    """
    This function performs a local pass on a partition to generate candidate itemsets.
    It works as follows:
      1. Calculate the local threshold as: (number of baskets in this partition / total baskets) * global_support.
      2. Count the frequency of each individual item in the baskets.
      3. For every basket, hash every pair of items and count their occurrences.
      4. Build a bitmap based on which hash buckets meet the local threshold.
      5. Identify frequent single items and consider them as candidate itemsets.
      6. Generate candidate pairs from these frequent single items and filter them using the bitmap.
      7. If there are candidate pairs, prune the baskets to only include frequent singles and count the exact frequency of these pairs.
      8. Extend the candidate generation to larger itemsets (triples, quadruples, etc.) using combinations,
         and count their frequency exactly.
      9. Return all candidate itemsets found in this partition.
    """
    baskets = list(partition_baskets)
    part_size = len(baskets)
    if part_size == 0:
        return []

    p = float(part_size) / float(total_count)
    local_thresh = p * float(global_support)
    if local_thresh < 1:
        local_thresh = 1
    t = local_thresh

    # PASS 1: Count individual items and count pairs using a hash function.
    item_cnt = defaultdict(int)
    hash_cnt = defaultdict(int)
    for basket in baskets:
        for itm in basket:
            item_cnt[itm] += 1
        for i in range(len(basket)-1):
            for j in range(i+1, len(basket)):
                idx = custom_hash(basket[i], basket[j], num_buckets)
                hash_cnt[idx] += 1

    # Create a bitmap: mark a bucket as frequent if its count reaches the local threshold.
    bitmap = [0] * num_buckets
    for hkey, val in hash_cnt.items():
        if val >= t:
            bitmap[hkey] = 1

    # Identify frequent single items (appear in at least t baskets).
    single_freq = []
    local_candidates = []
    for k, v in item_cnt.items():
        if v >= t:
            single_freq.append(k)
            local_candidates.append((tuple([k]), 1))

    # Generate candidate pairs from the frequent single items.
    pair_freq = []
    for i in range(len(single_freq)-1):
        for j in range(i+1, len(single_freq)):
            pair_freq.append((single_freq[i], single_freq[j]))

    # Filter candidate pairs using the bitmap.
    filtered_pairs = []
    for pr in pair_freq:
        hidx = custom_hash(pr[0], pr[1], num_buckets)
        if bitmap[hidx] == 1:
            filtered_pairs.append(pr)
    pair_freq = filtered_pairs

    if len(pair_freq) == 0:
        return local_candidates

    # PASS 2: Prune baskets to keep only frequent singles and count candidate pairs exactly.
    freq_singles_set = set(single_freq)
    pruned_baskets = []
    for b in baskets:
        newb = [x for x in b if x in freq_singles_set]
        pruned_baskets.append(newb)

    pair_counts = dict.fromkeys(pair_freq, 0)
    for b in pruned_baskets:
        bset = set(b)
        for pair_ in pair_counts.keys():
            if pair_[0] in bset and pair_[1] in bset:
                pair_counts[pair_] += 1

    freq_pairs = []
    for pr, cval in pair_counts.items():
        if cval >= t:
            freq_pairs.append(pr)
            local_candidates.append((tuple(sorted(pr, key=str)), 1))

    if not freq_pairs:
        return local_candidates

    # Extend candidate generation to larger itemsets (k >= 3)
    curr_freq = freq_pairs
    k = 3
    while True:
        allitems = set()
        for fs in curr_freq:
            for xx in fs:
                allitems.add(xx)
        allitems = sorted(list(allitems), key=str)
        cand_k = list(combinations(allitems, k))
        if len(cand_k) == 0:
            break
        cand_counts = dict.fromkeys(cand_k, 0)
        for b in pruned_baskets:
            bset = set(b)
            for ck in cand_k:
                if set(ck).issubset(bset):
                    cand_counts[ck] += 1
        new_freqs = []
        for ck_, cval_ in cand_counts.items():
            if cval_ >= t:
                new_freqs.append(ck_)
                local_candidates.append((ck_, 1))
        if not new_freqs:
            break
        curr_freq = new_freqs
        k += 1

    return local_candidates

def SON_count_stage(baskets_iter, candidate_itemsets):
    """
    For each basket, count how many times each candidate itemset appears.
    This function is used in the global counting stage of the SON algorithm.
    """
    baskets = list(baskets_iter)
    out_map = defaultdict(int)
    for b in baskets:
        bset = set(b)
        for c in candidate_itemsets:
            if set(c).issubset(bset):
                out_map[tuple(c)] += 1
    return out_map.items()

def group_and_format_for_output(items_list):
    """
    Group itemsets by their size, sort them lexicographically, and create a string where:
      - Each itemset is formatted as ('item1', 'item2', ...) with each item in single quotes.
      - All itemsets of the same size appear on one line, separated by commas (with no trailing comma).
      - Different size groups are separated by an empty line.
    """
    unique_itemsets = sorted(set(tuple(x) for x in items_list), key=lambda x: (len(x), tuple(str(i) for i in x)))
    groups = {}
    for itemset in unique_itemsets:
        groups.setdefault(len(itemset), []).append(itemset)
    result_lines = []
    for size in sorted(groups.keys()):
        group_sorted = sorted(groups[size], key=lambda x: tuple(str(i) for i in x))
        formatted = ["(" + ", ".join("'" + str(x) + "'" for x in itemset) + ")" for itemset in group_sorted]
        line = ",".join(formatted)
        result_lines.append(line)
    return "\n\n".join(result_lines)

##############################################################################
#                                  MAIN                                      #
##############################################################################

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: spark-submit task2.py <filter_threshold> <support> <input_file> <output_file>")
        sys.exit(1)

    filter_threshold = int(sys.argv[1])
    support = float(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    start_t = time.time()
    sc = SparkContext(appName="task2_snr_style")
    sc.setLogLevel("ERROR")

    # 1) Read the input file
    lines = sc.textFile(input_file)
    header = lines.first()
    data_rdd = lines.filter(lambda x: x != header).map(lambda x: x.split(','))

    # 2) Pre-process the data to generate (date-customer, product) pairs.
    #    For example, for a raw row, create a new customer key as "date-customerID".
    def transform(row):
        date_customer = row[0][1:-5] + row[0][-3:-1] + '-' + str(int(row[1][1:-1]))
        product_id = str(int(row[5][1:-1]))
        return (date_customer, product_id)
    processed_rdd = data_rdd.map(transform)

    # 3) Build the market-basket model by grouping by date-customer and filtering baskets with more than the threshold items.
    def group_prod(rdd, k):
        return (rdd.map(lambda row: (row[0], row[1]))
                  .groupByKey()
                  .mapValues(set)
                  .map(lambda x: (x[0], list(x[1])))
                  .filter(lambda x: len(x[1]) > k))
    baskets_rdd = group_prod(processed_rdd, filter_threshold)
    items_rdd = baskets_rdd.values()

    # 4) Count the total number of baskets.
    total_baskets = items_rdd.count()

    # 5) Partition the data (using a simple partitioning function).
    n_partitions = 10
    def f_partition(x):
        return (len(x[0]) + random.randint(0,1000)) % n_partitions
    keyed_baskets = baskets_rdd.partitionBy(n_partitions, partition_function)
    baskets = keyed_baskets.values()

    # 6) SON Stage 1: Run the local candidate generation on each partition.
    candidates_stage1 = (
        baskets
        .mapPartitions(lambda part: local_candidate_generation(
            partition_baskets=part,
            total_count=total_baskets,
            global_support=support,
            num_buckets=1000
        ))
        .reduceByKey(lambda x, y: x + y)
        .collect()
    )
    local_cand_itemsets = [sorted(x[0]) for x in candidates_stage1]
    local_cand_itemsets.sort()
    if local_cand_itemsets:
        unique_list = [local_cand_itemsets[-1]]
        for i in range(len(local_cand_itemsets)-2, -1, -1):
            if local_cand_itemsets[i] != unique_list[-1]:
                unique_list.append(local_cand_itemsets[i])
        local_cand_itemsets = list(reversed(unique_list))

    # 7) Format the candidate itemsets for output.
    candidates_str = group_and_format_for_output(local_cand_itemsets)

    # 8) SON Stage 2: Globally count candidate itemsets and filter by the support threshold.
    freq_map = (
        baskets
        .mapPartitions(lambda part: SON_count_stage(part, local_cand_itemsets))
        .reduceByKey(lambda x, y: x + y)
        .filter(lambda x: x[1] >= int(support))
        .collect()
    )
    freq_itemsets = [sorted(x[0]) for x in freq_map]
    freq_itemsets.sort()
    if freq_itemsets:
        unique_freq = [freq_itemsets[-1]]
        for i in range(len(freq_itemsets)-2, -1, -1):
            if freq_itemsets[i] != unique_freq[-1]:
                unique_freq.append(freq_itemsets[i])
        freq_itemsets = list(reversed(unique_freq))
    frequent_str = group_and_format_for_output(freq_itemsets)

    # 9) Write the final output file with both Candidates and Frequent Itemsets sections.
    out_str = "Candidates:\n" + candidates_str + "\n\nFrequent Itemsets:\n" + frequent_str
    with open(output_file, "w") as f:
        f.write(out_str)

    end_t = time.time()
    print("Duration:", end_t - start_t)

    sc.stop()
