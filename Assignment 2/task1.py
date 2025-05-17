#!/usr/bin/env python3
import sys
import time
from itertools import combinations
from collections import defaultdict
from pyspark import SparkContext

###############################################################################
#                              HASH FUNCTION                                  #
###############################################################################

def xor_hash(a, b, num_buckets):
    """
    Convert 'a','b' to int if possible, else fallback to hash,
    then (x ^ y) % num_buckets.
    """
    try:
        aa = int(a)
        bb = int(b)
    except ValueError:
        aa = hash(a)
        bb = hash(b)
    return (aa ^ bb) % num_buckets

###############################################################################
#                      BUILD BASKETS FOR CASE1 / CASE2                        #
###############################################################################

def build_baskets_case1(rdd):
    """
    Case 1: (user -> set(businesses)).
    Return RDD of sets of business IDs.
    """
    return (rdd
            .map(lambda row: (row[0], row[1]))
            .groupByKey()
            .mapValues(set)
            .map(lambda x: x[1]))

def build_baskets_case2(rdd):
    """
    Case 2: (business -> set(users)).
    Return RDD of sets of user IDs.
    """
    return (rdd
            .map(lambda row: (row[1], row[0]))
            .groupByKey()
            .mapValues(set)
            .map(lambda x: x[1]))

###############################################################################
#                             LOCAL PCY PASS                                  #
###############################################################################

def local_pcy(part_baskets, total_count, global_support, num_buckets=1000):
    """
    SON local pass (PCY) with fraction-based local threshold:
      local_threshold = max(1, support * (len_partition/total_count)).

    Steps:
      - Count singletons; build pair-hash counts => build a bitmap
      - Identify freq singles, candidate pairs
      - Remove non-freq singles from baskets & count pairs exactly
      - Extend to k>=3 via naive combo approach.

    Returns [(itemset, 1), ...] representing local frequent sets.
    """
    baskets = list(part_baskets)
    part_size = len(baskets)
    if part_size == 0:
        return []

    # local threshold
    fraction = part_size / float(total_count)
    local_thresh_float = global_support * fraction
    if local_thresh_float < 1:
        local_thresh_float = 1.0
    local_threshold = local_thresh_float

    # Pass1: count singletons, hash pairs
    item_count = {}
    hash_count = {}

    for basket_set in baskets:
        basket_list = list(basket_set)

        # count each item
        for itm in basket_list:
            item_count[itm] = item_count.get(itm, 0) + 1

        # hash each pair
        for i in range(len(basket_list) - 1):
            for j in range(i + 1, len(basket_list)):
                idx = xor_hash(basket_list[i], basket_list[j], num_buckets)
                hash_count[idx] = hash_count.get(idx, 0) + 1

    # build a bit_map from hash_count
    bit_map = [0]*num_buckets
    for idx, val in hash_count.items():
        if val >= local_threshold:
            bit_map[idx] = 1

    # gather freq singletons
    freq_singles = []
    for it, cnt in item_count.items():
        if cnt >= local_threshold:
            freq_singles.append(it)

    local_candidates = []
    # add singletons as (tuple([s]), 1)
    for s in freq_singles:
        local_candidates.append((tuple([s]), 1))

    # build candidate pairs from freq singles
    pair_candidates = []
    for i in range(len(freq_singles)):
        for j in range(i+1, len(freq_singles)):
            pair_candidates.append((freq_singles[i], freq_singles[j]))

    # remove pairs whose hashed bucket isn't frequent
    good_pairs = []
    for pr in pair_candidates:
        h_idx = xor_hash(pr[0], pr[1], num_buckets)
        if bit_map[h_idx] == 1:
            good_pairs.append(pr)

    if not good_pairs:
        # no freq pairs => done
        return local_candidates

    # remove from baskets the non-freq singles (for exact counting of pairs+)
    freq_singles_set = set(freq_singles)
    updated_baskets = []
    for bset in baskets:
        # keep only freq singles
        keep_items = [x for x in bset if x in freq_singles_set]
        updated_baskets.append(keep_items)

    # exact count pairs
    pair_count = dict.fromkeys(good_pairs, 0)
    for b in updated_baskets:
        b_as_set = set(b)
        for p_tup in pair_count.keys():
            if p_tup[0] in b_as_set and p_tup[1] in b_as_set:
                pair_count[p_tup] += 1

    freq_pairs = []
    for pr, cval in pair_count.items():
        if cval >= local_threshold:
            freq_pairs.append(pr)
            local_candidates.append((tuple(sorted(pr)), 1))

    if not freq_pairs:
        return local_candidates

    # k>=3 combos
    current_sets = freq_pairs
    k = 3
    while True:
        # gather distinct items from freq (k-1)-sets
        all_items = set()
        for cset in current_sets:
            for x in cset:
                all_items.add(x)

        # build combos of size k
        cands_k = list(combinations(sorted(all_items), k))
        cand_count = dict.fromkeys(cands_k, 0)

        for b_list in updated_baskets:
            bset = set(b_list)
            for ck in cands_k:
                if set(ck).issubset(bset):
                    cand_count[ck] += 1

        freq_new = []
        for ccombo, ccount in cand_count.items():
            if ccount >= local_threshold:
                freq_new.append(ccombo)
                local_candidates.append((tuple(ccombo), 1))

        if not freq_new:
            break
        current_sets = freq_new
        k += 1

    return local_candidates

###############################################################################
#                              SON PASS 2                                     #
###############################################################################

def pass2_count(basket_iter, cand_list):
    """
    For each basket, check each candidate. If candidate is subset of basket, emit (cand,1).
    """
    out_map = defaultdict(int)
    for bset in basket_iter:
        b_as_set = set(bset)
        for ctuple in cand_list:
            if set(ctuple).issubset(b_as_set):
                out_map[ctuple] += 1
    return out_map.items()

###############################################################################
#                          OUTPUT FORMATTING                                  #
###############################################################################

def group_and_sort_itemsets(itemsets):
    """
    Group itemsets by size -> sort each group in lexicographical order 
    (treat each element as string).
    
    Returns a list of (size, [list_of_tuples_sorted]) in ascending size order.
    """
    from collections import defaultdict
    group_map = defaultdict(list)
    for it in itemsets:
        group_map[len(it)].append(it)

    # Build a sorted list of (size, [itemsets]).
    results = []
    for sz in sorted(group_map.keys()):
        # Sort itemsets of size=sz lexicographically as strings
        group_map[sz].sort(key=lambda tup: tuple(str(x) for x in tup))
        results.append((sz, group_map[sz]))
    return results

def format_itemset(itup):
    """
    Convert a tuple like ('2', '6') into "('2', '6')" 
    or a single tuple ('2',) into "('2')".
    Each item is wrapped in single quotes and a space is added after each comma.
    """
    return "(" + ", ".join("'" + str(x) + "'" for x in itup) + ")"

def write_results(cand_grouped, freq_grouped, out_file):
    """
    Print all candidates and frequent itemsets in the required grouped format:

    Candidates:
    ('i1', 'i2'),('i3', 'i4')...

    ('i5')...

    Frequent Itemsets:
    ('i1', 'i2'),('i3', 'i4')...

    etc.

    Each size's itemsets are on ONE line, separated by commas, then a blank line.
    """
    with open(out_file, "w") as f:
        # ---------------------
        # Candidates
        # ---------------------
        f.write("Candidates:\n")
        for (sz, cands) in cand_grouped:
            # put all cands of this size on ONE line, separated by commas
            line_str = ",".join(format_itemset(c) for c in cands)
            f.write(line_str + "\n\n")  # blank line after

        # ---------------------
        # Frequent Itemsets
        # ---------------------
        f.write("Frequent Itemsets:\n")
        for (sz, frs) in freq_grouped:
            line_str = ",".join(format_itemset(fset) for fset in frs)
            f.write(line_str + "\n\n")  # blank line after each size group

###############################################################################
#                                   MAIN                                      #
###############################################################################

def main():
    if len(sys.argv) != 5:
        print("Usage: spark-submit task1.py <case_number> <support> <input_file> <output_file>")
        sys.exit(-1)

    case_num = int(sys.argv[1])
    support = float(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    sc = SparkContext(appName="task1_updated_code")
    sc.setLogLevel("ERROR")

    start_t = time.time()

    lines = sc.textFile(input_file)
    header = lines.first()  # handle a potential header row
    data_rdd = lines.filter(lambda x: x != header).map(lambda x: x.strip().split(","))

    # Build baskets depending on case #
    if case_num == 1:
        baskets_rdd = build_baskets_case1(data_rdd)
    else:
        baskets_rdd = build_baskets_case2(data_rdd)

    total_count = baskets_rdd.count()

    # ------------------
    # SON Phase 1
    # ------------------
    local_candidates = (
        baskets_rdd
        .mapPartitions(lambda part: local_pcy(part, total_count, support, 1000))
        .reduceByKey(lambda a, b: a + b)
        .map(lambda x: x[0])
        .distinct()
        .collect()
    )

    # group candidates by size and sort them
    cand_grouped = group_and_sort_itemsets(local_candidates)

    # ------------------
    # SON Phase 2
    # ------------------
    cand_list = list(local_candidates)  # broadcast to pass2
    freq_counts = (
        baskets_rdd
        .mapPartitions(lambda p: pass2_count(p, cand_list))
        .reduceByKey(lambda x, y: x + y)
        .filter(lambda x: x[1] >= support)
        .map(lambda x: x[0])
        .collect()
    )

    freq_grouped = group_and_sort_itemsets(freq_counts)

    # ------------------
    # Write out results
    # ------------------
    write_results(cand_grouped, freq_grouped, output_file)

    end_t = time.time()
    print("Duration:", end_t - start_t)

    sc.stop()

if __name__ == "__main__":
    main()
