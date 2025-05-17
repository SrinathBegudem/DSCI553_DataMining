import sys
import csv
import binascii
import time   # For measuring execution time
from blackbox import BlackBox

# -------------------------------------------------------------------------
# FLAJOLET-MARTIN CONFIGURATION
# -------------------------------------------------------------------------

NUM_HASH_FUNCTIONS = 16    # total number of hash functions
GROUP_SIZE = 4             # number of groups (each group has NUM_HASH_FUNCTIONS / GROUP_SIZE hash funcs)
BIG_PRIME = 16769023       # a large prime for hashing

# Fixed "a" and "b" parameters for each hash function
# Must remain constant across runs to ensure stable hashing
A_PARAMS = [
    13, 49, 17, 31, 53, 71, 101, 113,
    127, 139, 199, 241, 251, 269, 281, 307
]
B_PARAMS = [
    7, 11, 23, 29, 47, 59, 83, 97,
    109, 131, 151, 191, 223, 227, 239, 257
]

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------

def count_trailing_zeros(num):
    """
    Count how many trailing 0 bits exist in 'num' (in binary).
    Example: num=12 (binary 1100) has 2 trailing zeros.
    For num=0, define trailing zeros as 0 by convention.
    """
    if num == 0:
        return 0
    tz = 0
    # While the least significant bit is 0, shift right
    while (num & 1) == 0:
        tz += 1
        num >>= 1
    return tz

def myhashs(user_id):
    """
    Given a user_id (string), produce NUM_HASH_FUNCTIONS hashed values
    using the stable formula:
         hash_val = (A[i] * user_int + B[i]) % BIG_PRIME

    We keep 'A_PARAMS' and 'B_PARAMS' as constants so the grader can reproduce results.
    """
    # Convert string to int using binascii (assignment's recommended approach)
    user_int = int(binascii.hexlify(user_id.encode('utf8')), 16)

    hash_values = []
    for i in range(NUM_HASH_FUNCTIONS):
        hv = (A_PARAMS[i] * user_int + B_PARAMS[i]) % BIG_PRIME
        hash_values.append(hv)
    return hash_values

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Command line args:
    # python task2.py <input_file> <stream_size> <num_asks> <output_file>
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    # Create BlackBox instance to generate user streams
    bx = BlackBox()

    # Prepare list of rows for CSV: header first
    output_rows = [["Time", "Ground Truth", "Estimation"]]

    # We'll track sums to compute ratio at the end
    total_ground_truth = 0
    total_estimation = 0

    # Start timing
    start_time = time.time()

    # Process each batch of users
    for t in range(num_asks):
        # Get 'stream_size' users from the blackbox
        stream_users = bx.ask(input_file, stream_size)

        # The ground truth is just how many unique users appear in this batch
        unique_users = set(stream_users)
        batch_gt = len(unique_users)
        total_ground_truth += batch_gt

        # We track the maximum trailing zero count for each hash function
        max_trailing_zeros = [0] * NUM_HASH_FUNCTIONS

        # Update max trailing zeros for each user
        for uid in stream_users:
            hash_vals = myhashs(uid)
            for i, hv in enumerate(hash_vals):
                tz = count_trailing_zeros(hv)
                # If we found more trailing zeros than previously known, update it
                if tz > max_trailing_zeros[i]:
                    max_trailing_zeros[i] = tz

        # Combine the 16 hash function estimates into 4 groups,
        # each group having 4 hash functions (16/4).
        # The final Flajolet-Martin estimate is the median of these group averages.
        group_estimates = []
        hash_per_group = NUM_HASH_FUNCTIONS // GROUP_SIZE
        for g in range(GROUP_SIZE):
            start_idx = g * hash_per_group
            end_idx = start_idx + hash_per_group
            # For each hash function in this group, the estimate = 2^(max trailing zeros)
            estimates_in_group = [
                2 ** max_trailing_zeros[idx] for idx in range(start_idx, end_idx)
            ]
            # We then average these estimates for this group
            group_avg = sum(estimates_in_group) / len(estimates_in_group)
            group_estimates.append(group_avg)

        # Sort these group averages and take the median
        group_estimates.sort()
        # For 4 values, median is the average of the middle two
        mid_val = (group_estimates[1] + group_estimates[2]) / 2
        fm_estimation = mid_val

        # Add to running sum for ratio calculation
        total_estimation += fm_estimation

        # Store the integer approximation in the CSV (or keep float if desired)
        output_rows.append([t, batch_gt, int(fm_estimation)])

    # Write to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)

    # End timing
    end_time = time.time()
    duration = end_time - start_time

    # Compute ratio of (sum of all estimates / sum of all ground truths)
    ratio = 0
    if total_ground_truth > 0:
        ratio = total_estimation / total_ground_truth

    # Print results for debugging/analysis
    print(f"Execution completed in {duration:.2f} seconds.")
    print(f"Sum of FM Estimates / Sum of Ground Truth = {ratio:.5f}")
