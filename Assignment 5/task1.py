import sys
import csv
import binascii
from blackbox import BlackBox
import time

# -----------------------------------------------------------------------------
# GLOBAL CONSTANTS - Bloom Filter Configuration
# -----------------------------------------------------------------------------
FILTER_SIZE = 69997      # The specified length of the Bloom Filter bit array
HASH_COUNT = 5           # Number of distinct hash functions

# Large prime number for hashing (can choose any large prime)
# We'll use it in the formula: ((a * x + b) % BIG_PRIME) % FILTER_SIZE
BIG_PRIME = 16769023

# "a" and "b" values for each hash function (must remain constant each run)
# These are chosen arbitrarily, but kept stable to ensure consistent hashing.
A_PARAMS = [12, 37, 51, 73, 91]
B_PARAMS = [7, 31, 57, 85, 101]

# -----------------------------------------------------------------------------
# GLOBAL DATA STRUCTURES
# -----------------------------------------------------------------------------
# 1) The Bloom filter bit array, initially all zeros.
bloom_filter = [0] * FILTER_SIZE

# 2) Keep track of actual users seen so far, to detect false positives.
previous_users = set()

# -----------------------------------------------------------------------------
# FUNCTION: myhashs(s)
# -----------------------------------------------------------------------------
def myhashs(s):
    """
    Convert a user_id string 's' into HASH_COUNT different hash values,
    using stable hash parameters (A_PARAMS, B_PARAMS, BIG_PRIME).
    
    Args:
        s (str): The user_id string to be hashed.

    Returns:
        list of int: Each int is an index into the bloom_filter array.
                     e.g., [h1, h2, h3, h4, h5]
    """
    user_int = int(binascii.hexlify(s.encode('utf8')), 16)
    hash_values = []
    
    for i in range(HASH_COUNT):
        hash_val = (A_PARAMS[i] * user_int + B_PARAMS[i]) % BIG_PRIME
        hash_index = hash_val % FILTER_SIZE
        hash_values.append(hash_index)
    
    return hash_values

# -----------------------------------------------------------------------------
# MAIN EXECUTION BLOCK
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Read Command-Line Arguments
    input_file = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_asks = int(sys.argv[3])
    output_file = sys.argv[4]

    # 2) Create BlackBox instance and start timer
    bx = BlackBox()
    start_time = time.time()

    # 3) Prepare CSV output rows - The assignment wants "Time,FPR"
    output_data = [["Time", "FPR"]]

    # 4) For each batch from the data stream
    for t in range(num_asks):
        
        stream_users = bx.ask(input_file, stream_size)
        false_positives = 0
        
        # Process each user in the batch
        for user_id in stream_users:
            indices = myhashs(user_id)
            
            # Check if all bits at those indices are set -> might_be_seen
            might_be_seen = True
            for idx in indices:
                if bloom_filter[idx] == 0:
                    might_be_seen = False
                    break
            
            # If we "might be seen" but user not actually in previous_users,
            # that's a false positive
            if might_be_seen and (user_id not in previous_users):
                false_positives += 1
            
            # Otherwise, if not all bits set, set them now
            if not might_be_seen:
                for idx in indices:
                    bloom_filter[idx] = 1
            
            # Always add user to the actual set of seen users
            previous_users.add(user_id)
        
        # Calculate FPR
        total_users = len(stream_users)
        fpr = (false_positives / total_users) if total_users > 0 else 0

        # Add row to output data
        output_data.append([t, fpr])

        # Print debug info for this batch
        # print(f"Batch {t}: False Positives = {false_positives}, FPR = {fpr:.5f}")

    # 5) Write results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output_data)

    # 6) End timer and print total execution time
    end_time = time.time()
    print(f"Execution completed in {end_time - start_time:.2f} seconds.")
