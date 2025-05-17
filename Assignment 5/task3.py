import sys
import random
import time
from blackbox import BlackBox

# -----------------------------------------------------------------------------
# Task 3: Reservoir Sampling for User Stream
# -----------------------------------------------------------------------------
# This script processes a stream of users in batches and maintains a fixed-size
# sample (reservoir) of 100 users.
#
# Key points:
# - First 100 users are directly added to the reservoir.
# - After that, each new user is accepted with probability 100/n.
# - If accepted, the new user replaces a random user in the reservoir.
# - After every 100 users, we snapshot and output 5 indices from the reservoir.
#
# Output format:
#   seqnum,0_id,20_id,40_id,60_id,80_id
# Where:
#   - seqnum is the count of users processed so far (100, 200, 300, ...)
#   - _id entries are the user IDs at those reservoir positions
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Parse command-line arguments
    # -------------------------------------------------------------------------
    input_file = sys.argv[1]       # Path to users.txt
    stream_size = int(sys.argv[2]) # Number of users to fetch per ask (typically 100)
    num_asks = int(sys.argv[3])    # How many times to ask the BlackBox
    output_file = sys.argv[4]      # Output file name for CSV results

    # -------------------------------------------------------------------------
    # 2. Initialize random seed ONCE (as required by the assignment)
    # -------------------------------------------------------------------------
    random.seed(553)

    # -------------------------------------------------------------------------
    # 3. Set up BlackBox simulator
    # -------------------------------------------------------------------------
    bx = BlackBox()

    # -------------------------------------------------------------------------
    # 4. Initialize the reservoir and counters
    # -------------------------------------------------------------------------
    reservoir = []      # This will hold at most 100 users
    user_count = 0      # Total number of users seen so far (global sequence number)

    # -------------------------------------------------------------------------
    # 5. Prepare output: header line and result container
    # -------------------------------------------------------------------------
    output_lines = []
    output_lines.append("seqnum,0_id,20_id,40_id,60_id,80_id")

    # -------------------------------------------------------------------------
    # 6. Start timing
    # -------------------------------------------------------------------------
    start_time = time.time()

    # -------------------------------------------------------------------------
    # 7. Process the stream in multiple asks (each gives stream_size users)
    # -------------------------------------------------------------------------
    for _ in range(num_asks):
        stream_users = bx.ask(input_file, stream_size)

        for user_id in stream_users:
            user_count += 1

            # --- Fill initial reservoir (first 100 users) ---
            if len(reservoir) < 100:
                reservoir.append(user_id)
            else:
                # --- Perform reservoir sampling (probabilistic replacement) ---
                if random.random() < 100 / user_count:
                    replace_index = random.randint(0, 99)
                    reservoir[replace_index] = user_id

            # --- Every 100 users, snapshot reservoir state ---
            if user_count % 100 == 0:
                snapshot = [
                    str(user_count),
                    reservoir[0],
                    reservoir[20],
                    reservoir[40],
                    reservoir[60],
                    reservoir[80]
                ]
                output_lines.append(",".join(snapshot))

    # -------------------------------------------------------------------------
    # 8. Write all output lines to file
    # -------------------------------------------------------------------------
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + "\n")

    # -------------------------------------------------------------------------
    # 9. Print total duration
    # -------------------------------------------------------------------------
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution completed in {duration:.2f} seconds.")
