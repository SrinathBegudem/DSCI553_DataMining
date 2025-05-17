import sys
import time
import random
from itertools import combinations
from collections import defaultdict
from pyspark import SparkContext, SparkConf

# --------------------------
# Helper Functions
# --------------------------

def read_input_data(input_file):
    """
    Reads the input CSV file using Spark RDDs for distributed processing.
    
    Steps:
    1. Load the file as an RDD using Spark's textFile method.
    2. Extract the header (the first line) and skip it.
    3. Split each row by commas to separate the fields.
    4. Create (business_id, user_id) pairs.
    5. Group the user IDs by business ID, resulting in each business mapped to a set of users.
    6. Also, collect these mappings as a dictionary for fast lookup later.
    
    Args:
        input_file (str): The path to the CSV file.
    
    Returns:
        Tuple: 
          - An RDD of (business_id, set(user_ids)) for distributed operations.
          - A dictionary with the same mapping for quick lookups.
    """
    # Load the dataset into an RDD
    lines = sc.textFile(input_file)
    
    # Get the header line (first row) so we can remove it from the data
    header = lines.first()  
    
    # Remove the header and split each row into a list of values
    data = lines.filter(lambda row: row != header).map(lambda row: row.split(","))
    
    # Create pairs of (business_id, user_id) and group users by business_id
    business_users_rdd = data.map(lambda row: (row[1], row[0])).groupByKey().mapValues(set)
    
    # Collect the business-user mapping into a dictionary for fast lookups during similarity computation
    business_users_dict = business_users_rdd.collectAsMap()
    
    return business_users_rdd, business_users_dict


def generate_hash_functions(n, max_value, prime=16777619):
    """
    Generates 'n' hash functions with unique random coefficients.
    
    These hash functions are of the form:
        f(x) = ((a * x + b) % prime) % max_value
    where 'prime' is a prime number chosen for efficient modulo operations, and
    'max_value' is typically the number of unique users (used as bins).
    
    We use random.sample to ensure that the coefficients 'a' and 'b' are unique.
    
    Args:
        n (int): Number of hash functions to generate.
        max_value (int): Upper bound for coefficient selection (number of bins).
        prime (int): A prime number for the modulo operation (default is 16777619).
    
    Returns:
        Tuple: (a, b, prime)
          - 'a': A list of randomly selected coefficients for the hash functions.
          - 'b': A list of randomly selected coefficients for the hash functions.
          - 'prime': The prime number used.
    """
    random.seed(42)  # Set seed for reproducibility
    a = random.sample(range(1, max_value), n)  # Unique random coefficients for 'a'
    b = random.sample(range(1, max_value), n)  # Unique random coefficients for 'b'
    return a, b, prime


def compute_minhash_signatures(business_users_rdd, hash_funcs, user_dict, num_hashes, prime, max_value):
    """
    Computes the MinHash signature for each business.
    
    For each business, we compute a signature vector where each element is the minimum hash 
    value over all users who have reviewed that business for a specific hash function.
    
    This results in a compact signature that approximates the business's set of users.
    
    Args:
        business_users_rdd (RDD): RDD with (business_id, set(user_ids)).
        hash_funcs (tuple): Contains the lists 'a', 'b', and the 'prime' used in hash functions.
        user_dict (dict): Mapping of user IDs to unique integer indices.
        num_hashes (int): Total number of hash functions.
        prime (int): Prime number used for hashing.
        max_value (int): Number of bins (typically the number of unique users).
    
    Returns:
        RDD: Contains (business_id, signature) where signature is a list of integers.
    """
    # Unpack the hash function parameters
    a, b, prime = hash_funcs
    
    def minhash(user_set):
        """
        For a given set of users (who reviewed a business), compute the minhash signature.
        
        For each hash function i, compute the hash value for each user in the set and take the minimum.
        """
        return [
            min(((a[i] * user_dict[user] + b[i]) % prime) % max_value for user in user_set)
            for i in range(num_hashes)
        ]
    
    # Apply the minhash function on each business to create its signature vector
    return business_users_rdd.map(lambda x: (x[0], minhash(x[1])))


def apply_lsh(signature_rdd, num_bands, rows_per_band):
    """
    Applies Locality Sensitive Hashing (LSH) on the MinHash signature matrix.
    
    Steps:
    1. For each business signature, divide it into 'num_bands' bands.
    2. For each band, form a key using the band index and the tuple of hash values in that band.
    3. Group businesses that share the same band key.
    4. For each group (bucket) with more than one business, generate all possible business pairs.
    
    Args:
        signature_rdd (RDD): RDD with (business_id, signature).
        num_bands (int): Number of bands to split the signature.
        rows_per_band (int): Number of rows (hash values) per band.
    
    Returns:
        RDD: Contains candidate business pairs (each as a tuple) that have the same band signature.
    """
    def banding(business):
        """
        Split a business's signature into bands and create keys.
        
        Each key is a tuple (band_index, band_signature) that helps group businesses with similar signatures.
        """
        business_id, signature = business
        return [
            ((i, tuple(signature[i * rows_per_band: (i + 1) * rows_per_band])), business_id)
            for i in range(num_bands)
        ]
    
    # Apply the banding function, group by band keys, and generate candidate pairs for businesses in the same bucket.
    return (signature_rdd.flatMap(banding)
                         .groupByKey()
                         .mapValues(list)
                         .filter(lambda x: len(x[1]) > 1)
                         .flatMap(lambda x: combinations(sorted(x[1]), 2))
                         .distinct())


def calculate_jaccard_similarity(candidate_pairs_rdd, business_users_dict):
    """
    Computes the Jaccard similarity for each candidate pair.
    
    For each candidate pair, retrieve the sets of users for both businesses from the dictionary,
    and compute the Jaccard similarity as the size of the intersection divided by the size of the union.
    Only pairs with a similarity of 0.5 or higher are returned.
    
    Args:
        candidate_pairs_rdd (RDD): RDD containing candidate business pairs.
        business_users_dict (dict): Dictionary mapping business_id to set of user_ids.
    
    Returns:
        RDD: Contains tuples (business1, business2, similarity) for pairs meeting the threshold.
    """
    def jaccard(pair):
        b1, b2 = pair
        # Retrieve the set of users for each business
        users1, users2 = business_users_dict[b1], business_users_dict[b2]
        # Calculate the Jaccard similarity
        similarity = len(users1 & users2) / len(users1 | users2)
        # Only return the pair if the similarity is at least 0.5
        return (b1, b2, similarity) if similarity >= 0.5 else None
    
    # Map over candidate pairs and filter out those that don't meet the similarity threshold
    return candidate_pairs_rdd.map(jaccard).filter(lambda x: x is not None)


def write_output_file(output_file, results_rdd):
    """
    Writes the final candidate business pairs with their Jaccard similarity scores to a CSV file.
    
    The output CSV file will have a header "business_id_1,business_id_2,similarity".
    Each pair is sorted lexicographically and the entire file is sorted.
    
    Args:
        output_file (str): Path to the output CSV file.
        results_rdd (RDD): RDD containing (business_id_1, business_id_2, similarity) tuples.
    """
    # Collect the results from the RDD to the driver (assuming the candidate set is small)
    results = results_rdd.collect()
    # Sort the results lexicographically
    results.sort()
    # Write the results to a CSV file with the required header
    with open(output_file, 'w') as f:
        f.write("business_id_1,business_id_2,similarity\n")
        for b1, b2, sim in results:
            f.write(f"{b1},{b2},{sim}\n")


def main():
    """
    Main function to execute Task 1.
    
    Steps:
    1. Initialize the Spark context and set the log level to ERROR.
    2. Read the input file and prepare the data (business to user mappings).
    3. Create a dictionary that maps each user to a unique integer index.
    4. Set the parameters for MinHashing and LSH (number of hash functions, bands, rows per band, etc.).
    5. Generate hash function parameters.
    6. Compute the MinHash signature for each business.
    7. Apply LSH to generate candidate pairs of similar businesses.
    8. Compute the exact Jaccard similarity for candidate pairs and filter those with similarity >= 0.5.
    9. Write the results to a CSV file with the required output format.
    10. Stop the Spark context.
    """
    start_time = time.time()  # Start timing the execution
    
    # Initialize Spark context with appropriate configuration
    conf = SparkConf().setAppName("Task1_LSH")
    global sc
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")  # Reduce log verbosity
    
    # Read input and output file paths from command-line arguments
    input_file, output_file = sys.argv[1], sys.argv[2]
    
    # Load and preprocess the data: get both the RDD and a lookup dictionary of business-user mappings
    business_users_rdd, business_users_dict = read_input_data(input_file)
    
    # Generate a dictionary mapping each unique user to a unique integer index (for hashing)
    user_dict = business_users_rdd.flatMap(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    
    # Define parameters for MinHash and LSH
    num_hashes = 50         # Total number of hash functions to generate
    num_bands = 25          # Divide the signature into 25 bands
    rows_per_band = num_hashes // num_bands  # Number of rows (hash values) per band (should be > 1)
    max_value = len(user_dict)  # Use the total number of unique users as the modulus 'm'
    
    # Generate the hash function coefficients and use the prime 16777619 for optimized modulo operations
    hash_funcs = generate_hash_functions(num_hashes, max_value)
    
    # Compute the MinHash signatures for each business using the generated hash functions
    signature_rdd = compute_minhash_signatures(
        business_users_rdd, hash_funcs, user_dict, num_hashes, hash_funcs[2], max_value
    )
    
    # Apply LSH to the signature RDD to generate candidate pairs of businesses that might be similar
    candidate_pairs_rdd = apply_lsh(signature_rdd, num_bands, rows_per_band)
    
    # Calculate the Jaccard similarity for each candidate pair and filter out those below 0.5
    results_rdd = calculate_jaccard_similarity(candidate_pairs_rdd, business_users_dict)
    
    # Write the final, sorted candidate pairs with their similarity scores to the output CSV file
    write_output_file(output_file, results_rdd)
    
    print("Task 1 completed successfully in", time.time() - start_time, "seconds.")
    sc.stop()  # Stop the Spark context to free up resources

    
if __name__ == "__main__":
    main()
