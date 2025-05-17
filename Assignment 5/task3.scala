import java.io._
import scala.collection.mutable.MutableList
import scala.io.Source

/**
 * DSCI553 - Foundations and Applications of Data Mining
 * Assignment 5: Streaming Algorithms Implementation
 * Task 3: Reservoir Sampling
 * 
 * This implementation demonstrates the reservoir sampling technique, which is used
 * to maintain a fixed-size random sample from a potentially infinite data stream.
 * 
 * The key insight of reservoir sampling is that we initially fill our reservoir,
 * and then probabilistically replace elements to maintain a uniform random sample
 * as more data arrives. The probability of keeping a new element decreases as we
 * process more of the stream, which ensures statistical fairness.
 */
object task3 {
  /**
   * The StreamSource class simulates a continuous data stream by retrieving
   * random elements from a source file.
   * 
   * In real-world applications, streams come from live data sources like APIs,
   * logs, or user interactions, but for this assignment we're simulating a stream
   * by randomly selecting elements from a static source.
   */
  class StreamSource {
    // We must initialize this random generator with exactly this seed value
    // to ensure reproducible results across different runs and implementations.
    // The assignment specifically requires this exact seed value (553).
    private val streamRandom = scala.util.Random
    streamRandom.setSeed(553)
    
    /**
     * Retrieves a batch of random user IDs from the specified data source.
     * 
     * In streaming systems, we typically process data in batches rather than
     * one-by-one for efficiency. This method simulates receiving a batch of data
     * from our stream source.
     * 
     * @param sourceFile  Path to the file containing user IDs
     * @param countNeeded Number of user IDs to retrieve in this batch
     * @return An array of randomly selected user IDs
     */
    def retrieveUserBatch(sourceFile: String, countNeeded: Int): Array[String] = {
      // First, load all available user IDs from the source file
      // In a real streaming system, we wouldn't have access to all data at once
      val sourcePath = sourceFile
      val userPool = Source.fromFile(sourcePath).getLines().toArray
      
      // Create a container to hold our selected users for this batch
      var userBatch = new Array[String](countNeeded)
      
      // Select random users to simulate a stream batch
      // Note: We select with replacement to properly simulate a stream
      // where the same user could appear multiple times
      for (i <- 0 to countNeeded - 1) {
        // The randomized selection is critical for stream simulation
        // We must use the exact random seed specified to ensure the sequence
        // of selections matches the expected pattern
        val selectedIndex = streamRandom.nextInt(userPool.length)
        userBatch(i) = userPool(selectedIndex)
      }
      
      // Return the selected batch of users
      return userBatch
    }
  }
  
  /**
   * Main execution method for the reservoir sampling algorithm.
   * 
   * The key steps of reservoir sampling are:
   * 1. Fill the reservoir with the first k elements (k=100 in our case)
   * 2. For each subsequent element n:
   *    - Keep it with probability k/n
   *    - If kept, replace a random element in the reservoir
   * 
   * This approach ensures that at any point, each element seen so far
   * has exactly the same probability of being in the reservoir.
   */
  def main(args: Array[String]): Unit = {
    // ======================================================================
    // SETUP AND INITIALIZATION
    // ======================================================================
    
    // Initialize the random number generator for sampling decisions
    // This must be separate from the stream generation randomness
    // And must use the exact seed specified (553) for reproducible results
    val reservoirRandom = scala.util.Random
    reservoirRandom.setSeed(553)
    
    // Parse command line arguments 
    val dataSourcePath = args(0)       // Path to input file (users.txt)
    val elementsPerBatch = args(1).toInt  // Batch size for streaming
    val totalBatches = args(2).toInt   // Number of batches to process
    val resultsPath = args(3)          // Where to save output CSV
    
    // Start timing execution for performance measurement
    val executionStart = System.nanoTime
    
    // Initialize our stream data source
    val dataStream = new StreamSource
    
    // ======================================================================
    // RESERVOIR INITIALIZATION
    // ======================================================================
    
    // This is our reservoir - a fixed-size buffer that will hold our sample
    // The MutableList type is used as it allows dynamic addition and efficient
    // indexed access, both of which we need for reservoir sampling
    var reservoir: MutableList[String] = MutableList()
    
    // Prepare the output with the required CSV header
    // The format matches exactly what's specified in the assignment:
    // sequence number followed by user IDs at specific reservoir positions
    var outputContent = "seqnum,0_id,20_id,40_id,60_id,80_id\n"
    
    // This counter tracks the total number of elements we've seen so far
    // We use Float type for consistent division precision with the reference
    var elementCounter: Float = 0
    
    // ======================================================================
    // MAIN PROCESSING LOOP - APPLYING RESERVOIR SAMPLING
    // ======================================================================
    
    // For each batch from our simulated stream
    for (batchIndex <- 0 to totalBatches - 1) {
      // Get the next batch of users from our stream
      val currentUsers = dataStream.retrieveUserBatch(dataSourcePath, elementsPerBatch)
      
      // Process each user in the current batch
      for (userID <- currentUsers) {
        // Keep track of total processed elements
        elementCounter += 1
        
        // CASE 1: Reservoir isn't full yet - direct insertion
        // For the first 100 elements, we simply fill the reservoir
        if (reservoir.length < 100) {
          reservoir += userID
        }
        // CASE 2: Reservoir is full - apply probabilistic replacement
        else {
          // The key to reservoir sampling: probability of keeping an element
          // decreases as we process more data, maintaining uniform distribution
          // We keep element with probability 100/n, where n is total seen so far
          val keepProbability = 100 / elementCounter
          
          // Generate random value between 0 and 1 for probabilistic decision
          val randomValue = reservoirRandom.nextFloat()
          
          // If random value falls below our threshold, keep this element
          if (randomValue < keepProbability) {
            // Select a random element in the reservoir to replace
            val replacementIndex = reservoirRandom.nextInt(100)
            
            // Replace that element with our current user
            reservoir(replacementIndex) = userID
          }
          // If random value is above threshold, discard this element
        }
        
        // ======================================================================
        // OUTPUT SNAPSHOT GENERATION
        // ======================================================================
        
        // Every 100 elements, take a snapshot of the reservoir state
        // This gives us a picture of how the sample evolves over time
        if (elementCounter % 100 == 0) {
          // Build the output line with sequence number and values at specific indices
          // Format matches assignment requirements: seqnum,0_id,20_id,40_id,60_id,80_id
          outputContent = outputContent + elementCounter.toInt.toString + "," + 
                         reservoir(0) + "," + 
                         reservoir(20) + "," + 
                         reservoir(40) + "," + 
                         reservoir(60) + "," + 
                         reservoir(80) + "\n"
        }
      }
    }
    
    // ======================================================================
    // RESULTS OUTPUT
    // ======================================================================
    
    // Write the collected results to the output file
    val outputFile = new File(resultsPath)
    val writer = new BufferedWriter(new FileWriter(outputFile))
    writer.write(outputContent)
    writer.close()
    
    // Calculate and report total execution time
    val executionEnd = System.nanoTime
    val totalSeconds = (executionEnd - executionStart) / 1e9d
    println(f"Execution completed in $totalSeconds%.2f seconds.")
  }
}