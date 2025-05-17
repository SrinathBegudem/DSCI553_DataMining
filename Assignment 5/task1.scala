import java.io._
import scala.io.Source
import scala.util.Random
import math.BigInt
// Import Spark classes for configuration and context creation.
import org.apache.spark.{SparkConf, SparkContext}

object task1 {

  /**
    * BlackBox class to simulate reading random lines ("users") from the input file.
    * Matches the behavior of blackbox.py in your Python solution.
    * We seed the Random with 553 for reproducibility.
    */
  class BlackBox {
    private val rand = new Random()
    rand.setSeed(553)  // Ensures same user sampling order across runs

    /**
      * ask(filename, num) randomly picks `num` lines (users) from `filename` 
      * and returns them as an Array[String].
      */
    def ask(filename: String, num: Int): Array[String] = {
      val allLines = Source.fromFile(filename).getLines().toArray
      val stream   = new Array[String](num)

      // Randomly select `num` lines from allLines
      for(i <- 0 until num) {
        val index = rand.nextInt(allLines.length)  // random index
        stream(i) = allLines(index)
      }
      stream
    }
  }

  // --------------------------------------------------------------------------
  // Global constants for the Bloom Filter (matching your Python code)
  // --------------------------------------------------------------------------
  val FILTER_SIZE = 69997            // Size of the Bloom Filter bit array
  val HASH_COUNT  = 5                // Number of hash functions
  val BIG_PRIME   = 16769023         // Large prime for hashing
  val A_PARAMS    = Array(12, 37, 51, 73, 91)   // "a" coefficients
  val B_PARAMS    = Array(7, 31, 57, 85, 101)   // "b" coefficients

  // Bloom filter bit array (initialized to 0)
  val bloomFilter = Array.fill[Int](FILTER_SIZE)(0)
  // Set to keep track of actual users seen so far for false positive checking
  var previouslySeen = Set[String]()

  /**
    * myhashs(user): Converts a user_id string into HASH_COUNT different indices.
    * Uses a stable set of parameters for reproducibility.
    *
    * @param user The user_id string to be hashed.
    * @return     An Array[Int] of indices, each in the range [0, FILTER_SIZE-1].
    */
  def myhashs(user: String): Array[Int] = {
    // Convert user_id string to a positive BigInt (mimics Python's binascii conversion)
    val userBigInt = BigInt(1, user.getBytes("UTF-8"))
    val hashIndices = new Array[Int](HASH_COUNT)
    for (i <- 0 until HASH_COUNT) {
      // Compute hash value using (a * x + b) mod BIG_PRIME, then take modulo FILTER_SIZE
      val hashVal = (A_PARAMS(i) * userBigInt + B_PARAMS(i)) mod BIG_PRIME
      val index   = (hashVal.toLong % FILTER_SIZE).toInt
      hashIndices(i) = index
    }
    hashIndices
  }

  /**
    * Main entry point.
    * Usage:
    * spark-submit --class task1 --executor-memory 4G --driver-memory 4G hw5.jar <inputFile> <streamSize> <numRequests> <outputFile>
    */
  def main(args: Array[String]): Unit = {
    // Parse command-line arguments:
    val inputFile   = args(0)        // e.g., "users.txt"
    val streamSize  = args(1).toInt  // e.g., 100
    val numRequests = args(2).toInt  // e.g., 30
    val outputFile  = args(3)        // e.g., "output_task1.csv"

    // --- Set up Spark Context with memory specifications and log level ---
    val conf = new SparkConf()
      .setAppName("Task1")
      .setMaster("local[*]")
      .set("spark.executor.memory", "4g")
      .set("spark.driver.memory", "4g")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")  // Only display ERROR-level logs

    // Create the BlackBox instance for generating the user stream.
    val bx = new BlackBox()

    // Start timing the execution (for performance info)
    val startTime = System.nanoTime()

    // Prepare the CSV output with header "Time,FPR"
    val resultBuilder = new StringBuilder
    resultBuilder.append("Time,FPR\n")

    // Process each batch of users from the BlackBox.
    for (t <- 0 until numRequests) {
      // Retrieve a batch of users.
      val streamUsers = bx.ask(inputFile, streamSize)
      var falsePositives = 0

      // Process each user in the batch.
      for (userId <- streamUsers) {
        val indices = myhashs(userId)
        // Check if the user might have been seen (i.e., all corresponding bits are set)
        var mightBeSeen = true
        for (idx <- indices if mightBeSeen) {
          if (bloomFilter(idx) == 0) {
            mightBeSeen = false
          }
        }
        // If user "might be seen" but is not in previouslySeen, count as false positive.
        if (mightBeSeen && !previouslySeen.contains(userId)) {
          falsePositives += 1
        }
        // If not all bits are set, update the Bloom filter bits.
        if (!mightBeSeen) {
          for (idx <- indices) {
            bloomFilter(idx) = 1
          }
        }
        // Always record the user as seen.
        previouslySeen += userId
      }

      // Calculate the false positive rate for this batch.
      val batchFPR = if (streamUsers.nonEmpty) falsePositives.toDouble / streamUsers.length else 0.0
      // Append the result in the form "Time,FPR" to the CSV output.
      resultBuilder.append(s"$t,$batchFPR\n")
    }

    // Write the resulting CSV data to the output file.
    val writer = new BufferedWriter(new FileWriter(outputFile))
    writer.write(resultBuilder.toString)
    writer.close()

    // Measure and print the total execution time.
    val endTime = System.nanoTime()
    val totalSeconds = (endTime - startTime).toDouble / 1e9
    println(f"Execution completed in $totalSeconds%.2f seconds.")

    // Stop the SparkContext
    sc.stop()
  }
}
