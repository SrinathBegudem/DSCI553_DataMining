import java.io._
import scala.io.Source
import scala.util.Random
import math.BigInt

// Spark imports if you want to run within a Spark context
import org.apache.spark.{SparkConf, SparkContext}

object task2 {

  /**
    * BlackBox class simulates a streaming data source.
    * It randomly selects `num` lines from the provided file for each ask().
    * We seed the Random with 553 for consistent reproducibility, just like Python's blackbox.py.
    */
  class BlackBox {
    private val rand = new Random()
    rand.setSeed(553)

    /**
      * ask(filename, num): return an Array[String] of `num` random lines.
      *
      * @param filename  The file (e.g. "users.txt") from which to read user IDs.
      * @param num       Number of lines (user IDs) to return.
      * @return          Array of user IDs randomly chosen from the file.
      */
    def ask(filename: String, num: Int): Array[String] = {
      // Read all lines at once
      val allLines = Source.fromFile(filename).getLines().toArray
      val stream   = new Array[String](num)

      // Randomly pick `num` users from these lines
      for (i <- 0 until num) {
        val idx = rand.nextInt(allLines.length)
        stream(i) = allLines(idx)
      }
      stream
    }
  }

  // ----------------------------------------------------------------------------
  // Global constants matching your Python approach
  // ----------------------------------------------------------------------------
  val NUM_HASH_FUNCTIONS = 16
  val GROUP_SIZE         = 4          // We have 4 groups, each containing 4 hash funcs
  val BIG_PRIME          = 16769023L  // Large prime for hashing

  // "a" parameters for each of the 16 hash functions
  val A_PARAMS = Array(
    13, 49, 17, 31, 53, 71, 101, 113,
    127, 139, 199, 241, 251, 269, 281, 307
  )

  // "b" parameters for each of the 16 hash functions
  val B_PARAMS = Array(
    7, 11, 23, 29, 47, 59, 83, 97,
    109, 131, 151, 191, 223, 227, 239, 257
  )

  /**
    * countTrailingZeros(num): number of trailing zero bits in the binary representation of `num`.
    * E.g., 12 (binary 1100) => 2 trailing zeros.
    * If num == 0, we define trailing zeros as 0 by convention.
    *
    * @param num The long integer to examine.
    * @return    Count of trailing zeros in the binary form of num.
    */
  def countTrailingZeros(num: Long): Int = {
    if (num == 0L) return 0
    var tmp = num
    var tz = 0
    // Shift right while the least significant bit is 0
    while ((tmp & 1L) == 0L) {
      tz += 1
      tmp = tmp >> 1
    }
    tz
  }

  /**
    * myhashs(userId): produce 16 hash values for the given userId string.
    * Each hash = (a_i * userInt + b_i) mod BIG_PRIME.
    * 
    * We convert `userId` into a BigInt using its UTF-8 bytes, 
    * mirroring Python’s binascii approach.
    *
    * @param userId The user ID as a String.
    * @return       Array[Long] of length 16 representing each hash function’s output.
    */
  def myhashs(userId: String): Array[Long] = {
    // Convert userId string to a BigInt
    val userBigInt = BigInt(1, userId.getBytes("UTF-8"))

    val hashes = new Array[Long](NUM_HASH_FUNCTIONS)
    for (i <- 0 until NUM_HASH_FUNCTIONS) {
      val hv = (A_PARAMS(i) * userBigInt + B_PARAMS(i)) mod BigInt(BIG_PRIME)
      hashes(i) = hv.toLong
    }
    hashes
  }

  /**
    * Main entry point to run Flajolet–Martin distinct counting.
    * 
    * Usage:
    *   spark-submit --class task2 --executor-memory 4G --driver-memory 4G hw5.jar \
    *     <inputFile> <streamSize> <numRequests> <outputFile>
    */
  def main(args: Array[String]): Unit = {

    // A) Minimal SparkContext so we can set log level to ERROR
    val conf = new SparkConf().setAppName("Task2").setMaster("local[*]")
    val sc   = new SparkContext(conf)
    sc.setLogLevel("ERROR")  // Only show ERROR logs, reduce verbosity

    // B) Validate arguments
    if (args.length < 4) {
      System.err.println("Usage: task2 <inputFile> <streamSize> <numRequests> <outputFile>")
      sc.stop()
      sys.exit(1)
    }
    // Parse arguments
    val inputFile   = args(0)           // e.g. "users.txt"
    val streamSize  = args(1).toInt     // e.g. 300
    val numRequests = args(2).toInt     // e.g. 30
    val outputFile  = args(3)           // e.g. "output_task2.csv"

    // Start timing
    val startTime = System.nanoTime()

    // Create BlackBox instance
    val bx = new BlackBox()

    // We'll store CSV lines in a StringBuilder
    val sb = new StringBuilder
    sb.append("Time,Ground Truth,Estimation\n")

    // We also track sums to compute ratio
    var totalGroundTruth = 0.0
    var totalEstimation  = 0.0

    // C) Process each request from the BlackBox
    for (t <- 0 until numRequests) {

      // 1) Retrieve a batch of users
      val batch = bx.ask(inputFile, streamSize)

      // 2) Compute ground truth = number of unique users in this batch
      val uniqueUsers = batch.toSet
      val groundTruth = uniqueUsers.size
      totalGroundTruth += groundTruth

      // 3) For each hash function, track the maximum trailing-zero count
      val maxTrailingZeros = Array.fill[Int](NUM_HASH_FUNCTIONS)(0)
      for (user <- batch) {
        val hvList = myhashs(user)
        for (i <- hvList.indices) {
          val tz = countTrailingZeros(hvList(i))
          if (tz > maxTrailingZeros(i)) {
            maxTrailingZeros(i) = tz
          }
        }
      }

      // 4) Convert trailing-zero counts to estimates = 2^(count)
      val fmEstimates = maxTrailingZeros.map(z => math.pow(2.0, z))

      // 5) Group these 16 estimates into 4 groups, each of size 4
      //    Then compute the average within each group
      val groupSize  = NUM_HASH_FUNCTIONS / GROUP_SIZE  // 16/4 = 4
      val groupAverages = new Array[Double](GROUP_SIZE)
      for (g <- 0 until GROUP_SIZE) {
        val startIdx = g * groupSize
        val endIdx   = startIdx + groupSize
        val groupVal = fmEstimates.slice(startIdx, endIdx)
        val avg      = groupVal.sum / groupVal.length
        groupAverages(g) = avg
      }

      // 6) Sort group averages and take the median of the 4 values
      scala.util.Sorting.quickSort(groupAverages)
      val fmMedian = (groupAverages(1) + groupAverages(2)) / 2.0

      // 7) Count this median as the final estimate for the batch
      totalEstimation += fmMedian

      // Convert estimation to int for CSV output
      val estimatedInt = fmMedian.toInt

      // 8) Append CSV line: "Time,Ground Truth, Estimation"
      sb.append(s"$t,$groundTruth,$estimatedInt\n")
    }

    // Write CSV to <outputFile>
    val writer = new BufferedWriter(new FileWriter(outputFile))
    writer.write(sb.toString)
    writer.close()

    // Calculate end time
    val endTime = System.nanoTime()
    val elapsedSec = (endTime - startTime) / 1e9
    println(f"Execution completed in $elapsedSec%.2f seconds.")

    // D) Debug ratio: sum(estimations)/sum(ground truths), just like your Python
    var ratio = 0.0
    if (totalGroundTruth > 0) {
      ratio = totalEstimation / totalGroundTruth
    }
    println(f"Sum of FM Estimates / Sum of Ground Truth = $ratio%.5f")

    // Stop the SparkContext
    sc.stop()
  }
}
