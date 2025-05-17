import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import scala.util.Random
import java.io.{File, PrintWriter}
import scala.collection.Map  // Add this import

object task1 {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: task1 <input_file_name> <output_file_name>")
      System.exit(1)
    }
    
    val inputFile = args(0)
    val outputFile = args(1)
    
    // Initialize Spark
    val conf = new SparkConf().setAppName("Task1_LSH")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    
    // Track execution time
    val startTime = System.currentTimeMillis()
    
    // Read input data and create business-user mappings
    val (businessUsersRDD, businessUsersDict) = readInputData(sc, inputFile)
    
    // Create dictionary mapping users to unique indices
    val userDict = businessUsersRDD.flatMap(_._2).distinct().zipWithIndex().collectAsMap()
    
    // Define LSH parameters
    val numHashes = 50
    val numBands = 25
    val rowsPerBand = numHashes / numBands
    val maxValue = userDict.size
    
    // Generate hash functions
    val hashFuncs = generateHashFunctions(numHashes, maxValue)
    
    // Compute MinHash signatures
    val signatureRDD = computeMinHashSignatures(
      businessUsersRDD, hashFuncs, userDict, numHashes, hashFuncs._3, maxValue
    )
    
    // Apply LSH to find candidate pairs
    val candidatePairsRDD = applyLSH(signatureRDD, numBands, rowsPerBand)
    
    // Calculate Jaccard similarity for candidate pairs
    val resultsRDD = calculateJaccardSimilarity(candidatePairsRDD, businessUsersDict)
    
    // Write results to output file
    writeOutputFile(outputFile, resultsRDD)
    
    val duration = (System.currentTimeMillis() - startTime) / 1000.0
    println(s"Task 1 completed successfully in $duration seconds.")
    sc.stop()
  }
  
  // Read and parse input data
  def readInputData(sc: SparkContext, inputFile: String): (RDD[(String, Set[String])], scala.collection.Map[String, Set[String]]) = {
    // Load dataset
    val lines = sc.textFile(inputFile)
    val header = lines.first()
    
    // Create (business_id, user_id) pairs
    val businessUsersRDD = lines
      .filter(_ != header)
      .map(_.split(","))
      .map(row => (row(1), row(0)))  // (business_id, user_id)
      .groupByKey()
      .mapValues(_.toSet)
    
    // Create lookup dictionary
    val businessUsersDict = businessUsersRDD.collectAsMap()
    
    (businessUsersRDD, businessUsersDict)
  }
  
  // Generate hash function parameters
  def generateHashFunctions(n: Int, maxValue: Int, prime: Int = 16777619): (Array[Int], Array[Int], Int) = {
    val rand = new Random(42) // Set seed for reproducibility
    val a = Array.fill(n)(rand.nextInt(maxValue) + 1) // Avoid 0
    val b = Array.fill(n)(rand.nextInt(maxValue) + 1)
    (a, b, prime)
  }
  
  // Compute MinHash signatures
  def computeMinHashSignatures(
    businessUsersRDD: RDD[(String, Set[String])], 
    hashFuncs: (Array[Int], Array[Int], Int),
    userDict: scala.collection.Map[String, Long],
    numHashes: Int,
    prime: Int,
    maxValue: Int
  ): RDD[(String, Array[Int])] = {
    val (a, b, _) = hashFuncs
    
    businessUsersRDD.mapValues { userSet =>
      Array.tabulate(numHashes) { i =>
        userSet.map { user =>
          val userIdx = userDict(user).toInt
          ((a(i).toLong * userIdx + b(i)) % prime % maxValue).toInt
        }.min
      }
    }
  }
  
  // Apply LSH to find candidate pairs
  def applyLSH(
    signatureRDD: RDD[(String, Array[Int])],
    numBands: Int,
    rowsPerBand: Int
  ): RDD[(String, String)] = {
    
    val bandedRDD = signatureRDD.flatMap { case (businessId, signature) =>
      // Create (band_index, band_signature) -> business_id pairs
      (0 until numBands).map { i =>
        val bandStart = i * rowsPerBand
        val bandEnd = bandStart + rowsPerBand
        val bandSignature = signature.slice(bandStart, bandEnd).mkString(",")
        ((i, bandSignature), businessId)
      }
    }
    
    bandedRDD
      .groupByKey()                     // Group businesses with the same band signature
      .filter(_._2.size > 1)            // Keep only buckets with multiple businesses
      .flatMap { case (_, businesses) =>
        val businessList = businesses.toList.sorted   // Sort for deterministic output
        // Generate all pairs of businesses in this bucket
        for {
          i <- 0 until businessList.size
          j <- (i + 1) until businessList.size
        } yield (businessList(i), businessList(j))
      }
      .distinct()                       // Remove duplicate candidate pairs
  }
  
  // Calculate Jaccard similarity
  def calculateJaccardSimilarity(
    candidatePairsRDD: RDD[(String, String)],
    businessUsersDict: scala.collection.Map[String, Set[String]]
  ): RDD[(String, String, Double)] = {
    
    candidatePairsRDD.map { case (b1, b2) =>
      val users1 = businessUsersDict.getOrElse(b1, Set.empty[String])
      val users2 = businessUsersDict.getOrElse(b2, Set.empty[String])
      
      val intersection = users1.intersect(users2).size
      val union = users1.union(users2).size
      val similarity = intersection.toDouble / union
      
      // Return the pair if similarity meets the threshold
      if (similarity >= 0.5) (b1, b2, similarity) else null
    }.filter(_ != null)
  }
  
  // Write results to CSV file
  def writeOutputFile(outputFile: String, resultsRDD: RDD[(String, String, Double)]): Unit = {
    val results = resultsRDD.collect().sortBy(x => (x._1, x._2))  // Sort lexicographically
    
    val pw = new PrintWriter(new File(outputFile))
    try {
      pw.println("business_id_1,business_id_2,similarity")
      results.foreach { case (b1, b2, sim) =>
        pw.println(s"$b1,$b2,$sim")
      }
    } finally {
      pw.close()
    }
  }
}