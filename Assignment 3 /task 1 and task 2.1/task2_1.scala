import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import java.io.{File, PrintWriter}
import scala.collection.mutable
import scala.math._
import scala.util.{Try, Success, Failure}

object task2_1 {
  def main(args: Array[String]): Unit = {
    // Check command line arguments
    if (args.length != 3) {
      println("Usage: task2_1 <train_file_name> <test_file_name> <output_file_name>")
      System.exit(1)
    }

    val trainFile = args(0)
    val testFile = args(1)
    val outputFile = args(2)

    // Initialize Spark
    val conf = new SparkConf().setAppName("Task2_1_ItemBasedCF")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    // Track execution time
    val startTime = System.currentTimeMillis()

    // Read training data
    val trainRDD = readTrainData(sc, trainFile)
    trainRDD.cache()

    // Compute averages
    val (businessAvg, userAvg, globalAvg) = computeAverages(trainRDD)

    // Build rating dictionaries
    val (userBusinesses, businessUsers) = buildRatingDicts(trainRDD)

    // Read test data
    val testPairs = readTestData(sc, testFile)

    // Initialize similarity cache
    val similarityCache = mutable.Map[(String, String), Double]()

    // Make predictions
    val predictions = testPairs.map { case (userId, businessId) =>
      val predictedRating = predictRating(
        userId, businessId, userBusinesses, businessUsers,
        businessAvg, userAvg, globalAvg, similarityCache
      )
      (userId, businessId, predictedRating)
    }

    // Write predictions to output file
    val pw = new PrintWriter(new File(outputFile))
    try {
      pw.println("user_id,business_id,prediction")
      predictions.foreach { case (userId, businessId, rating) =>
        pw.println(s"$userId,$businessId,$rating")
      }
    } finally {
      pw.close()
    }

    // Try to calculate RMSE if validation data has ratings (for self-evaluation)
    try {
      // Read validation data (test file) with ratings
      val validationLines = sc.textFile(testFile)
      val validationHeader = validationLines.first()
      
      val validationRDD = validationLines
        .filter(_ != validationHeader)  // Skip header
        .map(_.split(","))
        .filter(r => r.length >= 3)  // Make sure rating column exists
        .map(r => ((r(0), r(1)), r(2).toDouble))
      
      // Check if we have validation data with ratings
      if (!validationRDD.isEmpty()) {
        // Convert to map for faster lookup
        val validationDict = validationRDD.collectAsMap()
        
        // Initialize error variables
        var totalSquaredError = 0.0
        var count = 0
        
        // Initialize counters for different error ranges
        val errorRanges = mutable.Map(
          ">=0 and <1" -> 0,
          ">=1 and <2" -> 0,
          ">=2 and <3" -> 0,
          ">=3 and <4" -> 0,
          ">=4" -> 0
        )
        
        // Calculate error for each prediction
        predictions.foreach { case (userId, businessId, predRating) =>
          val key = (userId, businessId)
          if (validationDict.contains(key)) {
            val actual = validationDict(key)
            val error = abs(predRating - actual)
            
            // Update RMSE calculation
            totalSquaredError += error * error
            count += 1
            
            // Categorize error into appropriate range
            if (error < 1) errorRanges(">=0 and <1") += 1
            else if (error < 2) errorRanges(">=1 and <2") += 1
            else if (error < 3) errorRanges(">=2 and <3") += 1
            else if (error < 4) errorRanges(">=3 and <4") += 1
            else errorRanges(">=4") += 1
          }
        }
        
        // Calculate and print RMSE
        if (count > 0) {
          val rmse = sqrt(totalSquaredError / count)
          println(s"RMSE: $rmse")
        }
        
        // Print error distribution
        println("Error Distribution:")
        errorRanges.foreach { case (rangeLabel, count) =>
          println(s"  $rangeLabel: $count")
        }
      }
    } catch {
      case e: Exception => 
        // If RMSE calculation fails, just skip it
        println("Skipping RMSE calculation (expected if test file doesn't have ratings)")
    }

    val executionTime = (System.currentTimeMillis() - startTime) / 1000.0
    println(s"Task 2.1 completed successfully in $executionTime seconds.")
    sc.stop()
  }

  def readTrainData(sc: SparkContext, trainFile: String): RDD[(String, String, Double)] = {
    // Load dataset
    val lines = sc.textFile(trainFile)
    val header = lines.first()

    // Parse training data
    lines
      .filter(_ != header)
      .map(_.split(","))
      .filter(r => r.length >= 3)
      .map(r => (r(0), r(1), r(2).toDouble))
  }

  def readTestData(sc: SparkContext, testFile: String): Array[(String, String)] = {
    // Load dataset
    val lines = sc.textFile(testFile)
    val header = lines.first()

    // Parse test data without ratings
    lines
      .filter(_ != header)
      .map(_.split(","))
      .filter(r => r.length >= 2)
      .map(r => (r(0), r(1)))
      .collect()
  }

  def computeAverages(trainRDD: RDD[(String, String, Double)]): (Map[String, Double], Map[String, Double], Double) = {
    // Business averages
    val businessAvg = trainRDD
      .map(x => (x._2, (x._3, 1)))
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .mapValues(t => t._1 / t._2)
      .collectAsMap()
      .toMap

    // User averages
    val userAvg = trainRDD
      .map(x => (x._1, (x._3, 1)))
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2))
      .mapValues(t => t._1 / t._2)
      .collectAsMap()
      .toMap

    // Global average
    val (total, count) = trainRDD
      .map(x => (x._3, 1))
      .reduce((a, b) => (a._1 + b._1, a._2 + b._2))
    val globalAvg = total / count

    (businessAvg, userAvg, globalAvg)
  }

  def buildRatingDicts(trainRDD: RDD[(String, String, Double)]): (Map[String, Map[String, Double]], Map[String, Map[String, Double]]) = {
    // User -> Businesses dictionary
    val userBusinesses = trainRDD
      .map(x => (x._1, (x._2, x._3)))
      .groupByKey()
      .mapValues(recs => recs.toMap)
      .collectAsMap()
      .toMap

    // Business -> Users dictionary
    val businessUsers = trainRDD
      .map(x => (x._2, (x._1, x._3)))
      .groupByKey()
      .mapValues(recs => recs.toMap)
      .collectAsMap()
      .toMap

    (userBusinesses, businessUsers)
  }

  def computePearsonSimilarity(
    business1: String,
    business2: String,
    businessUsers: Map[String, Map[String, Double]],
    businessAvg: Map[String, Double],
    similarityCache: mutable.Map[(String, String), Double]
  ): Double = {
    // Use cache for faster computation
    val cacheKey = if (business1 < business2) (business1, business2) else (business2, business1)
    
    // Return cached value if available
    if (similarityCache.contains(cacheKey)) {
      return similarityCache(cacheKey)
    }

    // Find common users
    val users1 = businessUsers(business1).keySet
    val users2 = businessUsers(business2).keySet
    val commonUsers = users1.intersect(users2)

    // Handle special cases with few common users
    if (commonUsers.size <= 1) {
      val similarity = (5.0 - abs(businessAvg(business1) - businessAvg(business2))) / 5.0
      similarityCache(cacheKey) = similarity
      return similarity
    }

    // For exactly 2 common users, use average of rating differences
    if (commonUsers.size == 2) {
      val usersList = commonUsers.toList
      val sim1 = (5.0 - abs(businessUsers(business1)(usersList(0)) - businessUsers(business2)(usersList(0)))) / 5.0
      val sim2 = (5.0 - abs(businessUsers(business1)(usersList(1)) - businessUsers(business2)(usersList(1)))) / 5.0
      val similarity = (sim1 + sim2) / 2
      similarityCache(cacheKey) = similarity
      return similarity
    }

    // Extract ratings for common users
    val ratings1 = commonUsers.toList.map(user => businessUsers(business1)(user))
    val ratings2 = commonUsers.toList.map(user => businessUsers(business2)(user))

    // Calculate means
    val mean1 = ratings1.sum / ratings1.size
    val mean2 = ratings2.sum / ratings2.size

    // Center the ratings
    val centered1 = ratings1.map(_ - mean1)
    val centered2 = ratings2.map(_ - mean2)

    // Calculate Pearson correlation
    val numerator = centered1.zip(centered2).map { case (a, b) => a * b }.sum
    val denominator = sqrt(centered1.map(a => a * a).sum) * sqrt(centered2.map(b => b * b).sum)

    val similarity = if (denominator == 0) 0.0 else numerator / denominator

    // Cache the result
    similarityCache(cacheKey) = similarity
    similarity
  }

  def predictRating(
    userId: String,
    businessId: String,
    userBusinesses: Map[String, Map[String, Double]],
    businessUsers: Map[String, Map[String, Double]],
    businessAvg: Map[String, Double],
    userAvg: Map[String, Double],
    globalAvg: Double,
    similarityCache: mutable.Map[(String, String), Double],
    neighborCount: Int = 15
  ): Double = {
    // Check if user has already rated this business
    if (userBusinesses.contains(userId) && userBusinesses(userId).contains(businessId)) {
      return userBusinesses(userId)(businessId)
    }

    // Handle cold start problems
    if (!userBusinesses.contains(userId)) {
      return businessAvg.getOrElse(businessId, globalAvg)
    }

    if (!businessUsers.contains(businessId)) {
      return userAvg.getOrElse(userId, globalAvg)
    }

    // Find similar items that the user has rated
    val similarItems = userBusinesses(userId).flatMap { case (otherBusiness, rating) =>
      // Skip if same business or not in training data
      if (otherBusiness == businessId || !businessUsers.contains(otherBusiness)) {
        None
      } else {
        // Compute similarity
        val similarity = computePearsonSimilarity(
          businessId, otherBusiness, businessUsers, businessAvg, similarityCache
        )
        
        // Only use positive similarities
        if (similarity > 0) Some((similarity, rating)) else None
      }
    }.toList

    // Fallback if no similar items found
    if (similarItems.isEmpty) {
      return userAvg.getOrElse(userId, globalAvg)
    }

    // Calculate weighted average
    val topNeighbors = similarItems.sortBy(-_._1).take(neighborCount)
    val numerator = topNeighbors.map { case (sim, r) => sim * r }.sum
    val denominator = topNeighbors.map { case (sim, _) => abs(sim) }.sum

    // Handle edge case
    if (denominator == 0) {
      return userAvg.getOrElse(userId, globalAvg)
    }

    // Final rating prediction
    val predictedRating = numerator / denominator
    min(max(predictedRating, 1.0), 5.0)
  }
}