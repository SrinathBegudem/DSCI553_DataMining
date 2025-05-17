import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors

// Spark MLlib for Gradient-Boosted Trees
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy

import java.io.{File, PrintWriter}
import scala.math._

object task2_2 {

  // -------------------------------------------------------------------
  // Main entry point
  // -------------------------------------------------------------------
  def main(args: Array[String]): Unit = {

    // -------------------------------
    // 1. Parse arguments
    // -------------------------------
    if (args.length != 3) {
      println("Usage: spark-submit --class hw3.task2_2 hw3.jar <folder_path> <test_file_name> <output_file_name>")
      sys.exit(1)
    }
    val folderPath = args(0)      // Path to folder containing the JSON files + yelp_train.csv
    val testFile   = args(1)      // Path to test CSV file
    val outputFile = args(2)      // Where we write predictions

    // -------------------------------
    // 2. Spark Context
    // -------------------------------
    val conf = new SparkConf().setAppName("Task2_2_ModelBasedRS")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val startTime = System.currentTimeMillis()

    // -------------------------------
    // 3. Load & Process JSON Data
    // -------------------------------
    // We will parse:
    //   - review_train.json -> average (useful, funny, cool) per business
    //   - user.json -> (average_stars, review_count, fans) per user
    //   - business.json -> (stars, review_count) per business
    //
    // Then collect them into driver maps for fast lookups (broadcast later).

    // a) REVIEW data
    val reviewPath   = s"$folderPath/review_train.json"
    val reviewLines  = sc.textFile(reviewPath)

    // Parse each line as JSON. We do a manual parse with json4s.
    val businessReviewRDD: RDD[(String, (Double, Double, Double))] = reviewLines.mapPartitions { iter =>
      implicit val formats: DefaultFormats.type = DefaultFormats
      iter.map { line =>
        val jval = parse(line)
        val businessId = (jval \ "business_id").extract[String]
        val useful  = (jval \ "useful").extract[Double]
        val funny   = (jval \ "funny").extract[Double]
        val cool    = (jval \ "cool").extract[Double]
        (businessId, (useful, funny, cool))
      }
    }

    // Compute average (useful, funny, cool) for each business
    val reviewFeatures = businessReviewRDD
      .groupByKey()
      .mapValues { feats =>
        val n = feats.size
        val sums = feats.foldLeft((0.0, 0.0, 0.0)) {
          case ((su, sf, sc), (u, f, c)) => (su + u, sf + f, sc + c)
        }
        (sums._1 / n, sums._2 / n, sums._3 / n)
      }
      .collectAsMap()

    // b) USER data
    val userPath  = s"$folderPath/user.json"
    val userLines = sc.textFile(userPath)

    val userRDD: RDD[(String, (Double, Double, Double))] = userLines.mapPartitions { iter =>
      implicit val formats: DefaultFormats.type = DefaultFormats
      iter.map { line =>
        val jval = parse(line)
        val user_id       = (jval \ "user_id").extract[String]
        val average_stars = (jval \ "average_stars").extract[Double]
        val review_count  = (jval \ "review_count").extract[Double]
        val fans          = (jval \ "fans").extract[Double]
        (user_id, (average_stars, review_count, fans))
      }
    }
    val userFeatures = userRDD.collectAsMap()

    // c) BUSINESS data
    val busPath  = s"$folderPath/business.json"
    val busLines = sc.textFile(busPath)

    val busRDD: RDD[(String, (Double, Double))] = busLines.mapPartitions { iter =>
      implicit val formats: DefaultFormats.type = DefaultFormats
      iter.map { line =>
        val jval = parse(line)
        val businessId   = (jval \ "business_id").extract[String]
        val stars        = (jval \ "stars").extract[Double]
        val review_count = (jval \ "review_count").extract[Double]
        (businessId, (stars, review_count))
      }
    }
    val businessFeatures = busRDD.collectAsMap()

    // Broadcast all maps
    val brReviewMap   = sc.broadcast(reviewFeatures)
    val brUserMap     = sc.broadcast(userFeatures)
    val brBusinessMap = sc.broadcast(businessFeatures)

    // -------------------------------
    // 4. Load Training Data (yelp_train.csv) & Prepare Features
    // -------------------------------
    val trainFile = s"$folderPath/yelp_train.csv"
    val trainRaw  = sc.textFile(trainFile)
    val trainHeader = trainRaw.first()

    // (user_id, business_id, rating)
    val trainRDD = trainRaw
      .filter(_ != trainHeader)
      .map(_.split(","))
      .filter(_.length >= 3)
      .map { arr =>
        val userId = arr(0)
        val busId  = arr(1)
        val rating = arr(2).toDouble
        (userId, busId, rating)
      }

    // Build LabeledPoint for each record:
    //  [ avg_useful, avg_funny, avg_cool,
    //    user_avg_stars, user_review_count, user_fans,
    //    business_avg_stars, business_review_count ]
    val trainingData = trainRDD.map { case (userId, busId, rating) =>
      val (useful, funny, cool) = brReviewMap.value.getOrElse(busId, (0.0, 0.0, 0.0))
      val (uStars, uReviews, uFans) = brUserMap.value.getOrElse(userId, (3.5, 0.0, 0.0))
      val (bStars, bReviews) = brBusinessMap.value.getOrElse(busId, (3.5, 0.0))

      val features = Vectors.dense(
        useful, funny, cool,
        uStars, uReviews, uFans,
        bStars, bReviews
      )
      LabeledPoint(rating, features)
    }.cache()

    // -------------------------------
    // 5. Train Model (Gradient Boosted Trees)
    // -------------------------------
    // "trainRegressor(...)" is not present in some Spark distributions. 
    // We can use "train(...)" with a 'Regression' boostingStrategy.
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.numIterations = 55      // # of trees
    boostingStrategy.treeStrategy.maxDepth = 7

    // Train the model
    val gbtModel = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // -------------------------------
    // 6. Prepare Test Data
    // -------------------------------
    val testRaw   = sc.textFile(testFile)
    val testHeader= testRaw.first()
    val testRDD   = testRaw
      .filter(_ != testHeader)
      .map(_.split(","))
      .filter(_.length >= 2)
      .map(arr => (arr(0), arr(1)))  // (user_id, business_id)

    // We'll collect so we can write in the same order. 
    // For very large data, you might do a distributed write.
    val testPairs = testRDD.collect()

    // -------------------------------
    // 7. Predict
    // -------------------------------
    val broadcastPairs = sc.broadcast(testPairs)
    val predictions: Array[(String, String, Double)] = broadcastPairs.value.map { case (userId, busId) =>
      val (useful, funny, cool)     = brReviewMap.value.getOrElse(busId, (0.0, 0.0, 0.0))
      val (uStars, uReviews, uFans) = brUserMap.value.getOrElse(userId, (3.5, 0.0, 0.0))
      val (bStars, bReviews)        = brBusinessMap.value.getOrElse(busId, (3.5, 0.0))

      val featVec = Vectors.dense(useful, funny, cool, uStars, uReviews, uFans, bStars, bReviews)
      val rawPred = gbtModel.predict(featVec)

      // Clip to [1.0, 5.0]
      val finalPred = math.max(1.0, math.min(5.0, rawPred))
      (userId, busId, finalPred)
    }

    // -------------------------------
    // 8. Write Output CSV
    // -------------------------------
    val writer = new PrintWriter(new File(outputFile))
    try {
      writer.write("user_id,business_id,prediction\n")
      predictions.foreach { case (u, b, p) =>
        writer.write(s"$u,$b,$p\n")
      }
    } finally {
      writer.close()
    }

    // -------------------------------
    // 9. [Optional] RMSE if test file has ratings
    // -------------------------------
    try {
      val testDataWithRatings = testRaw
        .filter(_ != testHeader)
        .map(_.split(","))
        .filter(_.length >= 3)
        .map(arr => ((arr(0), arr(1)), arr(2).toDouble))

      val actualMap = testDataWithRatings.collectAsMap()
      var seSum = 0.0
      var count = 0L

      predictions.foreach { case (u, b, pred) =>
        val key = (u, b)
        if (actualMap.contains(key)) {
          val actual = actualMap(key)
          val err = pred - actual
          seSum += (err * err)
          count += 1
        }
      }

      if (count > 0) {
        val rmse = math.sqrt(seSum / count)
        println(f"RMSE on provided test file: $rmse%.4f (based on $count%d labeled records)")
      }
    } catch {
      case _: Throwable => // Test file might not have ratings
    }

    // -------------------------------
    // 10. Done
    // -------------------------------
    val duration = (System.currentTimeMillis() - startTime) / 1000.0
    println(f"Task2_2 completed in $duration%.2f seconds.")
    sc.stop()
  }
}