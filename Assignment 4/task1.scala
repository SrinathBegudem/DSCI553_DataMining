import org.apache.spark.sql.SparkSession
import org.graphframes.GraphFrame

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Task1.scala
 *
 * DSCI-553: Assignment 4 (Spring 2025)
 * -------------------------------------
 * Goal: Use GraphFrames + Label Propagation (LPA)
 *       to detect communities in an undirected graph,
 *       based on users who share >= threshold businesses.
 *
 * 1) We read input CSV with (user_id, business_id).
 * 2) Build user -> set(business) map.
 * 3) For each pair of users, if overlap >= threshold, we add an edge.
 * 4) Create a GraphFrame with these edges (undirected) & nodes.
 * 5) Run LPA with maxIter=5 to detect communities.
 * 6) Sort communities by (size asc, then lexicographically by smallest user).
 * 7) Write each community line => 'user1', 'user2', 'user3', ...
 *
 * This Scala code mirrors the working Python logic, ensuring
 * only users who share at least one threshold-based edge are included.
 *
 * Example Usage (spark-submit):
 *   spark-submit \
 *     --class Task1 \
 *     target/scala-2.12/hw4.jar \
 *     7 data/ub_sample_data.csv output_task1.txt
 */
object task1 {
  def main(args: Array[String]): Unit = {

    // -----------------------------------------------------------------------
    //  (A) Parse Command-Line Arguments
    // -----------------------------------------------------------------------
    // The assignment expects:
    //  arg(0) = filterThreshold (int)
    //  arg(1) = inputFilePath (string)
    //  arg(2) = outputFilePath (string)
    if (args.length != 3) {
      println("Usage: Task1 <filter_threshold> <input_file_path> <output_file_path>")
      sys.exit(1)
    }

    // Convert the first arg to an integer threshold
    val filterThreshold = args(0).toInt
    // The CSV file path
    val inputPath       = args(1)
    // The final output text file path
    val outputPath      = args(2)

    // -----------------------------------------------------------------------
    //  (B) Initialize SparkSession
    // -----------------------------------------------------------------------
    // "local[*]" means we use all available cores on this machine,
    // but you can omit master() and let spark-submit specify it.
    val spark = SparkSession.builder()
      .appName("Task1-GraphFrames")
      .getOrCreate()

    // We'll reduce Spark's console logs to "ERROR" only:
    val sc = spark.sparkContext
    sc.setLogLevel("ERROR")

    // -----------------------------------------------------------------------
    //  (C) Read & Preprocess the Input CSV
    // -----------------------------------------------------------------------
    // The CSV has a header, so we specify .option("header", "true").
    // Then we select the "user_id" and "business_id" columns to remain consistent.
    val rawDF = spark.read
      .option("header", "true")
      .csv(inputPath)

    // We convert each row => (String user, String business).
    // This is an RDD of (user_id, business_id) pairs.
    val userBizRDD = rawDF.rdd.map { row =>
      val user = row.getString(0)   // user_id
      val biz  = row.getString(1)   // business_id
      (user, biz)
    }

    // -----------------------------------------------------------------------
    //  (D) Build a Map of user -> Set[businesses]
    // -----------------------------------------------------------------------
    // We'll group by user and collect the businesses as a set, which
    // helps us quickly do an intersection to check overlap.
    // Then .collectAsMap() pulls it to the driver.
    val userBizMap = userBizRDD
      .groupByKey()
      .mapValues(_.toSet)
      .collectAsMap()

    // -----------------------------------------------------------------------
    //  (E) Gather All Potential Users from userBizMap
    // -----------------------------------------------------------------------
    // In Python, we used the same approach to avoid including users
    // that have NO edges. userBizMap.keySet simply has all users
    // that appear in the CSV. We'll filter down further when we build edges.
    val userArray = userBizMap.keySet.toArray.sorted

    // -----------------------------------------------------------------------
    //  (F) Identify Undirected Edges Based on Overlap
    // -----------------------------------------------------------------------
    // If two users share >= filterThreshold businesses,
    // we add an edge in both directions to edgesList
    // and also add them to nodeSet.
    val edgesList = new ArrayBuffer[(String, String)]()
    val nodeSet   = mutable.Set[String]()

    for (i <- userArray.indices; j <- i + 1 until userArray.length) {
      val u1 = userArray(i)
      val u2 = userArray(j)

      // Retrieve each user's set of businesses
      val busA = userBizMap(u1)
      val busB = userBizMap(u2)

      // Intersection
      val commonCount = busA.intersect(busB).size

      // If the overlap is >= threshold, there's an edge
      if (commonCount >= filterThreshold) {
        // Add edges in both directions (u1->u2, u2->u1)
        edgesList += ((u1, u2))
        edgesList += ((u2, u1))

        // Mark them as relevant nodes
        nodeSet += u1
        nodeSet += u2
      }
    }

    // -----------------------------------------------------------------------
    //  (G) Create GraphFrame (nodes, edges)
    // -----------------------------------------------------------------------
    // We'll convert nodeSet => a DataFrame column "id"
    // and edgesList => a DataFrame with columns "src" and "dst".
    import spark.implicits._
    val nodesDF = nodeSet.toSeq.toDF("id")
    val edgesDF = edgesList.toSeq.toDF("src", "dst")

    // Build the GraphFrame, which can be used to run LPA.
    val gf = GraphFrame(nodesDF, edgesDF)

    // -----------------------------------------------------------------------
    //  (H) Run Label Propagation Algorithm (LPA) with maxIter=5
    // -----------------------------------------------------------------------
    // This returns a DataFrame with columns ["id", "label"].
    // "label" is a long representing the community ID after LPA.
    val lpaDF = gf.labelPropagation.maxIter(5).run()

    // -----------------------------------------------------------------------
    //  (I) Group the LPA results => communities
    // -----------------------------------------------------------------------
    // We do a standard approach:
    //   - RDD of (label, user)
    //   - groupByKey => label => multiple users
    //   - sort each community's user IDs
    //   - drop the label, keep just the sorted user-list
    //   - finally, sort the entire set of communities by:
    //        1) ascending length
    //        2) lexicographically smallest user
    val communitiesRDD = lpaDF.rdd
      .map { row =>
        val user = row.getAs[String]("id")
        val labl = row.getAs[Long]("label")
        (labl, user)
      }
      .groupByKey()
      .mapValues(users => users.toList.sorted)
      .map(_._2) // drop the label, keep the sorted user list
      .sortBy(community => (community.size, community.headOption.getOrElse("")))

    // -----------------------------------------------------------------------
    //  (J) Write Communities to Output File
    // -----------------------------------------------------------------------
    // Each line => 'userA', 'userB', ...
    // We'll create a PrintWriter => output file, then print each community line.
    val finalComms = communitiesRDD.collect()
    val pw = new java.io.PrintWriter(outputPath)

    for (comm <- finalComms) {
      // We wrap each user in single quotes, joined by ", "
      val line = comm.map(u => s"'$u'").mkString(", ")
      pw.println(line)
    }

    pw.close()

    // -----------------------------------------------------------------------
    //  (K) Cleanup
    // -----------------------------------------------------------------------
    // Stop the Spark session. We’re done!
    spark.stop()
  }
}
