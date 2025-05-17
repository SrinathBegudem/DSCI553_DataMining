import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import scala.util.parsing.json.JSON
import java.io._

object task1 {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: spark-submit Task1.scala <input_filepath> <output_filepath>")
      sys.exit(1)
    }

    val conf = new SparkConf().setAppName("DSCI 553 Task 1").setMaster("local[*]")
    val sc = new SparkContext(conf)

    try {
      // Load input file
      val reviewRDD: RDD[String] = sc.textFile(args(0))

      // Parse JSON records
      val parsedRDD: RDD[Map[String, Any]] = reviewRDD.flatMap(line => JSON.parseFull(line).asInstanceOf[Option[Map[String, Any]]])

      // (A) Count total number of reviews
      val totalReviews: Long = parsedRDD.count()

      // (B) Count reviews from 2018
      val reviews2018: Long = parsedRDD.filter(_.getOrElse("date", "").toString.startsWith("2018")).count()

      // (C) Count distinct users
      val distinctUsers: Long = parsedRDD.map(_.getOrElse("user_id", "").toString).distinct().count()

      // (D) Top 10 users by number of reviews, sorted by (count DESC, user_id ASC)
      val topUsers = parsedRDD
        .map(x => (x.getOrElse("user_id", "").toString, 1))
        .reduceByKey(_ + _)
        .takeOrdered(10)(Ordering.by[(String, Int), (Int, String)](x => (-x._2, x._1))) // Fixed sorting

      // (E) Count distinct businesses
      val distinctBusinesses: Long = parsedRDD.map(_.getOrElse("business_id", "").toString).distinct().count()

      // (F) Top 10 businesses by number of reviews, sorted by (count DESC, business_id ASC)
      val topBusinesses = parsedRDD
        .map(x => (x.getOrElse("business_id", "").toString, 1))
        .reduceByKey(_ + _)
        .takeOrdered(10)(Ordering.by[(String, Int), (Int, String)](x => (-x._2, x._1))) // Fixed sorting

      // Prepare output
      val outputData =
        s"""
        {
          "n_review": $totalReviews,
          "n_review_2018": $reviews2018,
          "n_user": $distinctUsers,
          "top10_user": ${topUsers.map(x => s"""["${x._1}", ${x._2}]""").mkString("[", ", ", "]")},
          "n_business": $distinctBusinesses,
          "top10_business": ${topBusinesses.map(x => s"""["${x._1}", ${x._2}]""").mkString("[", ", ", "]")}
        }
        """.stripMargin

      // Save JSON output
      val writer = new PrintWriter(new File(args(1)))
      writer.write(outputData)
      writer.close()

      println(s"✅ Output successfully written to ${args(1)}")

    } finally {
      sc.stop()
    }
  }
}
