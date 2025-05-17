import org.apache.spark.{SparkConf, SparkContext}
import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.io.{File, PrintWriter}

object task3 {
  implicit val formats: DefaultFormats.type = DefaultFormats

  def main(args: Array[String]): Unit = {
    /*
      Usage:
      spark-submit --class Task3 <jar_file> <review_file> <business_file> <output_file_a> <output_file_b>
    */
    if (args.length != 4) {
      println("Usage: spark-submit --class Task3 <jar_file> <review_file> <business_file> <output_file_a> <output_file_b>")
      sys.exit(1)
    }

    val reviewPath   = args(0)
    val businessPath = args(1)
    val outputA      = args(2)
    val outputB      = args(3)

    val conf = new SparkConf().setAppName("DSCI 553 Task3").setMaster("local[*]")
    val sc   = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    try {
      // 1) Load & parse reviews
      val reviewRDD = sc.textFile(reviewPath).flatMap { line =>
        parseOpt(line)
      }.map { j =>
        val bid   = (j \ "business_id").extract[String]
        val stars = (j \ "stars").extract[Double]
        (bid, stars)
      }

      // 2) Load & parse businesses
      val businessRDD = sc.textFile(businessPath).flatMap { line =>
        parseOpt(line)
      }.map { j =>
        val bid  = (j \ "business_id").extract[String]
        val city = (j \ "city").extract[String]
        (bid, city)
      }

      // 3) Join on business_id => (bid, (stars, city))
      val joinedRDD = reviewRDD.join(businessRDD)
      // => (bid, (stars, city))

      // 4) city => (sumStars, count), then compute average
      val cityStarsRDD = joinedRDD.map {
        case (_, (stars, city)) =>
          (city, (stars, 1L))
      }.reduceByKey {
        case ((s1, c1), (s2, c2)) => (s1 + s2, c1 + c2)
      }.mapValues {
        case (s, c) =>
          // If you need no rounding, just do s/c
          // Here we match PySpark's 2-decimal rounding
          val avg = s / c
          BigDecimal(avg).setScale(2, BigDecimal.RoundingMode.HALF_UP).toDouble
      }

      // 5) Sort by descending average, tie-break ascending city
      //    This matches .sortBy(lambda x: (-x[1], x[0])) in PySpark
      val sortedCityStars = cityStarsRDD.sortBy(
        { case (city, avg) => (-avg, city) },
        ascending = true
      )

      // Collect for writing
      val finalCityStars = sortedCityStars.collect()

      // 6) Write file for Task3 (A)
      val writerA = new PrintWriter(new File(outputA))
      writerA.println("city,stars")
      finalCityStars.foreach {
        case (city, avg) => writerA.println(s"$city,$avg")
      }
      writerA.close()

      // 7) Compare M1 vs. M2 (top 10)
      //    For M1, we measure after we have the (city -> avg) RDD
      val m1Start = System.nanoTime()
      val subset  = finalCityStars.take(100) // local subset
      // sort in Python-like manner: first by -avg, then city
      val top10M1 = subset.sortBy { case (city, avg) => (-avg, city) }.take(10)
      val m1Time  = (System.nanoTime() - m1Start) / 1e9

      // M2: Spark's takeOrdered(10)
      val m2Start = System.nanoTime()
      val top10M2 = sortedCityStars.takeOrdered(10)(
        Ordering.by { case (city, avg) => (-avg, city) }
      )
      val m2Time = (System.nanoTime() - m2Start) / 1e9

      // 8) Write file for Task3 (B)
      val reasonStr = "Spark sorting is optimized for distributed computation, but on small data, Python sorting is also efficient."
      val jsonOutput =
        s"""
           |{
           |  "m1": $m1Time,
           |  "m2": $m2Time,
           |  "reason": "$reasonStr"
           |}
           |""".stripMargin

      val writerB = new PrintWriter(new File(outputB))
      writerB.write(jsonOutput)
      writerB.close()

    } finally {
      sc.stop()
    }
  }
}
