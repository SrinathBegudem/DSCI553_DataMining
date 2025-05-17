import org.apache.spark.{SparkConf, SparkContext, HashPartitioner}
import org.json4s._
import org.json4s.jackson.JsonMethods._
import java.io._

object task2 {
  implicit val formats: DefaultFormats.type = DefaultFormats

  def main(args: Array[String]): Unit = {
    // Expected args: <inputFile> <outputFile> <numPartitions>
    if (args.length != 3) {
      println("Usage: spark-submit --class Task2 <jar_file> <input_filepath> <output_filepath> <n_partitions>")
      sys.exit(1)
    }

    val inputPath     = args(0)
    val outputPath    = args(1)
    val numPartitions = args(2).toInt

    val conf = new SparkConf().setAppName("Task2").setMaster("local[*]")
    val sc   = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    try {
      // 1) Read the input with 20 partitions to closely match PySpark’s default distribution
      val rawRDD = sc.textFile(inputPath, 20)

      // 2) Safely parse JSON and extract "business_id"
      val businessRDD = rawRDD.flatMap { line =>
        parseOpt(line)
      }.flatMap { jval =>
        (jval \ "business_id").extractOpt[String]
      }.cache()

      // ---------------------------
      // DEFAULT PARTITION ANALYSIS
      // ---------------------------
      val defaultStart = System.currentTimeMillis()

      // We simply reuse the RDD as-is, which already has 20 partitions
      val defaultTop10 = businessRDD
        .map(bid => (bid, 1))
        .reduceByKey(_ + _)
        .takeOrdered(10)(Ordering.by { case (id, count) => (-count, id) })

      val defaultTimeSecs =
        (System.currentTimeMillis() - defaultStart) / 1000.0

      val defaultNumPartitions = businessRDD.getNumPartitions
      val defaultPartitionSizes = businessRDD
        .mapPartitions(iter => Iterator(iter.size))
        .collect()

      // ---------------------------
      // CUSTOM PARTITION ANALYSIS
      // ---------------------------
      val customStart = System.currentTimeMillis()

      // Partition by a HashPartitioner using user-specified numPartitions
      val partitionedRDD = businessRDD
        .map(bid => (bid, 1))
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()

      val customTop10 = partitionedRDD
        .reduceByKey(_ + _)
        .takeOrdered(10)(Ordering.by { case (id, count) => (-count, id) })

      val customTimeSecs =
        (System.currentTimeMillis() - customStart) / 1000.0

      val customNumPartitions = partitionedRDD.getNumPartitions
      val customPartitionSizes = partitionedRDD
        .mapPartitions(iter => Iterator(iter.size))
        .collect()

      // ---------------------------
      // Build JSON Output
      // ---------------------------
      val jsonOutput =
        s"""
           |{
           |  "default": {
           |    "n_partition": $defaultNumPartitions,
           |    "n_items": ${defaultPartitionSizes.mkString("[", ", ", "]")},
           |    "exe_time": $defaultTimeSecs
           |  },
           |  "customized": {
           |    "n_partition": $customNumPartitions,
           |    "n_items": ${customPartitionSizes.mkString("[", ", ", "]")},
           |    "exe_time": $customTimeSecs
           |  }
           |}
           |""".stripMargin

      val writer = new PrintWriter(new File(outputPath))
      writer.write(jsonOutput)
      writer.close()

    } finally {
      sc.stop()
    }
  }
}
