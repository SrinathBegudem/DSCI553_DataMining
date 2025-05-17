import org.apache.spark.{SparkConf, SparkContext, Partitioner}
import scala.collection.mutable.{HashMap, ListBuffer}
import java.io.PrintWriter

object task2 {
  // Custom Implicit Ordering for Seq[String]
  implicit val seqStringOrdering: Ordering[Seq[String]] = new Ordering[Seq[String]] {
    def compare(x: Seq[String], y: Seq[String]): Int = {
      val xStr = x.map(_.toString)
      val yStr = y.map(_.toString)
      Ordering.Iterable[String].compare(xStr, yStr)
    }
  }

  // Hash Function
  def customHash(x: String, y: String, nBuckets: Int): Int = {
    val xx = try { x.toInt } catch { case _: NumberFormatException => x.hashCode }
    val yy = try { y.toInt } catch { case _: NumberFormatException => y.hashCode }
    ((xx ^ yy).abs) % nBuckets
  }

  // Custom Partitioner
  class CustomPartitioner(numParts: Int) extends Partitioner {
    override def numPartitions: Int = numParts
    override def getPartition(key: Any): Int = {
      val k = key.asInstanceOf[String]
      (k.length + scala.util.Random.nextInt(1001)) % numParts
    }
  }

  // Local Candidate Generation with PCY
  def localCandidateGeneration(partitionBaskets: Iterator[Seq[String]], totalCount: Long, globalSupport: Double, numBuckets: Int = 1000): Iterator[(Seq[String], Int)] = {
    val baskets = partitionBaskets.toList
    val partSize = baskets.length
    if (partSize == 0) return Iterator.empty

    val p = partSize.toDouble / totalCount.toDouble
    val localThresh = Math.max(1.0, p * globalSupport)

    val itemCount = new HashMap[String, Int]()
    val hashCount = new HashMap[Int, Int]()
    
    baskets.foreach { basket =>
      basket.foreach { item =>
        itemCount(item) = itemCount.getOrElse(item, 0) + 1
      }
      for (i <- 0 until basket.length - 1; j <- i + 1 until basket.length) {
        val idx = customHash(basket(i), basket(j), numBuckets)
        hashCount(idx) = hashCount.getOrElse(idx, 0) + 1
      }
    }

    val bitmap = Array.fill(numBuckets)(0)
    hashCount.foreach { case (idx, count) =>
      if (count >= localThresh) bitmap(idx) = 1
    }

    val singleFreq = itemCount.filter(_._2 >= localThresh).keys.toList.sorted
    val localCandidates = ListBuffer[(Seq[String], Int)]()
    singleFreq.foreach { s => localCandidates += ((Seq(s), 1)) }

    val pairCandidates = for {
      i <- 0 until singleFreq.length
      j <- i + 1 until singleFreq.length
    } yield (singleFreq(i), singleFreq(j))

    val goodPairs = pairCandidates.filter { case (a, b) =>
      bitmap(customHash(a, b, numBuckets)) == 1
    }.toList

    if (goodPairs.isEmpty) return localCandidates.iterator

    val freqSinglesSet = singleFreq.toSet
    val prunedBaskets = baskets.map(b => b.filter(freqSinglesSet.contains))

    val pairCount = HashMap(goodPairs.map(p => (p, 0)): _*)
    prunedBaskets.foreach { b =>
      val bSet = b.toSet
      pairCount.foreach { case (p, _) =>
        if (bSet.contains(p._1) && bSet.contains(p._2)) pairCount(p) += 1
      }
    }

    val freqPairs = pairCount.filter(_._2 >= localThresh).keys.toList
    freqPairs.foreach { p => localCandidates += ((Seq(p._1, p._2).sorted, 1)) }
    if (freqPairs.isEmpty) return localCandidates.iterator

    var currentSets = freqPairs
    var k = 3
    while (currentSets.nonEmpty) {
      val allItems = currentSets.flatMap(p => Seq(p._1, p._2)).toSet.toList.sorted
      val candsK = allItems.combinations(k).toList
      if (candsK.isEmpty) return localCandidates.iterator

      val candCount = HashMap(candsK.map(c => (c, 0)): _*)
      prunedBaskets.foreach { b =>
        val bSet = b.toSet
        candCount.foreach { case (c, _) =>
          if (c.forall(bSet.contains)) candCount(c) += 1
        }
      }

      val freqNew = candCount.filter(_._2 >= localThresh).keys.toList
      freqNew.foreach { c => localCandidates += ((c, 1)) }
      currentSets = freqNew.map(c => (c(0), c(1)))
      k += 1
    }

    localCandidates.iterator
  }

  // SON Count Stage
  def sonCountStage(basketsIter: Iterator[Seq[String]], candidateItemsets: Seq[Seq[String]]): Iterator[(Seq[String], Int)] = {
    val baskets = basketsIter.toList
    val outMap = new HashMap[Seq[String], Int]()
    baskets.foreach { b =>
      val bSet = b.toSet
      candidateItemsets.foreach { c =>
        if (c.forall(bSet.contains)) outMap(c) = outMap.getOrElse(c, 0) + 1
      }
    }
    outMap.iterator
  }

  // Output Formatting
  def groupAndFormatForOutput(itemsList: Seq[Seq[String]]): String = {
    val uniqueItemsets = itemsList.map(_.sorted).distinct.sortBy(x => (x.length, x.map(_.toString)))
    val groups = uniqueItemsets.groupBy(_.length)
    val resultLines = groups.toSeq.sortBy(_._1).map { case (size, itemsets) =>
      val formatted = itemsets.map { itemset =>
        "(" + itemset.map(x => s"'$x'").mkString(", ") + ")"
      }.mkString(",")
      formatted
    }
    resultLines.mkString("\n\n")
  }

  // Main
  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      println("Usage: spark-submit --class task2 hw2.jar <filter_threshold> <support> <input_file> <output_file>")
      sys.exit(1)
    }

    val filterThreshold = args(0).toInt
    val support = args(1).toDouble
    val inputFile = args(2)
    val outputFile = args(3)

    val conf = new SparkConf()
      .setAppName("task2_snr_style")
      .set("spark.executor.memory", "4g")
      .set("spark.driver.memory", "4g")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val startT = System.nanoTime()

    // 1) Read the input file
    val lines = sc.textFile(inputFile)
    val header = lines.first()
    val dataRdd = lines.filter(_ != header).map(_.split(","))

    // 2) Pre-process into (date-customer, product) pairs
    def transform(row: Array[String]): (String, String) = {
      val dateCustomer = row(0).substring(1, row(0).length - 5) + row(0).substring(row(0).length - 3, row(0).length - 1) + "-" + row(1).substring(1, row(1).length - 1).toInt.toString
      val productId = row(5).substring(1, row(5).length - 1).toLong.toString
      (dateCustomer, productId)
    }
    val processedRdd = dataRdd.map(transform)

    // 3) Build baskets and filter by threshold
    def groupProd(rdd: org.apache.spark.rdd.RDD[(String, String)], k: Int): org.apache.spark.rdd.RDD[Seq[String]] = {
      rdd.groupByKey()
         .mapValues(_.toSet.toSeq)
         .filter(_._2.length > k)
         .map(_._2)
    }
    val basketsRdd = groupProd(processedRdd, filterThreshold)
    val totalBaskets = basketsRdd.count()

    // 4) Partition the data
    val nPartitions = 10
    val keyedBaskets = basketsRdd.map(b => (b.mkString(","), b)).partitionBy(new CustomPartitioner(nPartitions))
    val baskets = keyedBaskets.values

    // 5) SON Stage 1: Generate local candidates
    val candidatesStage1 = baskets
      .mapPartitions(part => localCandidateGeneration(part, totalBaskets, support, 1000))
      .reduceByKey(_ + _)
      .collect()
      .toSeq
    val localCandItemsets = candidatesStage1.map(_._1.sorted).distinct.sortBy(x => (x.length, x))

    // 6) Format candidates
    val candidatesStr = groupAndFormatForOutput(localCandItemsets)

    // 7) SON Stage 2: Global counting
    val freqMap = baskets
      .mapPartitions(part => sonCountStage(part, localCandItemsets))
      .reduceByKey(_ + _)
      .filter(_._2 >= support.toInt)
      .collect()
      .toSeq
    val freqItemsets = freqMap.map(_._1.sorted).distinct.sortBy(x => (x.length, x))

    // 8) Format frequent itemsets
    val frequentStr = groupAndFormatForOutput(freqItemsets)

    // 9) Write output
    val outStr = "Candidates:\n" + candidatesStr + "\n\nFrequent Itemsets:\n" + frequentStr
    val writer = new PrintWriter(outputFile)
    try {
      writer.write(outStr)
    } finally {
      writer.close()
    }

    val endT = System.nanoTime()
    println(s"Duration: ${(endT - startT) / 1e9d}")
    sc.stop()
  }
}