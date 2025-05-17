import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.util.Try
import java.io.PrintWriter

object task1 {

  // Hash function similar to Python's xor_hash:
  def xorHash(a: String, b: String, numBuckets: Int): Int = {
    val aa = Try(a.toInt).getOrElse(a.hashCode)
    val bb = Try(b.toInt).getOrElse(b.hashCode)
    ((aa ^ bb).abs) % numBuckets
  }

  // Build baskets for Case 1: user -> set(businesses)
  def buildBasketsCase1(rdd: org.apache.spark.rdd.RDD[Array[String]]): org.apache.spark.rdd.RDD[Set[String]] = {
    rdd.map(row => (row(0), row(1)))
       .groupByKey()
       .map { case (_, items) => items.toSet }
  }

  // Build baskets for Case 2: business -> set(users)
  def buildBasketsCase2(rdd: org.apache.spark.rdd.RDD[Array[String]]): org.apache.spark.rdd.RDD[Set[String]] = {
    rdd.map(row => (row(1), row(0)))
       .groupByKey()
       .map { case (_, users) => users.toSet }
  }

  // Optimized A-Priori candidate generation using prefix grouping.
  def aprioriGen(prevFreq: List[Seq[String]], k: Int): List[Seq[String]] = {
    // Group (k-1)-itemsets by their first k-2 items.
    val grouped = prevFreq.groupBy(itemset => itemset.take(k - 2))
    val candidates = ArrayBuffer[Seq[String]]()
    val prevFreqSet = prevFreq.toSet
    for ((prefix, group) <- grouped) {
      if (group.size >= 2) {
        // Sort the group lexicographically using mkString.
        val sortedGroup = group.sortBy(x => x.mkString(","))
        for (i <- sortedGroup.indices; j <- i + 1 until sortedGroup.size) {
          val candidate = (sortedGroup(i) ++ Seq(sortedGroup(j).last)).sorted
          // Prune candidate: every (k-1)-subset must be frequent.
          if (candidate.combinations(k - 1).forall(subset => prevFreqSet.contains(subset))) {
            candidates += candidate
          }
        }
      }
    }
    candidates.toList
  }

  // Local PCY pass (SON Phase 1) on a partition.
  // Returns an iterator of (candidate itemset, 1)
  def localPcy(partBaskets: Iterator[Set[String]], totalCount: Long, globalSupport: Double, numBuckets: Int = 1000): Iterator[(Seq[String], Int)] = {
    val baskets = partBaskets.toList
    val partSize = baskets.size
    if (partSize == 0) return Iterator.empty

    // Local support threshold proportional to the partition.
    val fraction = partSize.toDouble / totalCount.toDouble
    val localThreshFloat = globalSupport * fraction
    val localThreshold = if (localThreshFloat < 1) 1 else math.ceil(localThreshFloat).toInt

    // Count singletons and hash pairs.
    val itemCount = new HashMap[String, Int]()
    val hashCount = new HashMap[Int, Int]()
    baskets.foreach { basket =>
      val basketList = basket.toList
      basketList.foreach { item =>
        itemCount(item) = itemCount.getOrElse(item, 0) + 1
      }
      for (i <- 0 until basketList.length - 1) {
        for (j <- i + 1 until basketList.length) {
          val idx = xorHash(basketList(i), basketList(j), numBuckets)
          hashCount(idx) = hashCount.getOrElse(idx, 0) + 1
        }
      }
    }

    // Build a bitmap from hash counts.
    val bitMap = Array.fill(numBuckets)(0)
    hashCount.foreach { case (idx, count) =>
      if (count >= localThreshold) bitMap(idx) = 1
    }

    // Frequent singletons.
    val freqSingles = itemCount.filter { case (_, cnt) => cnt >= localThreshold }.keys.toList.sorted
    val localCandidates = ArrayBuffer[(Seq[String], Int)]()
    freqSingles.foreach(item => localCandidates += ((Seq(item), 1)))

    // Candidate pairs from frequent singletons.
    val pairCandidates = ArrayBuffer[(String, String)]()
    for (i <- freqSingles.indices; j <- i + 1 until freqSingles.length)
      pairCandidates += ((freqSingles(i), freqSingles(j)))
    val goodPairs = pairCandidates.filter { case (a, b) =>
      val idx = xorHash(a, b, numBuckets)
      bitMap(idx) == 1
    }
    if (goodPairs.isEmpty) return localCandidates.iterator

    // Remove non-frequent singles from baskets.
    val freqSinglesSet = freqSingles.toSet
    val updatedBaskets = baskets.map(b => b.filter(freqSinglesSet.contains).toList)

    // Exact counting for candidate pairs.
    val pairCount = new HashMap[(String, String), Int]()
    goodPairs.foreach(pair => pairCount(pair) = 0)
    updatedBaskets.foreach { bList =>
      val bSet = bList.toSet
      pairCount.keys.foreach { case pair =>
        if (bSet.contains(pair._1) && bSet.contains(pair._2))
          pairCount(pair) = pairCount(pair) + 1
      }
    }
    val freqPairs = goodPairs.filter(pair => pairCount(pair) >= localThreshold)
    freqPairs.foreach { pair =>
      localCandidates += ((Seq(pair._1, pair._2).sorted, 1))
    }
    if (freqPairs.isEmpty) return localCandidates.iterator

    // For k >= 3, use the optimized aprioriGen.
    var currentSets: List[Seq[String]] = freqPairs.map { case (a, b) => Seq(a, b) }.toList
    var k = 3
    while (currentSets.nonEmpty) {
      val candidateK = aprioriGen(currentSets, k)
      if (candidateK.isEmpty) {
        currentSets = Nil
      } else {
        val candCount = new HashMap[Seq[String], Int]()
        candidateK.foreach(c => candCount(c) = 0)
        updatedBaskets.foreach { bList =>
          val bSet = bList.toSet
          candidateK.foreach { candidate =>
            if (candidate.toSet.subsetOf(bSet))
              candCount(candidate) = candCount(candidate) + 1
          }
        }
        val freqNew = candCount.filter { case (_, cnt) => cnt >= localThreshold }.keys.toList
        freqNew.foreach(c => localCandidates += ((c, 1)))
        currentSets = freqNew
        k += 1
      }
    }

    localCandidates.iterator
  }

  // SON Phase 2: Global candidate counting.
  def pass2Count(basketsIter: Iterator[Set[String]], candList: Seq[Seq[String]]): Iterator[(Seq[String], Int)] = {
    val outMap = new HashMap[Seq[String], Int]()
    basketsIter.foreach { basket =>
      val bSet = basket  // already a set
      candList.foreach { candidate =>
        if (candidate.toSet.subsetOf(bSet))
          outMap(candidate) = outMap.getOrElse(candidate, 0) + 1
      }
    }
    outMap.iterator
  }

  // Group and sort itemsets by size.
  def groupAndSortItemsets(itemsets: Seq[Seq[String]]): Seq[(Int, Seq[Seq[String]])] = {
    val groupMap = itemsets.groupBy(_.length)
    groupMap.toSeq.sortBy(_._1).map { case (size, items) =>
      (size, items.sortBy(itemset => itemset.mkString(",")))
    }
  }

  // Format an itemset as ("'a'", "'b'", ...)
  def formatItemset(itup: Seq[String]): String = {
    "(" + itup.map(item => s"'$item'").mkString(", ") + ")"
  }

  // Write out results.
  def writeResults(candGrouped: Seq[(Int, Seq[Seq[String]])],
                   freqGrouped: Seq[(Int, Seq[Seq[String]])],
                   outFile: String): Unit = {
    val writer = new PrintWriter(outFile)
    try {
      writer.println("Candidates:")
      candGrouped.foreach { case (_, candidates) =>
        writer.println(candidates.map(formatItemset).mkString(","))
        writer.println()
      }
      writer.println("Frequent Itemsets:")
      freqGrouped.foreach { case (_, freqItems) =>
        writer.println(freqItems.map(formatItemset).mkString(","))
        writer.println()
      }
    } finally {
      writer.close()
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length != 4) {
      println("Usage: spark-submit --class task1 <case_number> <support> <input_file> <output_file>")
      sys.exit(1)
    }

    val caseNum = args(0).toInt
    val support = args(1).toDouble
    val inputFile = args(2)
    val outputFile = args(3)

    val conf = new SparkConf().setAppName("Task1SON")
      .set("spark.executor.memory", "4g")
      .set("spark.driver.memory", "4g")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val startT = System.nanoTime()

    // Build baskets and cache them.
    val lines = sc.textFile(inputFile)
    val header = lines.first()
    val dataRdd = lines.filter(line => line != header).map(line => line.trim.split(","))
    val basketsRdd = if (caseNum == 1) buildBasketsCase1(dataRdd) else buildBasketsCase2(dataRdd)
    basketsRdd.cache()
    val totalCount = basketsRdd.count()

    // SON Phase 1: Local candidate generation.
    val localCandidates = basketsRdd
      .mapPartitions(part => localPcy(part, totalCount, support, 1000))
      .reduceByKey(_ + _)
      .map(_._1)
      .distinct()
      .collect()
      .toSeq

    val candGrouped = groupAndSortItemsets(localCandidates)

    // Broadcast candidate list for SON Phase 2.
    val bcCandList = sc.broadcast(localCandidates)

    // SON Phase 2: Global count.
    val freqCounts = basketsRdd
      .mapPartitions(part => pass2Count(part, bcCandList.value))
      .reduceByKey(_ + _)
      .filter { case (_, count) => count >= support }
      .map(_._1)
      .collect()
      .toSeq

    val freqGrouped = groupAndSortItemsets(freqCounts)

    writeResults(candGrouped, freqGrouped, outputFile)

    val endT = System.nanoTime()
    println(s"Duration: ${(endT - startT) / 1e9}")
    sc.stop()
  }
}
