import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.collection.mutable.{Map => MMap, Set => MSet}
import scala.math.{sqrt, max}
import java.io.{File, PrintWriter}

object task {

  case class ClusterStats(n: Int, sum: Array[Double], sumSq: Array[Double])
  private val EPS = 1e-6

  private def mahalanobis(p: Array[Double],
                          c: Array[Double],
                          s: Array[Double]): Double = {
    var acc = 0.0
    var i   = 0
    while (i < p.length) {
      val z = (p(i) - c(i)) / max(s(i), EPS)
      acc += z * z
      i += 1
    }
    sqrt(acc)
  }

  private def updateStat(st: ClusterStats,
                         x: Array[Double]): ClusterStats = {
    val n2   = st.n + 1
    val sum2 = st.sum.zip(x).map { case (a, b) => a + b }
    val ss2  = st.sumSq.zip(x).map { case (a, b) => a + b * b }
    ClusterStats(n2, sum2, ss2)
  }

  private def mergeStat(a: ClusterStats,
                        b: ClusterStats): ClusterStats = {
    val n      = a.n + b.n
    val sum    = a.sum.zip(b.sum).map { case (x, y) => x + y }
    val sumSq  = a.sumSq.zip(b.sumSq).map { case (x, y) => x + y }
    ClusterStats(n, sum, sumSq)
  }

  private def centroid(st: ClusterStats): Array[Double] =
    st.sum.map(_ / st.n)

  private def stdev(st: ClusterStats): Array[Double] =
    st.sumSq.zip(st.sum).map { case (ss, s) =>
      sqrt(max(0.0, ss / st.n - (s / st.n) * (s / st.n)))
    }

  def main(args: Array[String]): Unit = {
    if (args.length != 3) {
      System.err.println("Usage: task <input> <k> <output>")
      sys.exit(1)
    }

    val conf = new SparkConf()
      .setAppName("BFR")
      .setIfMissing("spark.master", "local[*]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    // Read and parse
    val raw = sc.textFile(args(0))
    val parsed = raw.map { line =>
      val v  = line.split(',').map(_.trim.toDouble)
      val id = v(0).toInt
      val gt = v(1).toInt
      val fe = v.drop(2)
      (id, gt, fe)
    }.cache()

    // Ground‑truth map for O(1) lookup
    val truthMap: scala.collection.Map[Int, Int] =
      parsed.map { case (id, gt, _) => (id, gt) }
            .collectAsMap()

    val K       = args(1).toInt
    val bigK    = 5 * K
    val totalN  = parsed.count()
    val dim     = parsed.first()._3.length
    val thresh  = 2.0 * math.sqrt(dim)
    val outPath = args(2)
    val startMs = System.currentTimeMillis()

    // 5 random splits
    val chunks = parsed.randomSplit(Array.fill(5)(0.2), seed = 42).map(_.cache())

    // State
    val rs     = MSet[Int]()
    val dsStat = MMap[Int, ClusterStats]()
    val dsPts  = MMap[Int, MSet[Int]]()
    val csStat = MMap[Int, ClusterStats]()
    val csPts  = MMap[Int, MSet[Int]]()

    val writer = new PrintWriter(new File(outPath))
    writer.println("The intermediate results:")

    // ── ROUND 1 ──────────────────────────────────────────────────────────
    {
      // 20% chunk → bigK clusters
      val chunkVec = chunks(0).map(_._3).map(Vectors.dense)
      val k1       = new KMeans().setK(bigK).setSeed(42).setMaxIterations(20)
      val lab1     = k1.run(chunkVec).predict(chunkVec).collect()

      // singletons → RS
      val counts = lab1.groupBy(identity).mapValues(_.length)
      val single = counts.collect { case (c,1) => c }.toSet
      val dataArr= chunks(0).collect()
      rs ++= dataArr.zip(lab1).filter{case(_,c)=> single.contains(c)}.map(_._1._1)

      // remaining → K clusters → DS
      val remain = dataArr.zip(lab1).filter{case(_,c)=> !single.contains(c)}.map(_._1)
      val vec2   = sc.parallelize(remain.map(_._3).map(Vectors.dense))
      val k2     = new KMeans().setK(K).setSeed(42).setMaxIterations(20)
      val lab2   = k2.run(vec2).predict(vec2).collect()

      remain.zip(lab2).foreach { case ((id,_,fe), cid) =>
        val st0 = dsStat.getOrElse(cid, ClusterStats(0, Array.fill(dim)(0.0), Array.fill(dim)(0.0)))
        val st1 = updateStat(st0, fe)
        dsStat(cid) = st1
        dsPts.getOrElseUpdate(cid, MSet()) += id
      }

      // build CS out of RS if possible
      buildCS(rs.toSet, chunks(0), bigK, csStat, csPts, rs, dim)

      val dsc = dsStat.values.map(_.n).sum
      val csc = csStat.size
      val csp = csStat.values.map(_.n).sum
      writer.println(s"Round 1: $dsc,$csc,$csp,${rs.size}")
    }

    // ── ROUNDS 2‑5 ────────────────────────────────────────────────────────
    for (round <- 2 to 5) {
      val curRS = MSet[Int]()

      // assign new points to DS/CS or RS
      chunks(round-1).collect().foreach{case (id,_,fe) =>
        // DS
        var bestCid = -1
        var best    = Double.MaxValue
        dsStat.foreach{case(cid,st) =>
          val d = mahalanobis(fe, centroid(st), stdev(st))
          if (d < best) { best = d; bestCid = cid }
        }
        if (best < thresh) {
          dsStat(bestCid) = updateStat(dsStat(bestCid), fe)
          dsPts(bestCid)  += id
        } else {
          // CS
          bestCid = -1; best = Double.MaxValue
          csStat.foreach{case(cid,st) =>
            val d = mahalanobis(fe, centroid(st), stdev(st))
            if (d < best) { best = d; bestCid = cid }
          }
          if (best < thresh && bestCid != -1) {
            csStat(bestCid) = updateStat(csStat(bestCid), fe)
            csPts(bestCid)  += id
          } else {
            rs += id; curRS += id
          }
        }
      }

      // extend CS from current RS
      buildCS(curRS.toSet, chunks(round-1), bigK, csStat, csPts, rs, dim)

      // merge CS clusters if too close
      var merged = true
      while (merged && csStat.size > 1) {
        merged = false
        val ids = csStat.keys.toArray
        for {
          i <- ids.indices
          j <- i+1 until ids.length
          if csStat.contains(ids(i)) && csStat.contains(ids(j))
        } {
          val d = mahalanobis(
            centroid(csStat(ids(i))),
            centroid(csStat(ids(j))),
            stdev(csStat(ids(j)))
          )
          if (d < thresh) {
            val m = mergeStat(csStat(ids(i)), csStat(ids(j)))
            csStat(ids(j)) = m
            csPts(ids(j)) ++= csPts(ids(i))
            csStat.remove(ids(i)); csPts.remove(ids(i))
            merged = true
          }
        }
      }

      // on final round merge CS→DS
      if (round == 5) {
        val absorbed = MSet[Int]()
        csStat.keys.foreach{ cid =>
          var bestCid = -1
          var best    = Double.MaxValue
          dsStat.foreach{ case(did,ds) =>
            val d = mahalanobis(
              centroid(csStat(cid)),
              centroid(ds),
              stdev(ds)
            )
            if (d < best) { best = d; bestCid = did }
          }
          if (best < thresh) {
            val m = mergeStat(dsStat(bestCid), csStat(cid))
            dsStat(bestCid) = m
            dsPts(bestCid)  ++= csPts(cid)
            absorbed ++= csPts(cid)
            csStat.remove(cid); csPts.remove(cid)
          }
        }
      }

      val dsc = dsStat.values.map(_.n).sum
      val csc = csStat.size
      val csp = csStat.values.map(_.n).sum
      writer.println(s"Round $round: $dsc,$csc,$csp,${rs.size}")
    }

    // ── Final mapping & output ─────────────────────────────────────────────
    val result = MMap[Int,Int]()
    dsPts.foreach{ case(cid,ids) =>
      // majority vote against ground truth
      val freq = ids.groupBy(id => truthMap(id)).mapValues(_.size)
      val maj  = freq.maxBy(_._2)._1
      ids.foreach(pid => result(pid)=maj)
    }
    // CS & RS → -1
    csPts.values.flatten.foreach(pid => result(pid) = -1)
    rs.filterNot(result.keySet).foreach(pid => result(pid) = -1)
    // pad any ID missing
    parsed.map(_._1).collect().foreach(id => result.getOrElseUpdate(id,-1))

    writer.println()
    writer.println("The clustering results:")
    result.keys.toArray.sorted.foreach(id => writer.println(s"$id,${result(id)}"))
    writer.close()

    // ── metrics ────────────────────────────────────────────────────────────
    val good   = result.count{ case(id,lab) => lab != -1 && lab == truthMap(id) }
    val nonOut = parsed.filter{ case (_,gt,_) => gt != -1 }.count()
    val acc    = if (nonOut>0) good.toDouble/nonOut*100 else 0.0
    val dsPct  = dsStat.values.map(_.n).sum.toDouble/totalN*100
    val dur    = (System.currentTimeMillis() - startMs)/1000.0

    println(f"Duration:               $dur%.1f s")
    println(f"Discard‑set coverage:   $dsPct%.2f %%")
    println(f"Accuracy (non‑outliers):$acc%.2f %%")

    sc.stop()
  }

  // ── helper to build/extend CS from RS ids ─────────────────────────────
  private def buildCS(rsIds: Set[Int],
                      chunk   : org.apache.spark.rdd.RDD[(Int,Int,Array[Double])],
                      k       : Int,
                      csStat  : MMap[Int,ClusterStats],
                      csPts   : MMap[Int,MSet[Int]],
                      rsGlobal: MSet[Int],
                      dim     : Int): Unit = {

    if (rsIds.size < 3) return

    val pts = chunk.filter{ case(id,_,_) => rsIds.contains(id) }.collect()
    val feat = pts.map(_._3).map(Vectors.dense)
    val sc2  = chunk.context
    val km   = new KMeans().setK(math.min(k,feat.length)).setSeed(42).setMaxIterations(20)
    val mdl  = km.run(sc2.parallelize(feat))
    val lbls = mdl.predict(sc2.parallelize(feat)).collect()

    val counts  = lbls.groupBy(identity).mapValues(_.length)
    val single  = counts.collect{case(c,1)=>c}.toSet

    pts.zip(lbls).foreach{ case ((id,_,fe),cid) =>
      if (single.contains(cid)) {
        rsGlobal += id
      } else {
        val newCid = if (csStat.isEmpty) 0 else csStat.keys.max+1
        val st0    = csStat.getOrElse(newCid, ClusterStats(0, Array.fill(dim)(0.0), Array.fill(dim)(0.0)))
        csStat(newCid) = updateStat(st0, fe)
        csPts.getOrElseUpdate(newCid, MSet()) += id
        rsGlobal -= id
      }
    }
  }
}
