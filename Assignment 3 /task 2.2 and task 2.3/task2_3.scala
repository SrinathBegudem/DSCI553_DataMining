

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

import org.json4s._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy

import scala.collection.mutable
import java.io.{File, PrintWriter}
import scala.math._

object task2_3 {

  // ---------------------------------------------------------
  // 1) Reading CSV
  // ---------------------------------------------------------
  def readCsv(sc: SparkContext, path: String, hasRating: Boolean): RDD[Array[String]] = {
    val raw    = sc.textFile(path)
    val header = raw.first()
    val lines  = raw.filter(_ != header).map(_.split(","))

    if (hasRating) lines.filter(_.length >= 3)
    else           lines.filter(_.length >= 2)
  }

  // ---------------------------------------------------------
  // 2) Reading JSON
  // ---------------------------------------------------------
  def loadJson[K,V](sc: SparkContext, path: String)(mapFn: JValue => (K,V)): RDD[(K,V)] = {
    implicit val fm = DefaultFormats
    sc.textFile(path).map { line =>
      val jval = parse(line)
      mapFn(jval)
    }
  }

  // ---------------------------------------------------------
  // 3) CF Data
  // ---------------------------------------------------------
  case class CFData(
    busToUsers: Map[String, Set[String]],
    userToBus:  Map[String, Set[String]],
    busAvg:     Map[String, Double],
    userAvg:    Map[String, Double],
    busRates:   Map[String, Map[String,Double]],
    globalAvg:  Double
  )

  /** Builds CF data from yelp_train.csv. */
  def buildCFData(sc: SparkContext, trainRdd: RDD[Array[String]]): CFData = {
    val data = trainRdd.map(arr => (arr(0), arr(1), arr(2).toDouble)).cache()

    // business -> set(users)
    val busUsers = data.map{ case(u,b,r)=>(b,u) }
      .groupByKey().mapValues(_.toSet)
      .collectAsMap()

    // user -> set(businesses)
    val userBus  = data.map{ case(u,b,r)=>(u,b) }
      .groupByKey().mapValues(_.toSet)
      .collectAsMap()

    // business -> average rating
    val busAvg = data.map{ case(u,b,r)=>(b,(r,1)) }
      .reduceByKey{ case((sum1,c1),(sum2,c2))=>(sum1+sum2,c1+c2) }
      .mapValues{ case(s,c)=> s/c }
      .collectAsMap()

    // user -> average rating
    val userAvg= data.map{ case(u,b,r)=>(u,(r,1)) }
      .reduceByKey{ case((s1,c1),(s2,c2))=>(s1+s2,c1+c2) }
      .mapValues{ case(s,c)=> s/c }
      .collectAsMap()

    // business -> (user->rating)
    val busRates= data.map{ case(u,b,r)=>(b,(u,r)) }
      .groupByKey()
      .mapValues{ items =>
        val mm= mutable.Map[String,Double]()
        items.foreach{ case(ux,rx)=>
          mm(ux)=rx
        }
        mm.toMap
      }
      .collectAsMap()

    // global average
    val global= data.map(_._3).mean()

    CFData(
      busUsers.toMap,
      userBus.toMap,
      busAvg.toMap,
      userAvg.toMap,
      busRates.toMap,
      global
    )
  }

  // ---------------------------------------------------------
  // 4) Pearson Similarity + CF
  // ---------------------------------------------------------
  def computeSimilarity(
    b1:String,
    b2:String,
    cf: CFData,
    cache: mutable.Map[(String,String),Double]
  ): Double = {
    val (k1,k2)= if(b1<b2)(b1,b2) else(b2,b1)
    if(cache.contains((k1,k2))) return cache((k1,k2))

    // missing in training
    if(!cf.busToUsers.contains(b1) || !cf.busToUsers.contains(b2)){
      cache((k1,k2))=0.0
      return 0.0
    }
    val common = cf.busToUsers(b1).intersect(cf.busToUsers(b2))
    if(common.size<=1){
      // fallback
      val avg1= cf.busAvg.getOrElse(b1, cf.globalAvg)
      val avg2= cf.busAvg.getOrElse(b2, cf.globalAvg)
      val fallback= (5.0 - (avg1-avg2).abs)/5.0*0.5
      cache((k1,k2))=fallback
      return fallback
    }

    val map1= cf.busRates(b1)
    val map2= cf.busRates(b2)
    val r1= mutable.ListBuffer[Double]()
    val r2= mutable.ListBuffer[Double]()

    common.foreach{ user =>
      if(map1.contains(user) && map2.contains(user)){
        r1+= map1(user)
        r2+= map2(user)
      }
    }
    if(r1.size<=1){
      val avg1= cf.busAvg.getOrElse(b1, cf.globalAvg)
      val avg2= cf.busAvg.getOrElse(b2, cf.globalAvg)
      val fb= (5.0- (avg1-avg2).abs)/5.0*0.5
      cache((k1,k2))=fb
      return fb
    }

    val m1= r1.sum/r1.size
    val m2= r2.sum/r2.size
    val c1= r1.map(_-m1)
    val c2= r2.map(_-m2)
    val num= c1.zip(c2).map{ case(a,b)=> a*b}.sum
    val den= math.sqrt(c1.map(a=>a*a).sum)* math.sqrt(c2.map(b=>b*b).sum)
    if(den==0.0){
      cache((k1,k2))=0.0
      return 0.0
    }
    var sim= num/den
    // significance weighting
    val factor= math.min(1.0, r1.size/30.0)
    sim*= factor

    cache((k1,k2))= sim
    sim
  }

  /** Increase neighborCount => 12. */
  def predictCF(
    userId:String,
    busId:String,
    cf: CFData,
    cache: mutable.Map[(String,String),Double],
    neighborCount:Int=12
  ): (Double,Double) = {
    val hasUser= cf.userToBus.contains(userId)
    val hasBus = cf.busToUsers.contains(busId)
    if(!hasUser && !hasBus){
      return(cf.globalAvg,0.05)
    }
    if(!hasUser){
      val ba= cf.busAvg.getOrElse(busId, cf.globalAvg)
      return(ba,0.1)
    }
    if(!hasBus){
      val ua= cf.userAvg.getOrElse(userId, cf.globalAvg)
      return(ua,0.1)
    }
    val rated= cf.userToBus(userId)
    val neighbors= mutable.ListBuffer[(Double,Double)]()
    for(rb <- rated if rb!=busId){
      val s= computeSimilarity(busId,rb, cf, cache)
      if(s>0.0){
        val rating= cf.busRates(rb).getOrElse(userId, cf.globalAvg)
        neighbors+= ((s,rating))
      }
    }
    if(neighbors.isEmpty){
      val ua= cf.userAvg.getOrElse(userId, cf.globalAvg)
      return(ua,0.1)
    }
    val top= neighbors.sortBy(-_._1).take(neighborCount)
    val num= top.map{ case(s,r)=> s*r }.sum
    val den= top.map{ case(s,_)=>(math.abs(s)) }.sum
    if(den==0.0){
      val ua= cf.userAvg.getOrElse(userId, cf.globalAvg)
      return(ua,0.1)
    }
    val raw= num/den
    val clipped= math.max(1.0, math.min(5.0, raw))

    // confidence
    val cF= top.size.toDouble/neighborCount
    val simSum= top.map(_._1).sum
    val avgSim= if(top.nonEmpty) simSum/top.size else 0.0
    val conf= math.min(0.6, cF*avgSim)
    (clipped, conf)
  }

  // ---------------------------------------------------------
  // 5) Model-based GBT (6D features)
  // ---------------------------------------------------------
  def buildFeature(
    userId:String,
    busId:String,
    busReview: Map[String,(Double,Double,Double,Double)],
    userMap:   Map[String,(Double,Double)],
    busMap:    Map[String,(Double,Double)],
    cf:CFData
  ): Array[Double] = {
    // busReview => (avgUseful, avgFunny, avgCool, #reviews)
    val (uF,fF,_,_)= busReview.getOrElse(busId,(0.0,0.0,0.0,0.0))
    val (uStars,uCount)= userMap.getOrElse(userId,(cf.userAvg.getOrElse(userId,cf.globalAvg), 0.0))
    val (bStars,bCount)= busMap.getOrElse(busId,(cf.busAvg.getOrElse(busId,cf.globalAvg), 0.0))
    Array[Double](
      uF, fF,
      uStars,
      bStars,
      uCount, bCount
    )
  }

  /** GBT with 20 trees, depth=3. */
  def trainGBT(train: RDD[LabeledPoint], iters:Int=20, depth:Int=3) = {
    val bs= BoostingStrategy.defaultParams("Regression")
    bs.numIterations= iters
    bs.treeStrategy.maxDepth= depth
    GradientBoostedTrees.train(train, bs)
  }

  // ---------------------------------------------------------
  // 6) Hybrid Weighted
  // ---------------------------------------------------------
  /** Weighted average => alpha in [0.05..0.3], start from ~0.15*cfConf. */
  def computeAlpha(userId:String, busId:String, cfConf:Double, cf:CFData): Double = {
    var w= 0.15* cfConf
    // user rating count
    if(cf.userToBus.contains(userId)){
      val c= cf.userToBus(userId).size
      if(c>30) w+=0.05
      else if(c<5) w-=0.03
    } else {
      w=0.05
    }
    // business rating count
    if(cf.busToUsers.contains(busId)){
      val p= cf.busToUsers(busId).size
      if(p>50) w+=0.05
      else if(p<10) w-=0.03
    } else {
      w=0.05
    }
    math.max(0.05, math.min(0.3, w))
  }

  // ---------------------------------------------------------
  // MAIN
  // ---------------------------------------------------------
  def main(args:Array[String]):Unit={
    if(args.length!=3){
      println("Usage: spark-submit --class task2_3 <jar> <folder> <testFile> <outputFile>")
      sys.exit(1)
    }

    val folder= args(0)
    val testF = args(1)
    val outF  = args(2)

    val conf= new SparkConf().setAppName("Task2_3_Iterations20_Neighbor12")
    val sc  = new SparkContext(conf)
    sc.setLogLevel("ERROR")

    val start= System.currentTimeMillis()

    // 1) load train
    val trainPath= s"$folder/yelp_train.csv"
    val trainRdd= readCsv(sc, trainPath, hasRating=true).cache()
    val cfData= buildCFData(sc, trainRdd)
    val simCache= mutable.Map[(String,String),Double]()

    // 2) test
    val testRdd= readCsv(sc, testF, hasRating=false)
    val testPairs= testRdd.map(a=>(a(0), a(1))).collect()

    // 3) JSON feats
    // busReview => (useful,funny,cool,#)
    val pathRev= s"$folder/review_train.json"
    val revRDD= loadJson(sc, pathRev){ jval =>
      implicit val fm= DefaultFormats
      val bid= (jval \ "business_id").extract[String]
      val uf= (jval \ "useful").extract[Double]
      val ff= (jval \ "funny").extract[Double]
      val cl= (jval \ "cool").extract[Double]
      (bid,(uf,ff,cl))
    }
    val busReviewFeats= revRDD.groupByKey().mapValues{ items =>
      val n= items.size.toDouble
      var su=0.0; var sf=0.0; var sc=0.0
      items.foreach{ case(uF,fF,cF)=>
        su+=uF; sf+=fF; sc+=cF
      }
      // (avgUseful, avgFunny, avgCool, #reviews)
      (su/n, sf/n, sc/n, n)
    }.collectAsMap().toMap

    // user => (avgStars, reviewCount)
    val pathUser= s"$folder/user.json"
    val userRDD= loadJson(sc, pathUser){ jval =>
      implicit val fm= DefaultFormats
      val uid= (jval \ "user_id").extract[String]
      val ast= (jval \ "average_stars").extract[Double]
      val rct= (jval \ "review_count").extract[Double]
      (uid,(ast,rct))
    }
    val userFeats= userRDD.collectAsMap().toMap

    // business => (stars, reviewCount)
    val pathBus= s"$folder/business.json"
    val busRDD= loadJson(sc, pathBus){ jval =>
      implicit val fm= DefaultFormats
      val bid= (jval \ "business_id").extract[String]
      val st = (jval \ "stars").extract[Double]
      val rc = (jval \ "review_count").extract[Double]
      (bid,(st,rc))
    }
    val busFeats= busRDD.collectAsMap().toMap

    // 4) training for GBT
    val training= trainRdd.map{ arr =>
      val (u,b) = (arr(0), arr(1))
      val rating= arr(2).toDouble
      val feats= buildFeature(u,b,busReviewFeats,userFeats,busFeats, cfData)
      LabeledPoint(rating, Vectors.dense(feats))
    }.cache()

    // 20 trees, depth=3
    val gbtModel= trainGBT(training, iters=20, depth=3)

    // 5) model preds
    val modelPreds= testPairs.map{ case(u,b)=>
      val feats= buildFeature(u,b,busReviewFeats,userFeats,busFeats, cfData)
      val raw= gbtModel.predict(Vectors.dense(feats))
      math.max(1.0, math.min(5.0, raw))
    }

    // 6) CF preds
    val cfPreds= testPairs.map{ case(u,b)=>
      predictCF(u,b, cfData, simCache, neighborCount=12)
    }

    // 7) combine
    val combined= new Array[(String,String,Double)](testPairs.length)
    var i=0
    while(i< testPairs.length){
      val (uu,bb)= testPairs(i)
      val (cfVal,cfConf)= cfPreds(i)
      val mbVal= modelPreds(i)
      val alpha= computeAlpha(uu,bb, cfConf, cfData)
      val raw= alpha*cfVal + (1.0-alpha)*mbVal
      val clipped= math.max(1.0, math.min(5.0, raw))
      combined(i)= (uu,bb,clipped)
      i+=1
    }

    // 8) output
    val writer= new PrintWriter(new File(outF))
    try{
      writer.println("user_id,business_id,prediction")
      combined.foreach{ case(u,b,p)=>
        writer.println(s"$u,$b,$p")
      }
    } finally{
      writer.close()
    }

    // optional rmse snippet if test has rating
/*
    try {
      val testWithRating= readCsv(sc, testF, hasRating=true)
      val ratingMap= testWithRating.map(x=>( (x(0),x(1)), x(2).toDouble)).collectAsMap()
      var seSum=0.0
      var c=0L
      combined.foreach{ case(u,b,pr)=>
        val k= (u,b)
        if(ratingMap.contains(k)){
          val diff= pr - ratingMap(k)
          seSum+= diff*diff
          c+=1
        }
      }
      if(c>0){
        val rmse= math.sqrt(seSum/c)
        println(f"RMSE on test: $rmse%.4f ($c records have rating)")
      }
    } catch {
      case _:Throwable=>
    }
*/
    val dur= (System.currentTimeMillis()-start)/1000.0
    println(f"Task2_3 completed in $dur%.2f seconds.")
    sc.stop()
  }
}
