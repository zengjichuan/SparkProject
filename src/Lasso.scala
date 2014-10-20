import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import scala.math._
import scala.io.Source
import scopt.OptionParser
import scala.util.control.Breaks._

import breeze.linalg._
/**
* Created by ZengJichuan on 2014/10/14.
*/
object Lasso {

  case class LassoSCD(
    colId:Int,
    A:SparseVector[Double],
    Aty:Double,
    Cov:Double)

  case class DataA(colId:Int, a: SparseVector[Double])

  class LassoProb(
    Ax_t:SparseVector[Double],
    x_t:SparseVector[Double]){
    var Ax:SparseVector[Double] = Ax_t
    var x:SparseVector[Double] = x_t
  }
  /**
   * parse data file A
   * @param file input file name
   * @param nm num of samples
   * @return  array of DataX
   */
  def loadA(file:String, nm:Int): Array[DataA] = {
    val parsed = Source.fromFile(file).getLines().toArray
      .map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map {line =>
        val items = line.split("\\s+")
        val colId = items.head.toInt
        val (indices, values) = items.tail.map { item =>
          val indexAndValue = item.split(":")
          val index = indexAndValue(0).toInt    //already 0-based
          val value = indexAndValue(1).toDouble
          (index, value)
        }.unzip
        (colId, indices.toArray, values.toArray)
      }
    parsed.map{case (colId, indices, values) => DataA(colId, new SparseVector(indices, values, nm))}
  }

  /**
   * Parse data file y
   * @param file input file name
   * @param np number of features
   * @return
   */
  def loadY(file:String, np:Int):SparseVector[Double] = {
    //    for (line <- Source.fromFile(file).getLines.toArray) yield line.split("\\s+")(2).toDouble
    val (indices, values) = Source.fromFile(file).getLines().toArray.map(_.trim)
      .filter(line => !(line.isEmpty || line.startsWith("#")))
      .map { line =>
      val items = line.split("\\s+")
      val index = items(0).toInt
      val value = items(2).toDouble
      (index, value)
    }.unzip
    new SparseVector[Double](indices.toArray, values.toArray, np)
  }

  /**
   * Compute covariance and ATy for each column feature, and return a RDD
   * @param sc
   * @param loadA
   * @param loadY
   * @param numSlice
   * @return
   */
  def initAndGetLassoRDD(sc: SparkContext, loadA:Array[DataA], loadY:SparseVector[Double], numSlice:Int):RDD[LassoSCD] = {
    val colInfo = loadA.map{dataA =>
      val Aty = dataA.a.t*loadY*2
      val Cov = pow(dataA.a.norm(2),2)*2
      LassoSCD(dataA.colId, dataA.a, Aty, Cov)
    }
    sc.parallelize(colInfo, numSlice)
  }

  def getTermThreshold(regPathStep:Int, regPathLen:Int, threshold:Double) = {
    if(k==0)
      threshold
    else
      threshold + regPathStep * (threshold * 50)/regPathLen
  }
  def softThreshold(lambda:Double, value:Double):Double = {
    if (value > lambda)
      lambda - value
    if (value < -lambda)
      -lambda -value
    else
      0
  }
  def shoot(lassoCol:LassoSCD, lambda:Double, lassoProb:LassoProb):(Int,Double,SparseVector[Double]) = {
    val colId = lassoCol.colId
    val oldx = lassoProb.x(colId)
    val AtAxj = lassoCol.A.t*lassoProb.Ax
    val S_j = 2*AtAxj - lassoCol.Cov*oldx-lassoCol.Aty
    val newx = softThreshold(lambda, S_j)/lassoCol.Cov
    val delta = newx - oldx
    (colId, delta, lassoCol.A*delta)
  }
  def computeObject(Ax:SparseVector[Double], x:SparseVector[Double], y:SparseVector[Double])
    :(Double,Double,Double,Double)={
    val l2err = pow((Ax-y).norm(2),2)
    val l1x = x.norm(1)
    val l0x = x.used
    val obj = lambda * l1x + l2err
    (obj, l2err, l1x, l0x)
  }
  case class Params(
    inputA:String = null,
    inputY:String = null,
    numIter:Int = 100,
    lambda:Double = 0.5,
    threshold:Double = 1e-5,
    regPathLen:Int = 0,
    nm:Int = 0,
    np:Int = 0,
    numSlice:Int = 2)

  def main(args: Array[String]) {
    val defaultParams = Params()
    val parser = new OptionParser[Params]("LASSO") {
      head(
        """
        | Implementation of Lasso Shotgun-algorithm from paper:
        | Parallel Coordinate Descent for L1-Regularized Loss Minimization.""".stripMargin)
      opt[String]("inputX").text("input matrix X file name").required()
        .action((x, c) => c.copy(inputX = x))
      opt[String]("inputY").text("input vector y file name").required()
        .action((x, c) =>c.copy(inputY = x))
      opt[Int]("numIter").text(s"iteration number, default: ${defaultParams.numIter}")
        .action((x, c) =>c.copy(numIter = x))
      opt[Double]("lambda").text(s"lambda, default: ${defaultParams.lambda}")
        .action((x, c) =>c.copy(lambda = x))
      opt[Double]("regPathLen").text(s"regularization path length, default: ${defaultParams.regPathLen}")
        .action((x, c) =>c.copy(regPathLen = x))
      opt[Double]("threshold").text(s"threshold, default: ${defaultParams.threshold}")
        .action((x, c) =>c.copy(threshold = x))
      opt[Int]("np").text("number of features").required()
        .action((x, c) =>c.copy(np = x))
      opt[Int]("nm").text("number of samples").required()
        .action((x, c) =>c.copy(nm = x))
      opt[Int]("numSlice").text(s"number of slice, defalut: ${defaultParams.numSlice}")
        .action((x, c) =>c.copy(numSlice = x))
    }
    parser.parse(args, defaultParams).map{params =>
      lassoRun(params)
    }getOrElse{
      System.exit(1)
    }
  }

  def lassoRun(params:Params){
    val spark = new SparkContext("local", s"Lasso with $params")
    /**
      * NOTICE: We first use MLUtils.loadLibSVMFile to load matrix X in the form of X = [x_1, x_2,..., x_p]T.
      *         LabelPoint in here means [Column Id, Column feature], And we store y separately.
      *          But we found that the org.apache.spark.millib.linalg.Vector has few vector operations, So we
      *          try breeze.linalg.Vector.
      */
    print("Loading matrix A from ... ")
//    val X = MLUtils.loadLibSVMFile(spark, "data/covtype.test").cache()
    val ALoad = loadA(params.inputA, params.nm)
    println("done")

//    println(s"X contains ${X.count()} non-zero columns")
    print("Loading vector y from ... ")
    val yLoad = loadY(params.inputY, params.np)
    println("done")
    println("Initializing features...")
    val LassoRDD = initAndGetLassoRDD(spark, ALoad, yLoad, params.numSlice).cache()
    println("=========== Start Lasso Stochastic Corrdinate Descent =============")
    val lassoProb = new LassoProb()
    lassoProb.Ax = SparseVector.zeros[Double](params.nm)
    lassoProb.x = SparseVector.zeros[Double](params.np)

    // Initialize convergence step parameters
    var counter = 0
    val lambdaMin = params.lambda
    val lambdaMax = LassoRDD.map(p => p.Aty).reduce(math.max)
    val regPathLen = params.regPathLen
    val alpha = pow(lambdaMax/lambdaMin, 1.0/(1.0*regPathLength))
    var regPathStep = regPathLen

    println("Performing shot")
    for(i <- 1 to iter) {
      val lambda = lambdaMin * pow(alpha, regPathStep)
      /**
       * In Spark implemetation, we can hardly change the x and Ax while in parallel processing. So we back to
       * original shotgun model, that is applying SCD in serval columns at a time, and then update x and Ax.
       */
      val maxChanges = for {
        j <- 1 to np / slices
        val (updateIndices, updateDeltas, updateAx) = LassoRDD
          .takeSample(false, slices).map(p => shoot(p, lambda)).array.unzip3
        val deltaSpV = new SparseVector[Double](updateIndices, updateDeltas, params.nm)
        lassoProb.x += deltaSpV
        lassoProb.Ax +=(updateAx.reduce(_+_))
        val maxDelta = updateDeltas.map(math.abs).reduce(math.max)
        // need test wether to broadcast
      } yield maxDelta
      //adjust convergence step
      val converged = (maxChanges.reduce(math.max) <= getTermThreshold(regPathStep, regPathLen, params.threshold))
      if(converged || counter>min(100, (100-regPathStep)*2)){
        counter = 0
        regPathStep -= 1
      }
      //output objective
      val (obj, l2err, l1x, l0x) = computeObject(lassoProb.Ax, lassoProb.x, yLoad)
      println(s"Objective: ${obj}  L1: ${l1x}  L2err: ${l2err}  l0: ${l0x}")
      if(regPathStep<0) break
    }
//    println (X.map(p => p.label).reduce(_+_))//.foreach(x:LabeledPoint => ).take(10).foreach(println)
//    println(X.takeSample(false, 5).foreach(p=> println(p.features*p.features)))

    spark.stop()
  }
}
