import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import scala.math._
import scala.io.Source

/**
 * Created by ZengJichuan on 2014/10/14.
 */
object Lasso {
//  def parseX(line:String) = {
//
//  }
  def loadY(file:String):Array[Double] = {
    for (line <- Source.fromFile(file).getLines.toArray) yield line.split("\\s+")(2).toDouble
  }

  /**
   * Compute covariance and ATy for each column feature
   * @param p column feature
   * @return (covariance, Aty)
   */
//  def computeCov(p:LabeledPoint):(Double, Double) = {
//    val feat = p.features
//    feat.toArray.map()
//  }
  def main(args: Array[String]){
    val spark = new SparkContext(args(0), "Spark Pi")
    val slices = if (args.length > 1) args(1).toInt else 2
  /**
    * NOTICE: We first use MLUtils.loadLibSVMFile to load matrix X in the form of X = [x_1, x_2,..., x_p]T.
    *         LabelPoint in here means [Column Id, Column feature], And we store y separately.
   *          But we found that the org.apache.spark.millib.linalg.Vector has few vector operations, So we
   *          try breeze.linalg.Vector.
    */
    print("Loading matrix X from ... ")
    val X = MLUtils.loadLibSVMFile(spark, "data/covtype.test").cache()
    println("done")
    println(s"X contains ${X.count()} non-zero columns")
    print("Loading vector y from ... ")
    val y = loadY("data/covtype.y")
    println("done")
    println("Initializing features...")
//    val covers = X.map(p => computeCov(p))
//    // Initialize convergence step parameters
//    var counter = 0
//    val lambdaMin = lambda
//    val lambdaMax =
//    val regPathLength = regPathLength
//    val alpha = pow(lambdaMax/lambdaMin, 1.0/(1.0*regPathLength))
//    var regPathStep = regPathLength
//
//    println("Performing shot")
//    for(i <- 1 to iter) {
//      val actlambda = lambdaMin * pow(alpha, regPathStep)
//      for (j <- 1 to np / slices) {
//        X.takeSample(false, slices).map(p => shoot(p))
//      }
//    }
    println (X.map(p => p.label).reduce(_+_))//.foreach(x:LabeledPoint => ).take(10).foreach(println)
    println(X.takeSample(false, 5).foreach(p=> println(p.features*p.features)))

    spark.stop()
  }
}
