import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.SparkContext._
/**
 * Created by ZengJichuan on 2014/10/9.
 */

object WordCount {
  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
  def main(args: Array[String]) {
    if (args.length == 0) {
      System.err.println("Usage: WordCount <file>")
      System.exit(1)
    }

    val conf = new SparkConf().setAppName("WordCount")
    val sc = new SparkContext(conf)
    sc.textFile(args(0)).flatMap(_.split(" ")).map(x => (x, 1)).reduceByKey(_ + _).take(10).foreach(println)

    printToFile(new java.io.File("example.txt")) { p =>
      data.foreach(p.println)
    }

    sc.stop()
  }
}
