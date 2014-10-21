/**
 * Created by ZengJichuan on 2014/10/15.
 */
import breeze.linalg.{Vector, DenseVector, SparseVector}
import breeze.stats.distributions._

object Learn {

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
  def main(args: Array[String]){
    var change:Int = 0
    val ha = for {
      a <- 1 to 5
      val cc = a+2
      val b = a
    } yield {
      change=change+a*2
      b
    }
    ha.foreach(println)
    val ab = 1 to 3
    println(ab.filter(_>0))
    val f = (_:Int) +(_:Int)
    val b = sum(1,_:Int)

    println(b(5))
    val a: Int = 1
    println(a)
    val vec = DenseVector.zeros[Double](50)
    val vec1 = DenseVector[Double](1,2,3,4)
    val vec2 = DenseVector.rand(4).t
    var svec1 = new SparseVector(Array(0,2,3), Array(1,3,3),4)
    val svec2 = new SparseVector(Array(3,2), Array(1,3),4)
    val svec3 = SparseVector.zeros[Int](4)
    println("svec1 sum: "+svec1.sum)
    println("svec1 size: "+svec1.length)
    svec2.toArray.foreach(println)
    println("svec1 + svec2: "+(svec1+svec2))
    println("svec2 norm: "+svec2.used)
//    svec2.mapActiveValues(println)
    //show vector
    svec1 += svec2
    println("svec1: "+svec1)
    svec1.foreachKey(println)
    svec1.foreachValue(println)
    println("vec1 norm: "+vec1.norm(1))
    //vector operations

    println("vec1 dot vec2: "+vec2*vec1)

    //distribution
    val arr = Array((1,2,3),(2,3,4))
    val (f1, f2, f3) = arr.unzip3
    f2.foreach(println)
    val bb = 5+(f3.reduce(_+_))
    println(bb)
    val data = Array("Five","strings","in","a","file!")
    printToFile(new java.io.File("example.txt")) { p =>
      data.foreach(p.println)
    }
  }
  def sum(a:Int, b:Int)= a+b
}
