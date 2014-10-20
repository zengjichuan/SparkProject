/**
 * Created by ZengJichuan on 2014/10/15.
 */
import breeze.linalg.{Vector, DenseVector, SparseVector}
import breeze.stats.distributions._

object Learn {
  def main(args: Array[String]){
    for {a <- 1 to 5
        if a>3
        if a<5
    }println(a)
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
    val svec1 = new SparseVector(Array(1,3,4), Array(1,3,3),5)
    println ("svec1 sum: "+svec1.sum)
    //show vector
    println("svec1: "+svec1)
    svec1.foreachKey(println)
    svec1.foreachValue(println)
    println("vec1 norm: "+vec1.norm(1))
    //vector operations

    println("vec1 dot vec2: "+vec2*vec1)

    //distribution
  }
  def sum(a:Int, b:Int)= a+b
}
