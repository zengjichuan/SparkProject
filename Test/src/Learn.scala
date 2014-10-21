/**
 * Created by ZengJichuan on 2014/10/15.
 */

import breeze.linalg.{SparseVector, DenseVector}

import scala.util.control.Breaks._

object Learn {

  /**
   * Used for reading/writing to database, files, etc.
   * Code From the book "Beginning Scala"
   * http://www.amazon.com/Beginning-Scala-David-Pollak/dp/1430219890
   */
  def using[A <: {def close(): Unit}, B](param: A)(f: A => B): B =
    try { f(param) } finally { param.close() }

  def writeToFile(fileName:String, data:String) =
    using (new java.io.FileWriter(fileName)) {
      fileWriter => fileWriter.write(data)
  }
  def appendToFile(fileName:String, textData:String) =
    using (new java.io.FileWriter(fileName, true)){
      fileWriter => using (new java.io.PrintWriter(fileWriter)) {
        printWriter => printWriter.println(textData)
    }
  }

  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
  def main(args: Array[String]){
    val ab = 1 to 3
    println(ab.filter(_>0))
    val f = (_:Int) +(_:Int)

//    val svec1 = new OpenMapRealVector((3, Array(1,3), Array(2,3))
    var j = true
    breakable{for(i <- 1 to 3){
      if (j == false) j
      println(i)
      if (i>=2)
        break
    }}

    val vec = DenseVector.zeros[Double](50)
    val vec1 = DenseVector[Double](1,2,3,4)
    val vec2 = DenseVector.rand(4).t
    var svec1 = new SparseVector[Double](Array(0,2,3), Array(1.0,3.0,3.0),4)
    val svec2 = new SparseVector[Double](Array(3,1), Array(1.0,3.0),4)
    var svec3 = SparseVector.zeros[Double](4)
    println("svec1 sum: "+svec1.sum)
    println("svec1 size: "+svec1.length)
    svec2.toArray.foreach(println)
    println("svec1 + svec2: "+(svec1+svec2))
    println("svec2 norm: "+svec2.used)
    for(i <- 1 to 100) {
      svec3 += svec1
      svec3 += svec2
    }
        //    svec2.mapActiveValues(println)
    println("svec3: "+svec3)
    svec1.foreachKey(println)
    svec1.foreachValue(println)
    println("vec1 norm: "+vec1.norm(1))
    //vector operations

    println("vec1 dot vec2: "+vec2*vec1)

    //distribution
    val data = Array("Five","strings","in","a","file!")
    printToFile(new java.io.File("example.txt")) { p =>
      data.foreach(p.println)
    }
    val str = "hello world!"
    val file = new java.io.File("example.txt")
    printToFile(file) {p =>
      p.println(str)
      p.println("me")
    }

    printToFile(file) {p => p.println(str)}
    writeToFile("example1.txt", s"str${1.0423.formatted("%.3f")}")
    appendToFile("example.txt", str)
    appendToFile("example.txt", str)
  }
}
