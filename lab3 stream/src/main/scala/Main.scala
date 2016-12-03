

import org.apache.spark._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j._
import scala.util.Random
import java.lang.Math.min
import scala.collection.mutable.ArrayBuffer

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("dataMining").setMaster("local")
    val sc = new SparkContext(conf)
    val M = 1000//parameter
    val log = Logger.getLogger(this.getClass)

    log.warn("begin")
    val filePath = "src/main/resources/link-dynamic-simplewiki/out.link-dynamic-simplewiki"
    log.warn("load file")
    val dataBegin = sc.textFile(filePath).collect.drop(2).take(9200)

    //pretreatment data
    def f(x: String): (Int, (Int, Int)) = {
      val split = x.split("\\ ")
      if (split(2) == "-1") {
        return (-1, (split(0).toInt, split(1).toInt))
      } else {
        return (1, (split(0).toInt, split(1).toInt))
      }
    }
    val data = dataBegin.zipWithIndex.map(x => (x._2, f(x._1)))

    val random = new Random(0)

    //function

    def SampleUpdate(u: Int, v: Int, d0: Int, di: Int, t: Int, S: Array[(Int, Int)], count: Array[(Int, Int)]): (Boolean, Int, Int, Array[(Int, Int)], Array[(Int, Int)]) = {
      if (di + d0 == 0) {
        if (S.size < M) {
          return (true, d0, di, S :+ ((u, v)), count)
        } else if (random.nextDouble() < (M / t)) {
          val index = random.nextInt(S.size)
          val updateCount = UpdateCounter(-1, S.apply(index)._1, S.apply(index)._2, S, count)
          print("filter")
          return (true, d0, di, S.filter(x => x._1 != index) :+ (u, v), updateCount)
        } else {
          return (false, d0 , di, S, count)  //not mention in the paper
        }
      } else if (random.nextDouble() < (di / (di + d0))) {
        return (true, d0, di - 1, S :+ (u, v), count)
      } else {
        return (false, d0 - 1, di, S, count)
      }
    }

    def UpdateCounter(op: Int, u: Int, v: Int, S: Array[(Int, Int)], count: Array[(Int, Int)]): Array[(Int, Int)] = {
      //compute neighboor
      def neighboor(result: Array[Array[Int]], edge: (Int, Int)): Array[Array[Int]] = {
        if (edge._1 == u) {
          return (Array[Array[Int]](result.apply(0) :+ edge._2, result.apply(1)))
        } else if (edge._1 == v) {
          return (Array[Array[Int]](result.apply(0), result.apply(1) :+ edge._2))
        } else if (edge._2 == u) {
          return (Array[Array[Int]](result.apply(0) :+ edge._1, result.apply(1)))
        } else if (edge._2 == v) {
          return (Array[Array[Int]](result.apply(0), result.apply(1) :+ edge._1))
        } else {
          return (result)
        }
      }
      
      val newValue = ArrayBuffer[(Int,Int)]()
      val indexU = count.indexWhere(x => x._1 == u)
      if (indexU == -1){
        newValue.+=:(u,0)
      }
      val indexV = count.indexWhere(x => x._1 == v)
      if (indexV == -1){
        newValue.+=:(v,0)
      }
      //update the count for each neighboor
      def updateCount(c: Int)= {
        // c index 
        count.update(0, (0, count.apply(0)._2 + op))
        val indexC = count.indexWhere(x => x._1 == c)
        if (indexC == -1){
          val newIndex = newValue.indexWhere(x => x._1 == c)
          if (newIndex != -1){
            newValue.update(newIndex, (c, count.apply(indexC)._2 + op))
          } else{
            newValue.+=:(c,op)
          }
        } else{
            count.update(indexC, (c, count.apply(indexC)._2 + op))
        }
        // U 
        if (indexU == -1){
          val newIndex = newValue.indexWhere(x => x._1 == u)
          newValue.update(newIndex, (u, newValue.apply(newIndex)._2 + op))
        } else{
          count.update(indexU, (u, count.apply(indexU)._2 + op))
        }
        //V
        if (indexV == -1){
          val newIndex = newValue.indexWhere(x => x._1 == v)
          newValue.update(newIndex, (v, newValue.apply(newIndex)._2 + op))
        } else{
          count.update(indexV, (v, count.apply(indexV)._2 + op))
        }
      }

      val neighboorArray = S.foldLeft(Array[Array[Int]](Array[Int](),Array[Int]()))(neighboor)
      val neighboorUV = neighboorArray.apply(0).intersect(neighboorArray.apply(1))
      neighboorUV.foreach { x => updateCount(x) }
      if ((count++newValue).filter(x=>x._2>0).size == 0){
        println("not edge")
      return (Array[(Int, Int)]((0, 0)))
      } else{
        return (count++newValue).filter(x=>x._2>0)
      }
    }

    def algo(context: (Int, Int, Int, Int, Array[(Int, Int)], Array[(Int, Int)],Double), element: (Int, (Int, (Int, Int)))): (Int, Int, Int, Int, Array[(Int, Int)], Array[(Int, Int)],Double) = {
      val t = element._1
      val op = element._2._1
      val s = context._2 + op
      val d0 = context._3
      val di = context._4
      val u = element._2._2._1
      val v = element._2._2._2
      val S = context._5
      val count = context._6
      //the calcul is not correct
      val k = if (s<M){1}else if(s==M){0.0}else {if (op ==1 ){context._7*(s-M)/(s-3)}else{context._7}}
     println("time : "+t, "op : "+op," s: "+s+" S.size :"+S.size+" di : "+di+" d0 : "+d0+ " count.size :"+count.size+" head count : "+count.head+" k :"+k)
      if (op == 1) {
        val result = SampleUpdate(u, v, d0, di, t, S, count)
        if (result._1) {
          val updateCount = UpdateCounter(op, u, v, result._4, result._5)
          return (t, s, result._2, result._3, result._4, updateCount,k)
        }
        return (t, s, result._2, result._3, result._4, result._5,k)
      } else if (S.contains((u, v))) {
        val updateCount = UpdateCounter(op, u, v, S, count)
        val newS = S.filter(x=>x!=(u, v))
        return (t, s, d0, di + 1, newS, updateCount,k)
      } else {
        return (t, s, d0 + 1, di, S, count,k)
      }
    }

    //result
    val result = data.foldLeft((0, 0, 0, 0, Array[(Int, Int)](), Array[(Int, Int)]((0, 0)),0.0))(algo)
    
      val s = result._2.toDouble 
      val d0 = result._3
      val di = result._4
      val w = min(M,s+di+d0).toDouble
    val k =result._7.toDouble
   val taux= result._6.head._2
   val Mt=result._5.size
         println("s : "+s+" d0 :"+d0+" di = "+di+" Mt : "+Mt+" taux : "+taux+" k:"+k)
   if (Mt<3){
     println("nb Triangle =0")
   } else{
     val triangle = (taux.toDouble/k.toDouble)*(s*(s-1.0)*(s-2.0))/(Mt*(Mt-1.0)*(Mt-2.0))
     println("nb Triangle = "+triangle)
   }
  }
}