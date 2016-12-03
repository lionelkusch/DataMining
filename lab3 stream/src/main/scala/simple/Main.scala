package simple



import org.apache.spark._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j._
import scala.util.Random
import java.lang.Math.min
import scala.collection.mutable.ArrayBuffer
import scala.math.max
import scala.collection.mutable.Buffer

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("dataMining").setMaster("local")
    val sc = new SparkContext(conf)
    val M = 100.0//parameter
    val log = Logger.getLogger(this.getClass)

    log.warn("begin")
    
    val filePath = "src/main/resources/moreno_cattle/out.moreno_cattle_cattle"
    //val filePath = "src/main/resources/moreno_bison/out.moreno_bison_bison"
    //val filePath = "src/main/resources/ucidata-gama/out.ucidata-gama"
    //val filePath = "src/main/resources/testTriangle.dat"
    log.warn("load file")
    val dataBegin = sc.textFile(filePath).collect.drop(2)

    //pretreatment data
    def f(x: String): (Int, (Int, Int)) = {
      val split = x.split("\\ ")
      return (1, (split(0).toInt, split(1).toInt))
    }
    val data = dataBegin.zipWithIndex.map(x => (x._2, f(x._1)))

    val random = new Random(0)

    //function

    def SampleUpdate(u: Int, v: Int, t: Int, S: Array[(Int, Int)], count: Array[(Int, Int)]): (Boolean, Array[(Int, Int)], Array[(Int, Int)]) = {
        if (t < M) {
          return (true, S , count)
        } else if (random.nextDouble() < (M / t)) {
          val index = random.nextInt(S.size)
          val updateCount = UpdateCounter(-1, S.apply(index)._1, S.apply(index)._2, S.filter(x => x._1 != index), count)
          //println("filter")
          return (true, S.filter(x => x != S.apply(index)), updateCount)
        } else {
          return (false, S, count)  //not mention in the paper
        }
      }

    def UpdateCounter(op: Int, u: Int, v: Int, S: Array[(Int, Int)], count: Array[(Int, Int)]): Array[(Int, Int)] = {
      //compute neighboor
      def neighboor(result: Array[Array[Int]], edge: (Int, Int)): Array[Array[Int]] = {
        if (edge._1 == u ) {
          return (Array[Array[Int]](result.apply(0) :+ edge._2, result.apply(1)))
        } else if (edge._1 == v ) {
          return (Array[Array[Int]](result.apply(0), result.apply(1) :+ edge._2))
        } else if (edge._2 == u) {
          return (Array[Array[Int]](result.apply(0) :+ edge._1, result.apply(1)))
        } else if (edge._2 == v) {
          return (Array[Array[Int]](result.apply(0), result.apply(1) :+ edge._1))
        } else {
          return (result)
        }
      }
      
      val newValue = ArrayBuffer[(Int,Int)]((u,0),(v,0))
      
      //update the count for each neighboor
      def Count(c: Int)= {
        count.update(0, (0, count.apply(0)._2 + op))
        // c 
            newValue+=((c,op))
        // U 
          newValue.update(0, (u, newValue.apply(0)._2 + op))
        //V
          newValue.update(1, (v, newValue.apply(1)._2 + op))
      }
      
      def updateCount(e:(Int,Int)):(Int,Int)={
        val index = count.indexWhere(x => x._1 == e._1 )
        if (index != -1){
          count.update(index, (e._1,count.apply(index)._2+e._2))
          return (-1,0)
        } else {
          return e
        }
      }
      
      val neighboorArray = S.foldLeft(Array[Array[Int]](Array[Int](),Array[Int]()))(neighboor)
      val neighboorUV = neighboorArray.apply(0).distinct.intersect(neighboorArray.apply(1).distinct)
      neighboorUV.foreach { x => Count(x) }
      val result = (count++(newValue.map(updateCount))).filter(x=>x._2>0)
      if (result.size == 0){
      return (Array[(Int, Int)]((0, 0)))
      } else{
        return result
      }
    }

    def algo(context: (Double, Array[(Int, Int)], Array[(Int, Int)],Double,Int,Double), element: (Int, (Int, (Int, Int)))): (Double, Array[(Int, Int)], Array[(Int, Int)],Double,Int,Double) = {
      def mean(nextT:Double,taux:Int):Double={
          return(taux.toDouble*(max(1.0,((nextT)*(nextT-1.0)*(nextT-2.0))/((M)*(M-1)*(M-2)))))
      }
      val t = context._1
      val op = element._2._1
      val u = element._2._2._1
      val v = element._2._2._2
      val S = context._2
      val count = context._3
      val n = context._5+1
      val sum = context._6
     println("n :"+n+" t : "+t+ "op : "+op+" S.size :"+S.size+ " count.size :"+count.size+" head count : "+count.head+"mean : "+(sum/(t.toDouble))+" eta :"+context._4)
      if (op == 1 && !S.contains((u,v)) && !S.contains((v,u))) {
        val nextT=t+1
        val result = SampleUpdate(u, v, nextT.toInt, S, count)
        if (result._1) {
          val updateCount = UpdateCounter(op, u, v, result._2:+(u,v), result._3)
          return (nextT, result._2:+(u,v),updateCount, mean(nextT,result._3.head._2),n,sum+mean(nextT,result._3.head._2))
        }
        return (nextT, result._2, result._3,mean(nextT,result._3.head._2),n, sum+mean(nextT,result._3.head._2))
      }
        return (t, S, count,mean(t,count.head._2),n,sum+mean(t,count.head._2))
    }

    //result
    val result = data.foldLeft((0.0, Array[(Int, Int)](), Array[(Int, Int)]((0, 0)),0.0,0,0.0))(algo)
   val t = result._1
   val taux= result._3.head._2
   val eta = result._4
   val mean = (result._5.toDouble*taux.toDouble)/(t-M).toDouble
         println(" M : "+M+" taux : "+taux+" eta :"+eta+" sum : "+result._5+" count size : "+result._3.size)
   if (t<=M){
     println("nb Triangle ="+taux)
   } else{
     println("nb Triangle = "+mean)
   }
    
  
    
    
    
    
//    //true number of triangle in S
//   def CountTriangle (s:Array[(Int, Int)], count:Array[(Int, Int)]):Int={
//      def neighboor(u:Int,v:Int,S:Buffer[(Int, Int)]): Array[Int] = {
//        def f(result:Array[Int],edge:(Int,Int)):Array[Int]={
//          if (edge._1==u&&edge._2!=v){
//            return(result:+edge._2)
//          } else if (edge._2==u&&edge._1!=v){
//            return(result:+edge._1)
//          }
//          return(result)
//        }
//        S.foldLeft(Array[Int]())(f)
//      }
//      
//      def nbtriangle(u:Int,v:Int,S:Buffer[(Int, Int)]):Int={
//        neighboor(u,v,S).distinct.intersect(neighboor(v,u,S).distinct).size
//      }
//     def drop(u:Int,v:Int,S:Buffer[(Int, Int)]):Buffer[(Int, Int)]={
//       return(S.map(x=>{if ((x._1== u&&x._2==v)||(x._1== v&&x._2==u)){(0,0)}else{x}}))
//     }
//     val countU = Array[Int](0)
//     val S=s.toBuffer
//     //println(S.mkString(","))
//     for(i<-1 to count.size-1){
//       val neighboorI = neighboor(i,i, S).distinct
//       neighboorI.foreach(j=>{ if ( i!=j){ countU.update(0,nbtriangle(i, j,S)+countU.apply(0))}})
//     }
//     return(countU.apply(0)/6)
//   }
//   println("Triangle : "+CountTriangle (result._2,result._3))
  }
}