package streaming

import org.apache.log4j._
import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.storage.StorageLevel
import scala.util.Random
import java.lang.Math.min
import scala.collection.mutable.ArrayBuffer
import scala.math.max
import scala.collection.mutable.Buffer

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerConfig, ProducerRecord}
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka._
import org.apache.spark.storage.StorageLevel
import _root_.kafka.serializer.{DefaultDecoder, StringDecoder}

object Main {
  val random = new Random(0)
  def main(args: Array[String]) {
        val kafkaConf = Map(
	"metadata.broker.list" -> "localhost:9092",
	"zookeeper.connect" -> "localhost:2181",
	"group.id" -> "kafka-spark-streaming",
	"zookeeper.connection.timeout.ms" -> "1000")
	
   val conf = new SparkConf().setAppName("dataMining").setMaster("local[2]")
    val sc = new StreamingContext(conf, Seconds(10))
    sc.checkpoint("checkpoint")
    val M = 10000.0//parameter
    val log = Logger.getLogger(this.getClass)

    log.warn("begin")
    //val filePath = "hdfs://input/out.moreno_cattle_cattle"
    //val filePath = "src/main/resources/moreno_bison/out.moreno_bison_bison"
    //val filePath = "src/main/resources/ucidata-gama/out.ucidata-gama"
    //val filePath = "src/main/resources/testTriangle.dat"
    //val dataBegin = sc.textFileStream(filePath)
    val dataBegin = KafkaUtils.createDirectStream[String, String, StringDecoder, StringDecoder]( sc,kafkaConf, Set("avg"))
    def f(x: String): (Int, (Int, Int)) = {
      val split = x.split("\\,")
      return (1, (split(0).toInt, split(1).toInt))
    }
    val data = dataBegin.map(x => (1, f(x._2)))
    

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

    def algo(context: (Double, Array[(Int, Int)], Array[(Int, Int)],Double,Int,Double) , element: (Int, (Int, Int))): (Double, Array[(Int, Int)], Array[(Int, Int)],Double,Int,Double) = {
      def mean(nextT:Double,taux:Int):Double={
          return(taux.toDouble*(max(1.0,((nextT)*(nextT-1.0)*(nextT-2.0))/((M)*(M-1)*(M-2)))))
      }
      val t = context._1
      val op = element._2._1
      val u = element._2._1
      val v = element._2._2
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
    
    val resultBuffer = Buffer[(Double, Array[(Int, Int)], Array[(Int, Int)],Double,Int,Double)]((0.0, Array[(Int, Int)](), Array[(Int, Int)]((0, 0)),0.0,0,0.0))
    //result
    def update (element:Seq[(Int, (Int, Int))],pred:Option[(Double, Array[(Int, Int)], Array[(Int, Int)],Double,Int,Double)]):Option[(Double, Array[(Int, Int)], Array[(Int, Int)],Double,Int,Double)]={
      val count = element.foldLeft(pred.getOrElse(resultBuffer.head))(algo)
      resultBuffer.update(0,count)
      Some(count)
    }
    //val dataUpdate = data.updateStateByKey(update)
    //val resultBuffer = Buffer[(Double, Array[(Int, Int)], Array[(Int, Int)],Double,Int,Double)]((0.0, Array[(Int, Int)](), Array[(Int, Int)]((0, 0)),0.0,0,0.0))
    //dataUpdate.foreachRDD(x=>x.foreach(x=>if(resultBuffer.head._1<x._2._1){resultBuffer.update(0,x._2)}))
    data.foreachRDD(x=>x.foreach(x=>resultBuffer.update(0,algo(resultBuffer.head,x._2))))
    sc.start()
    sc.awaitTermination(900)
    val result = resultBuffer.apply(0)
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
    }
}