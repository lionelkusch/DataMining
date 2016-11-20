package dataMining.associationRule

import org.apache.spark._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{ Row, SQLContext, DataFrame }
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.Pipeline
import scala.collection.mutable.WrappedArray
import org.apache.spark.rdd.RDD
import org.apache.log4j._

object FunctionAssociationRule {

  /**
   * @param listItem 	: list of the group of items
   * @param basket 		: a basket of items
   * @param size			: the number of items in the group of the list
   * @return an array of group of items which is in the listItem and contains in the basket
   * 
   */
  def contain(listItem: Array[String], basket: Array[String], size: Int,logger:Logger): Array[String] = {
    def findItem(items: Array[String]): String = {
      if (items.diff(basket).size == 0) {
        return (items.mkString("|")) //return the group of items
      } else {
        return ("") // return nothing
      }
    }

    if (basket.size >= size) {
      listItem.map(x => findItem(x.split("\\|")))
                                  .distinct //drop the multiple element
                                  .diff(Array("") //drop the elements empty
                                      )
    } else {
      return (Array())
    }
  }

  /**
   * @param Item : list of group of items
   * @return an array of groups of items which are the mixing each group of items in Item
   */
  def multList(Item: Array[String], sc: SparkContext,logger:Logger): Array[String] = {
    def mix(x: String, y: String): String = {
      val valueX = x.split("\\|").map(_.toInt)
      val valueY = y.split("\\|").map(_.toInt)
      val diff = valueX.diff(valueY)
      //logger.info("diff :"+ diff.mkString(","))
      if (diff.size == 1) {
        return((valueY ++ diff).sortWith(_ < _).mkString("|"))//return the mix of items
      } else {
        return("")//return nothing
      }
    }

    val buffer = scala.collection.mutable.ArrayBuffer.empty[String]
    val bufferItem = Item.toBuffer
    for (i <- 0 to Item.size - 1) {
      bufferItem.remove(0) //avoid the duplicate
      //mix each item with the other
      buffer ++= bufferItem.map({ x => mix(x, Item.apply(i)) })
    }
    return(buffer.distinct.diff(Array("")).toArray)//drop the empty element
  }
/**
 * @param data 	: the list of baskets of items
 * @param s			: support threshold
 * @return the two first iteration of the algorithm which compute the frequent itemsets and the list of basket 
 */
  def init(data: Array[String],s:Int, sc: SparkContext,logger:Logger): ((Array[Array[String]], Array[(String, Int)]), Array[(String, Int)]) = {
    val items = data.map(_.split(" "))//create a Array of basket
    val itemPreCount = items.reduce(_ ++ _).map(x => (x, 1)) //pre-compute the count of element
    val itemCount = sc.parallelize(itemPreCount).reduceByKey(_ + _) // count the number of element
                                                .filter({ case (x, y) => y >= s } // filter the items
                                                ).collect
    val itemFrequent = itemCount.unzip._1//list of frequent item

    val pairitem = multList(itemFrequent, sc,logger) //compute the list of all pair for frequent item
    val pairPreCount = items.map(x => contain(pairitem, x, 2,logger)).reduce(_ ++ _).map(x => (x, 1))//pre-compute the count of element
    val pairCount = sc.parallelize(pairPreCount).reduceByKey(_ + _) // count the number of pair
                                                .filter({ case (x, y) => y >= s } //filter the previous result
                                                ).collect
    return ((items, itemCount), pairCount)
  }

  /**
   * @param s 			: support threshold
   * @param nbIter 	: number of element in the group of list
   * @param itemFeqPrevious
   */
  def freq(s: Int, nbIter: Int, items: Array[Array[String]], itemFreqPrevious: Array[String],sc: SparkContext, logger:Logger): Array[(String, Int)] = {
    val multpair = multList(itemFreqPrevious,sc,logger) //the list of all multi mix of frequent item
    val countInit = items.map(x => contain(multpair, x, nbIter,logger))//search if the item is in the basket
                                      .reduce(_ ++ _).map(x => (x, 1)//pre-count
                                          )
    val count = sc.parallelize(countInit).reduceByKey(_ + _)//count the number of item
                                         .filter({ case (x, y) => y >= s } //filter with the support threshold
                                         ).collect()
    return(count)
  }

  /**
   * @param itemFrequent 	: group of item often in the same basket
   * @param count 				: the number of representation of each frequent item
   * @param c							: interesting rule association
   */
  def calculconfidence(itemFrequent: String, count: Array[(String, Int)], c: Double,logger:Logger,sc: SparkContext): Array[((String, String),Double)] = {
    val items = itemFrequent.split("\\|")//find item in x
    val buffer = Array((items.apply(0), items.apply(1)), (items.apply(1), items.apply(0))).toBuffer // init the loop
    for (i <- 2 to items.size - 1) {
      val buffer1 = buffer.map(x => (x._1 + "|" + items.apply(i), x._2)) //add the item in right side of the rule
      val buffer2 = buffer.map(x => (x._1, x._2 + "|" + items.apply(i))) //add the item in left side of rule
      val buffer3 = Array((items.apply(i), items.take(i).reduceRight((x, y) => x + "|" + y)), //create a rule with the only the element in right side
                          (items.take(i).reduceRight((x, y) => x + "|" + y), items.apply(i))//create a rule with the only the element in left side
                          ).toBuffer
      buffer.clear
      buffer ++= buffer1 ++ buffer2 ++ buffer3
    }
    val countRule = buffer.unzip._1 //need only the count of the item in the left side of the rule
    val countX = count.apply(count.unzip._1.indexOf(itemFrequent))._2 //the count of the rule
    val countRuleCuple = count.filter({ x => countRule.contains(x._1) })//collect the count of item
                              .map(x => (x._1,"count:"+(countX.toDouble / x._2.toDouble).toString()) )// compute the confidence
    val result = sc.parallelize(countRuleCuple++buffer).reduceByKey({(x,y)=>if ((x.split(":")(0))=="count")
                                                                                {y+":"+x.split(":")(1)}
                                                                            else{(x+":"+y.split(":")(1))}})//associate the confidence to the rule
                                                       .map(x=>(((x._1,x._2.split(":")(0)),x._2.split(":")(1).toDouble)))//make in forme
                                                       .filter(x=>x._2>=c)//filter
                                                       .collect()
   return (result)
  }

  /**
   * @param data  : the array of string which contain the ordered int and slip by a space
   * @param s 	  : support threshold
   * @param c 	 	: interesting rule association
   */
  def AssociationRule(data: Array[String], s: Int, c: Double, sc: SparkContext,logger:Logger): Array[((String, String),Double)] = {
        
      logger.info("Init association Rule")
    val initial = init(data,s, sc,logger)
      logger.info("Frequent item")
    val buffer = scala.collection.mutable.ArrayBuffer.empty[Array[(String, Int)]]
    buffer += (initial._2)//init the loop of calculate the frequent item
    while (buffer(buffer.size - 1).size > 1) {
      buffer += freq( s, buffer.size + 2, initial._1._1, buffer(buffer.size - 1).unzip._1,sc,logger)
        logger.info("step calcul item :" + (buffer.size + 1))
    }
      println(initial._1._2.mkString(","))
      println(buffer.foreach(x=>println(x.mkString(","))))
      logger.info("calcul coef confidence")
    val confidenceInit = buffer.aggregate(Array[(String)]())({case(y:Array[String],x:Array[(String, Int)])=>y++(x.unzip._1)}, _ ++ _)
    val confidence = confidenceInit.map(x => calculconfidence(x, initial._1._2 ++ buffer.aggregate(Array[(String, Int)]())(_ ++ _, _ ++ _), c,logger,sc))
    val result = confidence.aggregate(Array[((String, String),Double)]())(_ ++ _, _ ++ _)
      logger.info("result size :"+result.size)
    return (result)
  }

}