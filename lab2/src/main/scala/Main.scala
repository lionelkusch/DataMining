import dataMining.associationRule._

import org.apache.spark._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j._
import java.io._

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab2").setMaster("local")
    val sc = new SparkContext(conf)
    val logger = Logger.getLogger(this.getClass)

    if (args.size >= 2 || args.size <= 4) {

      val filePath = if (args.size >= 3) { args.apply(2) } else { "src/main/resources/exampleCurse.dat" }
      val fileResult = if (args.size >= 4) { args.apply(3) } else { "result/result.txt" }
      val data = sc.textFile(filePath).collect
      println(data.mkString(","))
      println(args.mkString(","))
      val result = FunctionAssociationRule.AssociationRule(data, args(0).toInt, args(1).toDouble, sc, logger)
      logger.info("Result Assication Rule")
      logger.info(result.mkString("\n"))
      logger.info("write result")
      val file = new FileWriter(fileResult)
      file.write(result.mkString("\n"))
      file.close()

    }
    logger.info("end Assication Rule")
    sc.stop()
  }
}