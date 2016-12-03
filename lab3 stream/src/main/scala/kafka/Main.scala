package kafka

import java.util.Properties
import kafka.producer.ProducerConfig
import kafka.javaapi.producer.Producer
import kafka.producer.KeyedMessage
import scala.util.Random

object Main {

  def main(args: Array[String]) {
    val max = 10
    def getRandomVal: String = {
      val i = Random.nextInt()
      val key = Random.nextInt(max)
      val value = Random.nextInt(max)
      println("generated: " + key + "," + value)
      key + "," + value
    }

    val props = new Properties()
    props.put("metadata.broker.list", "127.0.0.1:9092")
    props.put("serializer.class", "kafka.serializer.StringEncoder")
    props.put("request.required.acks", "1")

    val config = new ProducerConfig(props)
    val producer = new Producer[String, String](config)
    val topic = "avg"

    while (true) {
      Thread.sleep(10)
      producer.send(new KeyedMessage[String, String](topic, null, getRandomVal))
    }

    producer.close
  }

}

