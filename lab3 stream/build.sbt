name := "DataMininglab3"

organization := "main"

version := "1.0"

scalaVersion := "2.10.0"

//resolvers += Resolver.mavenLocal

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.6.2" 
libraryDependencies += "org.log4s" %% "log4s" % "1.3.3" 
libraryDependencies += "org.apache.kafka" % "kafka_2.10" % "0.8.2.2" 
libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-streaming_2.10" % "1.6.2",
  "org.apache.spark" % "spark-streaming-kafka_2.10" % "1.6.2"
)


mainClass in assembly := Some("kafka/Main")

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
