built: 
	sbt update
	sbt package
	sbt assembly
clean:
	sbt clean clean-files
run:
	spark-submit ./target/scala-2.11/lab1-assembly-1.0.jar 2 0.75
eclipse:
	sbt eclipse
