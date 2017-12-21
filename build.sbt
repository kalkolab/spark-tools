name := "spark-tools"
version := "0.1"
scalaVersion := "2.11.12"

val sparkVersion = "2.2.0"
  
libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion  
libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion
libraryDependencies += "org.apache.spark" %% "spark-yarn" % sparkVersion

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.4" % "test"
libraryDependencies += "com.holdenkarau" %% "spark-testing-base" % "2.2.0_0.8.0" % "test"
libraryDependencies += "org.apache.spark" %% "spark-hive" % sparkVersion % "test"
