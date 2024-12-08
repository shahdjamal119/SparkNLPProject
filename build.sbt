import scala.collection.Seq

ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.13"

lazy val root = (project in file("."))
  .settings(
    name := "Spark"
  )
libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.5.1",
  "org.apache.spark" %% "spark-sql" % "3.5.1",
  "org.apache.spark" %% "spark-mllib" % "3.5.1",
  "com.johnsnowlabs.nlp" %% "spark-nlp" % "5.5.1",
  "org.tensorflow" % "tensorflow-core-platform" % "0.5.0"
)

// Add Spark NLP specific repository if needed
resolvers += "John Snow Labs Repository" at "https://johnsnowlabs.jfrog.io/artifactory/spark-nlp"