import org.apache.spark.sql.{Row, SparkSession}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import org.apache.spark.ml.Pipeline

object SparkNLPProcessor {

  def main(args: Array[String]): Unit = {
    val spark = initializeSparkSession()

    val dataPath = "/Users/mbair/Desktop/SparkNLPProject/src/main/spark_nlp_dataset.parquet"
    val dataset = loadDataset(spark, dataPath)

    val pipeline = buildNlpPipeline()
    val pipelineModel = pipeline.fit(dataset)
    val processedData = pipelineModel.transform(dataset)

    val relationships = extractEntityPosRelationships(processedData, spark)

    relationships.show(truncate = false)

    spark.stop()
  }

  private def initializeSparkSession(): SparkSession = {
    SparkSession.builder()
      .appName("Spark NLP Entity-POS Analysis")
      .master("local[*]")
      .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.1")
      .getOrCreate()
  }

  private def loadDataset(spark: SparkSession, path: String): DataFrame = {
    spark.read.parquet(path)
  }

  private def buildNlpPipeline(): Pipeline = {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val wordEmbeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("embeddings")

    val posTagger = PerceptronModel.pretrained("pos_anc", "en")
      .setInputCols(Array("document", "token"))
      .setOutputCol("pos")

    val nerModel = NerCrfModel.pretrained("ner_crf", "en")
      .setInputCols(Array("document", "token", "pos", "embeddings"))
      .setOutputCol("ner")

    new Pipeline().setStages(Array(documentAssembler, tokenizer, wordEmbeddings, posTagger, nerModel))
  }

  private def extractEntityPosRelationships(df: DataFrame, spark: SparkSession): DataFrame = {
    import spark.implicits._

    df.select("text", "pos", "ner").rdd.flatMap { row =>
      val text = row.getAs[String]("text")
      val posTags = row.getAs[Seq[Row]]("pos").map(_.getAs[String]("result"))
      val nerTags = row.getAs[Seq[Row]]("ner").map(_.getAs[String]("result"))

      nerTags.zip(posTags).map { case (entity, posTag) =>
        val explanation = explainEntityPosRelationship(entity, posTag)
        (text, entity, posTag, explanation)
      }
    }.toDF("Text", "Entity", "POS_Tag", "Explanation")
  }

  private def explainEntityPosRelationship(entity: String, posTag: String): String = posTag match {
    case "NNP" => s"$entity is a proper noun, representing a person, location, or organization."
    case "NN"  => s"$entity is a common noun, referring to an object or thing."
    case "VB"  => s"$entity is a verb, indicating an action or state."
    case "JJ"  => s"$entity is an adjective, describing a noun."
    case "RB"  => s"$entity is an adverb, modifying a verb or adjective."
    case _     => s"$entity with POS tag $posTag represents an unspecified category."
  }
}
