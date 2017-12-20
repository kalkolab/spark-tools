package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.attribute.NumericAttribute
import org.apache.spark.ml.feature.MeanEncoderModel.MeanEncoderModelWriter
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasOutputCol}
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types.{NumericType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

trait MeanEncoderBase extends Params with HasInputCol with HasOutputCol {

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val inputDataType = schema(inputColName).dataType
    require(inputDataType == StringType || inputDataType.isInstanceOf[NumericType],
      s"The input column $inputColName must be of string or numeric type, " +
        s"but got $inputDataType.")
    val inputFields = schema.fields
    val outputColName = $(outputCol)
    require(inputFields.forall(_.name != outputColName),
      s"Output column $outputColName already exists.")
    val attr = NumericAttribute.defaultAttr.withName($(outputCol))
    val outputFields = inputFields :+ attr.toStructField()
    StructType(outputFields)
  }
}

class MeanEncoder(override val uid: String) extends Estimator[MeanEncoderModel]
  with MeanEncoderBase with DefaultParamsWritable with HasLabelCol {
  def this() = this(Identifiable.randomUID("meanEncoder"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group setParam */
  def setLabelCol(value: String): this.type = set(labelCol, value)

  override def copy(extra: ParamMap): MeanEncoder = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema)

  override def fit(dataset: Dataset[_]): MeanEncoderModel = {
    import org.apache.spark.sql.functions._
    val inputColName = $(inputCol)
    val labelColName = $(labelCol)
    val targetMeans = dataset.select(col(inputColName).cast(StringType), col(labelColName))
      .groupBy(col(inputColName)).agg(mean(labelColName))
      .collect.map {
      case Row(categ: String, mean: Double) => categ -> mean
      case Row(null, mean: Double) => "null" -> mean
    }.toMap
    val priorMean = dataset.agg(mean(labelColName)).head.getDouble(0)
    copyValues(new MeanEncoderModel(uid, targetMeans, priorMean).setParent(this))
  }
}

class MeanEncoderModel (override val uid: String, val targetMeans: Map[String, Double], val priorMean: Double)
  extends Model[MeanEncoderModel] with MeanEncoderBase with MLWritable {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def copy(extra: ParamMap): MeanEncoderModel = {
    val copied = new MeanEncoderModel(uid, targetMeans, priorMean)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new MeanEncoderModelWriter(this)

  override def transform(dataset: Dataset[_]): DataFrame = {
    import org.apache.spark.sql.functions.{col, udf}
    val meanEncodeUdf = udf {
      categ: String => categ match {
        case s:String => targetMeans.get(categ).getOrElse(priorMean)
        case null => targetMeans.get("null").getOrElse(priorMean)
      }
    }
    dataset.toDF.withColumn($(outputCol), meanEncodeUdf(col($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    if (schema.fieldNames.contains($(inputCol))) {
      validateAndTransformSchema(schema)
    } else {
      // If the input column does not exist during transformation, we skip MeanEncoderModel.
      schema
    }
  }
}

object MeanEncoderModel extends MLReadable[MeanEncoderModel] {

  private[MeanEncoderModel]
  class MeanEncoderModelWriter(instance: MeanEncoderModel) extends MLWriter {

    private case class Data(means: Map[String, Double], priorMean: Double)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.targetMeans, instance.priorMean)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class MeanEncoderModelReader extends MLReader[MeanEncoderModel] {

    private val className = classOf[MeanEncoderModel].getName

    override def load(path: String): MeanEncoderModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
        .select("means", "priorMean")
        .head()
      val means = data.getAs[Map[String, Double]](0)
      val priorMean = data.getDouble(1)
      val model = new MeanEncoderModel(metadata.uid, means, priorMean)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  override def read: MLReader[MeanEncoderModel] = new MeanEncoderModelReader

  override def load(path: String): MeanEncoderModel = super.load(path)
}