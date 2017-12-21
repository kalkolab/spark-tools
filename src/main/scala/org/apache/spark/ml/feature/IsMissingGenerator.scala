package org.apache.spark.ml.feature

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{ParamMap, StringArrayParam}
import org.apache.spark.ml.param.shared.HasInputCols
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset}

class IsMissingGenerator(override val uid:String) extends Transformer with DefaultParamsWritable with HasInputCols {

  def this() = this(Identifiable.randomUID("missingGenerator"))

  /**
    * Param for output columns names.
    * @group param
    */
  final val outputCols: StringArrayParam = new StringArrayParam(this, "outputCols", "output columns names")

  /** @group getParam */
  final def getOutputCols: Array[String] = $(outputCols)

  /** @group setParam */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** @group setParam */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    val generatedCols = col("*") +:
      $(inputCols).zip($(outputCols)).map { case (inputCol, outputCol) =>
        when(isnan(col(inputCol)) || isnull(col(inputCol)), lit(1.0)).otherwise(lit(0.0)).alias(outputCol)
      }
    dataset.select(generatedCols:_*).toDF
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    require($(inputCols).length == $(inputCols).distinct.length, s"inputCols contains" +
      s" duplicates: (${$(inputCols).mkString(", ")})")
    require($(outputCols).length == $(outputCols).distinct.length, s"outputCols contains" +
      s" duplicates: (${$(outputCols).mkString(", ")})")
    require($(inputCols).length == $(outputCols).length, s"inputCols(${$(inputCols).length})" +
      s" and outputCols(${$(outputCols).length}) should have the same length")
    val outputFields = $(outputCols).map(StructField(_, DoubleType, nullable = false))
    StructType(schema ++ outputFields)
  }
}
