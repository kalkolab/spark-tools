package org.apache.spark.ml.feature

import org.apache.spark.annotation.Since
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{DoubleType, NumericType, StructType}

/**
  * A binary encoder that maps a column of category indices to a column of binary vectors, with
  * at most a single one-value per row that indicates the input category index.
  * For example with 5 categories, an input value of 2.0 would map to an output vector of
  * `[0.0, 0.0, 1.0, 0.0]`.
  * The last category is not included by default (configurable via `OneHotEncoder!.dropLast`
  * because it makes the vector entries sum up to one, and hence linearly dependent.
  * So an input value of 4.0 maps to `[0.0, 0.0, 0.0, 0.0]`.
  *
  * @see `StringIndexer` for converting categorical values into category indices
  */
class BinaryEncoder(override val uid: String) extends Transformer
  with HasInputCol with HasOutputCol with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("binaryEncoder"))

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  override def transformSchema(schema: StructType): StructType = {
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)

    require(schema(inputColName).dataType.isInstanceOf[NumericType],
      s"Input column must be of type NumericType but got ${schema(inputColName).dataType}")
    val inputFields = schema.fields
    require(!inputFields.exists(_.name == outputColName),
      s"Output column $outputColName already exists.")

    val inputAttr = Attribute.fromStructField(schema(inputColName))
    val outputAttrNames: Option[Array[String]] = inputAttr match {
      case nominal: NominalAttribute =>
        if (nominal.values.isDefined) {
          nominal.values
        } else if (nominal.numValues.isDefined) {
          nominal.numValues.map(n => Array.tabulate(n)(_.toString))
        } else {
          None
        }
      case binary: BinaryAttribute =>
        if (binary.values.isDefined) {
          binary.values
        } else {
          Some(Array.tabulate(2)(_.toString))
        }
      case _: NumericAttribute =>
        throw new RuntimeException(
          s"The input column $inputColName cannot be numeric.")
      case _ =>
        None // optimistic about unknown attributes
    }

    val outputAttrGroup = if (outputAttrNames.isDefined) {
      val attrs: Array[Attribute] = outputAttrNames.get.map { name =>
        BinaryAttribute.defaultAttr.withName(name)
      }
      new AttributeGroup($(outputCol), attrs)
    } else {
      new AttributeGroup($(outputCol))
    }

    val outputFields = inputFields :+ outputAttrGroup.toStructField()
    StructType(outputFields)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    // schema transformation
    val inputColName = $(inputCol)
    val outputColName = $(outputCol)
    var outputAttrGroup = AttributeGroup.fromStructField(
      transformSchema(dataset.schema)(outputColName))
    if (outputAttrGroup.size < 0) {
      // If the number of attributes is unknown, we check the values from the input column.
      val numAttrs = dataset.select(col(inputColName).cast(DoubleType)).rdd.map(_.getDouble(0))
        .aggregate(0.0)(
          (m, x) => {
            assert(x <= Int.MaxValue,
              s"BinaryEncoder only supports up to ${Int.MaxValue} indices, but got $x")
            assert(x >= 0.0 && x == x.toInt,
              s"Values from column $inputColName must be indices, but got $x.")
            math.max(m, x)
          },
          (m0, m1) => {
            math.max(m0, m1)
          }
        ).toInt
      val outputAttrNames = Array.tabulate(numAttrs)(_.toString)
      val outputAttrs: Array[Attribute] =
        outputAttrNames.map(name => BinaryAttribute.defaultAttr.withName(name))
      outputAttrGroup = new AttributeGroup(outputColName, outputAttrs)
    }
    val metadata = outputAttrGroup.toMetadata()

    // data transformation
    val size = math.ceil(math.log(outputAttrGroup.size)/math.log(2.0)).toInt
    val emptyValues = Array.empty[Double]
    val emptyIndices = Array.empty[Int]
    val encode = udf { label: Double =>
      if (label < outputAttrGroup.size) {
        val binaryString = label.toInt.toBinaryString
        val sparseBinaryString = binaryString.zipWithIndex
          .collect{ case(c, i) if c=='1' => (i + size - binaryString.length, 1.0)}.toArray
        Vectors.sparse(size, sparseBinaryString.map(_._1), sparseBinaryString.map(_._2))
      } else {
        Vectors.sparse(size, emptyIndices, emptyValues)
      }
    }

    dataset.select(col("*"), encode(col(inputColName).cast(DoubleType)).as(outputColName, metadata))
  }

  override def copy(extra: ParamMap): BinaryEncoder = defaultCopy(extra)
}

@Since("1.6.0")
object BinaryEncoder extends DefaultParamsReadable[BinaryEncoder] {

  override def load(path: String): BinaryEncoder = super.load(path)
}
