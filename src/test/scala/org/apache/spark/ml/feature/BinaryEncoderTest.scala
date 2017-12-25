package org.apache.spark.ml.feature

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{IntegerType, StringType}
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class BinaryEncoderTest extends FunSuite with DataFrameSuiteBase {

  override def enableHiveSupport: Boolean = false
  override def reuseContextIfPossible: Boolean = true
  import spark.implicits._
  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-3)

  val columns = Array("string", "ordinal", "label")
  val outputCol = "binEnc"
  val data = Seq(
    (Some("A"), 0, 1.0),
    (Some("A"), 1, 0.0),
    (Some("B"), 2, 1.0),
    (None, 0, 0.0),
    (Some("B"), 1, 1.0),
    (Some("B"), 2, 0.0),
    (None, 2, 0.0),
    (None, 0, 1.0),
    (Some("B"), 0, 1.0))
  val expectedOrdVectors: Map[Int, Vector] = Map(0 -> Vectors.dense(0.0,0.0), 1 -> Vectors.dense(1.0,0.0), 2 -> Vectors.dense(0.0,1.0))
  val expectedStrVectors: Map[String, Vector] = Map("A" -> Vectors.dense(0.0,1.0), "B" -> Vectors.dense(0.0,0.0), "null" -> Vectors.dense(1.0,0.0))

  test("BinaryEncoder should encode A,B,None to vector of size 2 after stringIndexer") {
    val dataframe = data.toDF(columns:_*)
    val stringIndexer = new StringIndexer().setInputCol(columns(0)).setOutputCol(columns(0) + "idx").setHandleInvalid("keep")
    val binaryEncoder = new BinaryEncoder().setInputCol(stringIndexer.getOutputCol).setOutputCol(outputCol)

    val expectedVectorSize = math.ceil(math.log(dataframe.select(columns(0)).distinct().count())/math.log(2.0))

    val pl = new Pipeline().setStages(Array(stringIndexer, binaryEncoder))
    val transformedDf = pl.fit(dataframe).transform(dataframe)
    transformedDf.select(col(columns(0)).cast(StringType), col(outputCol)).collect.map {
      case Row(cat: String, enc: Vector) =>
        assert(expectedVectorSize == enc.size)
        assert(expectedStrVectors(cat) == enc)
      case Row(null, enc: Vector) =>
        assert(expectedVectorSize == enc.size)
        assert(expectedStrVectors("null") == enc)
    }
  }

  test("BinaryEncoder should encode 0,1,2 to vector of size 2 after stringIndexer") {
    val dataframe = data.toDF(columns:_*)
    val stringIndexer = new StringIndexer().setInputCol(columns(1)).setOutputCol(columns(1) + "idx").setHandleInvalid("keep")
    val binaryEncoder = new BinaryEncoder().setInputCol(stringIndexer.getOutputCol).setOutputCol(outputCol)

    val expectedVectorSize = math.ceil(math.log(dataframe.select(columns(1)).distinct().count())/math.log(2.0))

    val pl = new Pipeline().setStages(Array(stringIndexer, binaryEncoder))
    val transformedDf = pl.fit(dataframe).transform(dataframe)
    transformedDf.select(col(columns(1)).cast(IntegerType), col(outputCol)).collect.map {
      case Row(cat: Int, enc: Vector) =>
        assert(expectedVectorSize == enc.size)
        assert(expectedOrdVectors(cat) == enc)
    }
  }

}
