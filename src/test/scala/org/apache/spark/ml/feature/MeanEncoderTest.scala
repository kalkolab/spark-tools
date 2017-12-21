package org.apache.spark.ml.feature

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StringType
import org.scalactic.TolerantNumerics
import org.scalatest.FunSuite

class MeanEncoderTest extends FunSuite with DataFrameSuiteBase {

  override def enableHiveSupport: Boolean = false
  override def reuseContextIfPossible: Boolean = true
  import spark.implicits._
  implicit val doubleEq = TolerantNumerics.tolerantDoubleEquality(1e-3)

  val columns = Array("string", "ordinal", "label")
  val outputCol = "targetMean"
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
  val expectedOrdMeans: Map[String, Double] = Map("1" -> .5, "0" -> 0.75, "2" -> 1.0/3)
  val expectedStrMeans: Map[String, Double] = Map("A" -> 0.5, "B" -> 0.75, "null" -> 1.0/3)

  test("MeanEncoder should create model with correct means") {
    val dataframe = data.toDF(columns:_*)
    val meanEncoder = new MeanEncoder().setInputCol(columns(1)).setLabelCol(columns(2)).setOutputCol(outputCol)

    val model = meanEncoder.fit(dataframe)
    assertResult(expectedOrdMeans)(model.targetMeans)
  }

  test ("MeanEncoder should correctly transform nominal features to means") {
    val dataframe = data.toDF(columns:_*)
    val meanEncoder = new MeanEncoder().setInputCol(columns(1)).setLabelCol(columns(2)).setOutputCol(outputCol)

    val transformedDf = meanEncoder.fit(dataframe).transform(dataframe)
    transformedDf.select(col(columns(1)).cast(StringType), col(outputCol)).collect.map {
      case Row(cat: String, mean: Double) => assert(expectedOrdMeans(cat) === mean)
    }
  }

  test ("MeanEncoder should correctly transform string features to means") {
    val dataframe = data.toDF(columns:_*)
    val meanEncoder = new MeanEncoder().setInputCol(columns(0)).setLabelCol(columns(2)).setOutputCol(outputCol)

    val transformedDf = meanEncoder.fit(dataframe).transform(dataframe)
    transformedDf.select(columns(0), outputCol).collect.map {
      case Row(cat: String, mean: Double) => assert(expectedStrMeans(cat) === mean)
      case Row(null, mean: Double) => assert(expectedStrMeans("null") === mean)
    }
  }

}
