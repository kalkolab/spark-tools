package org.apache.spark.ml.feature

import com.holdenkarau.spark.testing.DataFrameSuiteBase
import org.apache.spark.sql.Row
import org.scalatest.FunSuite

class IsMissingGeneratorTest extends FunSuite  with DataFrameSuiteBase {

  override def enableHiveSupport: Boolean = false
  override def reuseContextIfPossible: Boolean = true
  import spark.implicits._

  test("generate new double columns to reflect missing values") {
    val rows = Seq(
      (0.0, 0.72, -0.73, 0.0, 0.0, 0.37, Some(-0.53)),
      (0.0, 0.0, -1.2, 0.0, 0.0, -0.43, None),
      (0.0, 0.0, -1.3, Double.NaN, 0.0, -0.56, None),
      (0.0, 0.0, -1.3, 0.0, 0.0, -0.57, None),
      (0.0, Double.NaN, -1.3, 0.0, 0.0, -0.57, Some(-0.5)))

    val columns = Array("col1", "col2", "col3", "col4", "col5", "col6", "col7")
    val df = rows.toDF(columns:_*)

    val generator = new IsMissingGenerator().setInputCols(columns).setOutputCols(columns.map(_+"_missing"))

    val newDf = generator.transform(df)

    assertResult(columns.length * 2)(newDf.columns.length)
    assert(columns.forall(newDf.columns.contains))

    columns.foreach {
      col => newDf.select(col, col + "_missing").collect.foreach {
        case Row(v: Double, missing: Double) =>
          if (v.isNaN) assert(missing == 1.0, "missing col should be 1.0 for missing values") else require(missing == 0.0, "missing col should be 1.0 for existing values")
        case Row(null, missing: Double) => require(missing==1.0, "missing col should be 1.0 for missing values")
      }
    }
  }

}