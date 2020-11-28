package org.apache.spark.ml.made

import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.stats.mean
import com.google.common.io.Files
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.DataFrame
import org.scalatest.flatspec._
import org.scalatest.matchers._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta: Double = 0.01
  val coefs: DenseVector[Double] = LinearRegressionTest._coefs
  val bias: Double  = LinearRegressionTest._bias
  val y: DenseVector[Double] = LinearRegressionTest._y
  val data: DataFrame = LinearRegressionTest._data

  "Estimator" should "calculate" in {
    val estimator: LinearRegression = new LinearRegression()
      .setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")
      .setNIters(100)
      .setLR(1.0)

    val model = estimator.fit(data)

    model.bias should be(bias +- delta)
    model.weights(0) should be(coefs(0) +- delta)
    model.weights(1) should be(coefs(1) +- delta)
    model.weights(2) should be(coefs(2) +- delta)
  }

  "Model" should "predict" in {
    val model: LinearRegressionModel = new LinearRegressionModel(
      weights = Vectors.fromBreeze(coefs).toDense,
      bias = bias
    ).setFeaturesCol("features")
      .setLabelCol("y")
      .setPredictionCol("prediction")

    val pred = DenseVector(model.transform(data).select("prediction").collect().map(x => x.getAs[Vector](0)(0)))

    sqrt(mean(pow(pred - y, 2))) should be(0.0 +- delta)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("y")
        .setPredictionCol("prediction")
        .setNIters(100)
        .setLR(1.0)
    ))

    val tmpFolder = Files.createTempDir()
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    model.bias should be(bias +- delta)
    model.weights(0) should be(coefs(0) +- delta)
    model.weights(1) should be(coefs(1) +- delta)
    model.weights(2) should be(coefs(2) +- delta)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setFeaturesCol("features")
        .setLabelCol("label")
        .setPredictionCol("prediction")
        .setNIters(100)
        .setLR(1.0)
    ))

    val model = pipeline.fit(data)
    val coefs = model.stages(0).asInstanceOf[LinearRegressionModel].weights
    val bias = model.stages(0).asInstanceOf[LinearRegressionModel].bias

    val tmpFolder = Files.createTempDir()
    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val loaded_model = PipelineModel.load(tmpFolder.getAbsolutePath).stages(0).asInstanceOf[LinearRegressionModel]

    loaded_model.bias should be(bias +- delta)
    loaded_model.weights(0) should be(coefs(0) +- delta)
    loaded_model.weights(1) should be(coefs(1) +- delta)
    loaded_model.weights(2) should be(coefs(2) +- delta)

    val pred = DenseVector(loaded_model.transform(data).select("prediction").collect().map(x => x.getAs[Vector](0)(0)))
    sqrt(mean(pow(pred - y, 2))) should be(0.0 +- delta)
  }
}

object LinearRegressionTest extends WithSpark {

  lazy val _X: DenseMatrix[Double] = DenseMatrix.rand(10000, 3)
  lazy val _coefs: DenseVector[Double] = DenseVector(0.5, -0.1, 0.2)
  lazy val _bias: Double = 1.2
  lazy val _y: DenseVector[Double] = _X * _coefs + _bias + DenseVector.rand(10000) * 0.0001

  lazy val _data: DataFrame = {
    import sqlc.implicits._

    val tmp = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)
    val df = tmp(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq
      .toDF("x1", "x2", "x3", "y")

    val assembler = new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")
    val out = assembler.transform(df).select("features", "y")

    out
  }

}