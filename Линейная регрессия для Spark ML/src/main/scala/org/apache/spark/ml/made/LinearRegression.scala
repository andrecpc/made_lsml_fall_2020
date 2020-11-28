package org.apache.spark.ml.made

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, ParamMap}
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasMaxIter, HasPredictionCol, HasStepSize}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsReader, DefaultParamsWritable, DefaultParamsWriter, Identifiable, MLReadable, MLReader, MLWritable, MLWriter, MetadataUtils, SchemaUtils}
import org.apache.spark.mllib
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}
import org.apache.spark.sql.types.StructType

trait LinearRegressionParams extends HasLabelCol with HasFeaturesCol
  with HasPredictionCol {

  val learningRate:DoubleParam=new DoubleParam(this,"learningRate","learningRate")
  val nIters:IntParam=new IntParam(this,"nIters","nIters")

  setDefault(nIters,100)
  setDefault(learningRate,1.0)

  def setLabelCol(value: String) : this.type = set(labelCol, value)
  def setFeaturesCol(value: String): this.type = set(featuresCol, value)
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getFeaturesCol, new VectorUDT())

    if (schema.fieldNames.contains($(predictionCol))) {
      SchemaUtils.checkColumnType(schema, getPredictionCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getFeaturesCol).copy(name = getPredictionCol))
    }
  }
}

class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable  {

  def this() = this(Identifiable.randomUID("linearRegression"))

  def setNIters(value: Int): this.type = set(nIters, value)

  def setLR(value: Double): this.type = set(learningRate, value)

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder : Encoder[Vector] = ExpressionEncoder()

    val ones = dataset.withColumn("ones", lit(1))
    val assembler = new VectorAssembler()
      .setInputCols(Array($(featuresCol), "ones", $(labelCol)))
      .setOutputCol("features")
    val assembled: Dataset[Vector] = assembler
      .transform(ones)
      .select("features").as[Vector]

    val nFeatures: Int = MetadataUtils.getNumFeatures(dataset, $(featuresCol))
    var weights: breeze.linalg.DenseVector[Double] = breeze.linalg.DenseVector.rand[Double](nFeatures + 1)

    for (i <- 0 to $(nIters)) {
      val summary = assembled.rdd.mapPartitions((data: Iterator[Vector]) => {
        val summarizer = new MultivariateOnlineSummarizer()
        data.foreach(v => {
          val X = v.asBreeze(0 until weights.size).toDenseVector
          val y = v.asBreeze(weights.size)
          val grads = X * (breeze.linalg.sum(X * weights) - y)
          summarizer.add(mllib.linalg.Vectors.fromBreeze(grads))
        })
        Iterator(summarizer)
      }).reduce(_ merge _)

      weights = weights - $(learningRate) * summary.mean.asBreeze
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(weights(0 until weights.size - 1)).toDense,
      weights(weights.size - 1))
    ).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)
}

object LinearRegression extends DefaultParamsReadable[LinearRegression]

class LinearRegressionModel private[made](override val uid: String, val weights: DenseVector, val bias: Double)
  extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {

  private[made] def this(weights: DenseVector, bias: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), weights.toDense, bias)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights, bias))

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_predict",
      (x : Vector) => {
        Vectors.fromBreeze(breeze.linalg.DenseVector(weights.asBreeze.dot(x.asBreeze) + bias))
      })

    dataset.withColumn($(predictionCol), transformUdf(dataset($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      sqlContext.createDataFrame(Seq(weights -> bias)).write.parquet(path + "/vectors")
    }
  }

  object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
    override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
      override def load(path: String): LinearRegressionModel = {
        val metadata = DefaultParamsReader.loadMetadata(path, sc)

        val vectors = sqlContext.read.parquet(path + "/vectors")

        implicit val encoder: Encoder[Vector] = ExpressionEncoder()

        val weights = vectors.select(vectors("_1").as[Vector]).first()
        val bias = vectors.select(vectors("_2")).first().getDouble(0)

        val model = new LinearRegressionModel(weights.toDense, bias)
        metadata.getAndSetParams(model)
        model
      }
    }
  }
}