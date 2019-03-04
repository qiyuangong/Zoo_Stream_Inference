package com.intel.analytics.zoo.apps.streaming

import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.text.{TextFeature, TextSet}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import scopt.OptionParser


case class TextClassificationParams(
  embeddingPath: String = "./",
  classNum: Int = 20, tokenLength: Int = 200,
  sequenceLength: Int = 500, encoder: String = "cnn",
  encoderOutputDim: Int = 256, maxWordsNum: Int = 5000,
  trainingSplit: Double = 0.8, batchSize: Int = 128,
  nbEpoch: Int = 20, learningRate: Double = 0.01,
  partitionNum: Int = 4, model: Option[String] = None)

object StreamingTextClassification {

  def main(args: Array[String]) {
    val parser = new OptionParser[TextClassificationParams]("TextClassification Example") {
      opt[String]("embeddingPath")
        .required()
        .text("The directory for GloVe embeddings")
        .action((x, c) => c.copy(embeddingPath = x))
      opt[Int]("classNum")
        .text("The number of classes to do classification")
        .action((x, c) => c.copy(classNum = x))
      opt[Int]("partitionNum")
        .text("The number of partitions to cut the dataset into")
        .action((x, c) => c.copy(partitionNum = x))
      opt[Int]("tokenLength")
        .text("The size of each word vector, 50 or 100 or 200 or 300 for GloVe")
        .action((x, c) => c.copy(tokenLength = x))
      opt[Int]("sequenceLength")
        .text("The length of each sequence")
        .action((x, c) => c.copy(sequenceLength = x))
      opt[Int]("maxWordsNum")
        .text("The maximum number of words to be taken into consideration")
        .action((x, c) => c.copy(maxWordsNum = x))
      opt[String]("encoder")
        .text("The encoder for the input sequence, cnn or lstm or gru")
        .action((x, c) => c.copy(encoder = x))
      opt[Int]("encoderOutputDim")
        .text("The output dimension of the encoder")
        .action((x, c) => c.copy(encoderOutputDim = x))
      opt[Int]('b', "batchSize")
        .text("The number of samples per gradient update")
        .action((x, c) => c.copy(batchSize = x))
      opt[String]('m', "model")
        .text("Model snapshot location if any")
        .action((x, c) => c.copy(model = Some(x)))
    }

    parser.parse(args, TextClassificationParams()).map { param =>
      val sc = NNContext.initNNContext("Network Text Streaming Predict")
      val ssc = new StreamingContext(sc, Seconds(3))

      val model = TextClassifier.loadModel[Float](param.model.get)
      // Create a socket stream on target ip:port and count the
      // words in input stream of \n delimited text (eg. generated by 'nc')
      // Note that no duplication in storage level only for running locally.
      // Replication necessary in distributed scenario for fault tolerance.
      val lines = ssc.socketTextStream("localhost", 9999, StorageLevel.MEMORY_AND_DISK_SER)

      lines.foreachRDD { lineRdd =>
        if(!lineRdd.partitions.isEmpty) {
          // RDD to TextFeature
          println("First line " + lineRdd.top(1).apply(0))
          val textFeature = lineRdd.map(x => {
            val feature = TextFeature.apply(x)
            val tensor = Tensor(500).zero()
            feature(TextFeature.sample) = Sample(tensor)
            feature
          })
          // RDD[TextFeature] to TextSet
          val dataSet = TextSet.rdd(textFeature)
          // Predict
          val predictSet = model.predict(dataSet, batchPerThread = param.partitionNum)
          // Print result
          predictSet.toDistributed().rdd.take(5).map(_.getPredict.toTensor).foreach(println)
        }
      }
      ssc.start()
      ssc.awaitTermination()
    }
  }
}
