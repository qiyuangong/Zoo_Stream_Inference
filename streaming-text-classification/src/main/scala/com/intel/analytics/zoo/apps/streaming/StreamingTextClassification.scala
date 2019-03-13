/*
 * Copyright 2018 Analytics Zoo Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.zoo.apps.streaming

import scala.collection.immutable._
import scala.io.Source
import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.feature.text.{TextFeature, TextSet}
import com.intel.analytics.zoo.models.textclassification.TextClassifier
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import scopt.OptionParser


case class TextClassificationParams(
  host: String = "localhost",
  port: Int = 9999,
  indexPath: String = "textclassification.txt",
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
      opt[String]('h', "host")
        .text("host for network connection")
        .action((x, c) => c.copy(host = x))
      opt[Int]('p', "port")
        .text("Port for network connection")
        .action((x, c) => c.copy(port = x))
      opt[String]("indexPath")
        .text("Path of word to index text file")
        .action((x, c) => c.copy(indexPath = x))
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

      val wordIndex: Map[String, Int] = readWordIndex(param.indexPath)

      val model = TextClassifier.loadModel[Float](param.model.get)
      // Create a socket stream on target ip:port and count the
      // words in input stream of \n delimited text (eg. generated by 'nc')
      // Note that no duplication in storage level only for running locally.
      // Replication necessary in distributed scenario for fault tolerance.
      val lines = ssc.socketTextStream(param.host,
        param.port, StorageLevel.MEMORY_AND_DISK_SER)

      lines.foreachRDD { lineRdd =>
        if (!lineRdd.partitions.isEmpty) {
          // RDD to TextFeature
          println("First line " + lineRdd.top(1).apply(0))
          val textFeature = lineRdd.map(x => TextFeature.apply(x))
          // RDD[TextFeature] to TextSet
          val dataSet = TextSet.rdd(textFeature)
          // Pre-processing
          val transformed = dataSet.setWordIndex(wordIndex)
            .tokenize().normalize()
            .word2idx(removeTopN = 10, maxWordsNum = param.maxWordsNum)
            .shapeSequence(param.sequenceLength).generateSample()
          val predictSet = model.predict(transformed,
            batchPerThread = param.partitionNum)
          // Print result
          predictSet.toDistributed()
            .rdd.take(5)
            .map(_.getPredict.toTensor)
            .foreach(println)
        }
      }
      ssc.start()
      ssc.awaitTermination()
    }
  }

  def word2index(word: String, wordIndex: Map[String, Int]): Int = {
    wordIndex.apply(word)
  }

  def readWordIndex(path: String): Map[String, Int] = {
    Source.fromFile(path).getLines.map { x =>
      val token = x.split(" ")
      (token(0), token(1).toInt)
    }.toMap
  }
}
