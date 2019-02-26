package com.intel.analytics.zoo.apps.streaming

import java.nio.file.Paths

import com.intel.analytics.bigdl.transform.vision.image.ImageFeature
import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.objectdetection.{ObjectDetector, Visualizer}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import scopt.OptionParser

object StreamingObjectDetection {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(image: String = "",
                          outputFolder: String = "data/demo",
                          modelPath: String = "",
                          nPartition: Int = 1)

  val parser = new OptionParser[PredictParam]("Analytics Zoo Object Detection Demo") {
    head("Analytics Zoo Object Detection Demo")
    opt[String]('i', "image")
      .text("where you put the demo image data, can be image folder or image path")
      .action((x, c) => c.copy(image = x))
      .required()
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]("modelPath")
      .text("Analytics Zoo model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
      .required()
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictParam()).foreach { params =>
      val sc = NNContext.initNNContext("Streaming Object Detection")

      val ssc = new StreamingContext(sc.getConf, Seconds(3))
      val model = ObjectDetector.loadModel[Float](params.modelPath)
      // Read image stream from HDFS
      val fStream = ssc.fileStream()
//      val data = ImageSet.read(params.image, sc, params.nPartition,
//        imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)

      fStream.foreachRDD(imgRdd => {
        // RDD to TextFeature
        val imgFeature = imgRdd.map(x => ImageFeature.apply(x))
        // RDD[TextFeature] to TextSet
        val dataSet = ImageSet.rdd(imgFeature)
        // Predict
        val output = model.predictImageSet(dataSet)
        // Print result
        val visualizer = Visualizer(model.getConfig.labelMap, encoding = "jpg")
        val visualized = visualizer(output).toDistributed()
        val result = visualized.rdd.map(imageFeature =>
          (imageFeature.uri(), imageFeature[Array[Byte]](Visualizer.visualized))).collect()
        result.foreach(x => {
          Utils.saveBytes(x._2, getOutPath(params.outputFolder, x._1, "jpg"), true)
        })
        logger.info(s"labeled images are saved to ${params.outputFolder}")
      })





    }
  }

  private def getOutPath(outPath: String, uri: String, encoding: String): String = {
    Paths.get(outPath,
      s"detection_${ uri.substring(uri.lastIndexOf("/") + 1,
        uri.lastIndexOf(".")) }.${encoding}").toString
  }
}
